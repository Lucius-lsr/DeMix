import numpy as np
import lightgbm as lgb
import pandas as pd
import os
import glob
import json
from scipy.stats import spearmanr
import warnings
from sklearn.linear_model import Ridge

warnings.filterwarnings("ignore", category=UserWarning)

BENCHMARKS = {
    "general": [("ARC-e", 2), ("hellaswag", 4), ("piqa", 7), ("siqa", 8), ("winogrande", 9)],
    "code": [("MBPP", 14), ("HUMANEVAL", 15)],
    "math": [("GSM8K", 11), ("MATH", 12)],
}


def get_benchmark_data(checkpoint_dir, benchmarks):
    if "*" in checkpoint_dir:
        matched_dirs = glob.glob(checkpoint_dir)
        if not matched_dirs:
            print(f"Warning: No directory found matching pattern {checkpoint_dir}")
            return None
        checkpoint_dir = matched_dirs[0]

    target_csv_pattern = os.path.join(checkpoint_dir, "compass_outputs/*/summary/*.csv")
    csv_files = glob.glob(target_csv_pattern)

    if not csv_files:
        scores = {}
        for category, items in benchmarks.items():
            for name, _loc in items:
                scores[name] = None
            scores[f"{category}_avg"] = None
        return scores

    target_csv_file = csv_files[-1]
    try:
        df = pd.read_csv(target_csv_file, on_bad_lines="skip", engine="python")

        scores = {}
        for category, items in benchmarks.items():
            category_scores = []
            for name, loc in items:
                score = float(df.iloc[loc, 4])
                scores[name] = score
                category_scores.append(score)
            if category_scores:
                scores[f"{category}_avg"] = sum(category_scores) / len(category_scores)
        return scores

    except Exception as e:
        print(f"  -> Error processing file {target_csv_file}: {e}")
        return None


def train_regressor(X, Y, benchmarks, train_ratio=1.0, use_linear=False):
    num_samples = X.shape[0]
    num_targets = Y.shape[1]
    num_train = int(num_samples * train_ratio)

    X_train = X[:num_train]
    Y_train = Y[:num_train]
    X_test = X[num_train:] if num_train < num_samples else None
    Y_test = Y[num_train:] if num_train < num_samples else None

    reg_list = []
    for i in range(num_targets):
        if not use_linear:
            hyper_params = {
                "task": "train",
                "boosting_type": "gbdt",
                "objective": "regression",
                "metric": ["l1", "l2"],
                "num_iterations": 300,
                "seed": 42,
                "learning_rate": 1e-2,
                "verbosity": -1,
            }
            print("Using LightGBM")
            np.random.seed(42)
            gbm = lgb.LGBMRegressor(**hyper_params)
            reg = gbm.fit(X_train, Y_train[:, i])
        else:
            print("Using Ridge Regression")
            reg = Ridge(alpha=0.5)
            reg.fit(X_train, Y_train[:, i])

        if X_test is not None:
            r, _p = spearmanr(reg.predict(X_test), Y_test[:, i])
            print(benchmarks[i], "Correlation: {}".format(np.round(r * 100, 2)))

        reg_list.append(reg)

    return reg_list


def get_y(root_path, num_sample, benchmarks=BENCHMARKS):
    rng = range(num_sample)
    mixture_checkpoint_dirs = {str(i): f"{root_path}/mix_{i}/*" for i in rng}

    pred_data_dict = {}
    for model_name, path in mixture_checkpoint_dirs.items():
        data = get_benchmark_data(path, benchmarks)
        pred_data_dict[model_name] = data

    Y = np.array([list(data.values()) for data in pred_data_dict.values()])
    benchmarks_list = list(pred_data_dict[list(pred_data_dict.keys())[0]].keys())
    return Y, benchmarks_list


def get_y_regmix(root_path, rng, step, benchmarks=BENCHMARKS):
    mixture_checkpoint_dirs = {
        str(i): f"{root_path}/MODEL_MIX_{i}_*/checkpoint-{step}" for i in rng
    }

    pred_data_dict = {}
    for model_name, path in mixture_checkpoint_dirs.items():
        data = get_benchmark_data(path, benchmarks)
        pred_data_dict[model_name] = data

    Y = np.array([list(data.values()) for data in pred_data_dict.values()])
    benchmarks_list = list(pred_data_dict[list(pred_data_dict.keys())[0]].keys())
    return Y, benchmarks_list


def get_x(json_file, num_sample, sample_range=None):
    json_data = json.load(open(json_file, "r"))
    X = []
    cur_range = range(num_sample) if sample_range is None else sample_range
    for i in cur_range:
        data = json_data[f"mix_{i}"]
        x = [data[k] for k in data.keys()]
        sum_x = sum(x)
        normalized_x = [t / sum_x for t in x]
        X.append(normalized_x)
    return np.array(X)


def select_best_samples(
    reg_list,
    benchmarks_list,
    num_overall,
    num_best,
    num_selected,
    massive_json_path,
    save_path,
    avg=False,
):
    X_massive = get_x(massive_json_path, num_overall)

    print(f"Predicting on {X_massive.shape[0]} samples...")
    Y_pred_massive = np.zeros((X_massive.shape[0], len(reg_list)))
    for i, reg in enumerate(reg_list):
        Y_pred_massive[:, i] = reg.predict(X_massive)

    df_pred = pd.DataFrame(Y_pred_massive, columns=benchmarks_list)

    target_cols = ["general_avg", "math_avg", "code_avg"]
    for col in target_cols:
        if col not in df_pred.columns:
            raise ValueError(
                f"Key {col} not found in benchmarks. Available keys: {df_pred.columns.tolist()}"
            )

    df_pred["gen_pct"] = df_pred["general_avg"].rank(ascending=False, pct=True)
    df_pred["math_pct"] = df_pred["math_avg"].rank(ascending=False, pct=True)
    df_pred["code_pct"] = df_pred["code_avg"].rank(ascending=False, pct=True)

    df_pred["composite_score"] = (
        df_pred["gen_pct"] + df_pred["math_pct"] + df_pred["code_pct"]
    ) / 3.0

    top_best_df = df_pred.nsmallest(num_best, "composite_score")
    top_indices = top_best_df.index.tolist()

    print("Top 5 Selected Samples Stats:")
    print(top_best_df[["general_avg", "math_avg", "code_avg", "composite_score"]].head(5))

    with open(massive_json_path, "r") as f:
        massive_data = json.load(f)

    selected_data = {}

    if avg:
        assert num_selected == 1
        best_configs = [massive_data[f"mix_{idx}"] for idx in top_indices]
        keys = best_configs[0].keys()
        avg_config = {k: sum(c[k] for c in best_configs) / len(best_configs) for k in keys}
        selected_data["mix_0"] = avg_config
    else:
        top_indices_selected = np.random.choice(top_indices, num_selected, replace=False)
        for i, idx in enumerate(top_indices_selected):
            original_key = f"mix_{idx}"
            if original_key in massive_data:
                selected_data[f"mix_{i}"] = massive_data[original_key]
            else:
                print(f"Warning: Key {original_key} not found in source json.")

    os.makedirs(save_path, exist_ok=True)
    output_file = f"{save_path}/selected_{num_selected}_samples.json"
    with open(output_file, "w") as f:
        json.dump(selected_data, f, indent=4)

    print(f"\nSuccess! Selected {num_selected} samples from best {num_best} samples saved to {output_file}")


def filter_x_y(X, Y):
    valid_indices = ~np.any(Y == None, axis=1)
    X = X[valid_indices]
    Y = Y[valid_indices]
    print("X shape: ", X.shape, "Y shape: ", Y.shape)
    return X, Y


def demix(
    ori_samples_json,
    massive_samples_json,
    y_root_iter0,
    y_root_iter1,
    y_root_iter2,
    out_root,
):
    for first_train_samples in [32, 64, 128, 256, 512]:
        exp_name = f"exp_linear_30b_{int(1.75 * first_train_samples)}"

        X_0 = get_x(ori_samples_json, first_train_samples)
        Y_0, benchmarks_list = get_y(os.path.join(y_root_iter0, exp_name, "iter0"), first_train_samples)
        X_0, Y_0 = filter_x_y(X_0, Y_0)

        X_1 = get_x(
            os.path.join(out_root, exp_name, f"selected_{first_train_samples//2}_samples.json"),
            first_train_samples // 2,
        )
        Y_1, benchmarks_list = get_y(os.path.join(y_root_iter1, exp_name, "iter1"), first_train_samples // 2)
        X_1, Y_1 = filter_x_y(X_1, Y_1)

        if first_train_samples == 32:
            reg_list = train_regressor(X_1, Y_1, benchmarks_list, train_ratio=1, use_linear=True)
        else:
            X_2 = get_x(
                os.path.join(out_root, exp_name, f"selected_{first_train_samples//4}_samples.json"),
                first_train_samples // 4,
            )
            Y_2, benchmarks_list = get_y(os.path.join(y_root_iter2, exp_name, "iter2"), first_train_samples // 4)
            X_2, Y_2 = filter_x_y(X_2, Y_2)
            reg_list = train_regressor(
                np.concatenate([X_0, X_1, X_2]),
                np.concatenate([Y_0, Y_1, Y_2]),
                benchmarks_list,
                train_ratio=1.0,
            )

        select_best_samples(
            reg_list,
            benchmarks_list,
            200_000,
            1280,
            1,
            massive_samples_json,
            os.path.join(out_root, exp_name),
            avg=True,
        )


if __name__ == "__main__":
    seed = 42
    np.random.seed(seed)

    ORI_SAMPLES_JSON = "PATH/TO/ori_512_samples.json"
    MASSIVE_SAMPLES_JSON = "PATH/TO/20w_massive_samples.json"

    Y_ROOT_ITER0 = "PATH/TO/benchmark/iter0_root"
    Y_ROOT_ITER1 = "PATH/TO/benchmark/iter1_root"
    Y_ROOT_ITER2 = "PATH/TO/benchmark/iter2_root"

    Y_ROOT_REGMIX = "PATH/TO/benchmark/regmix_root"
    Y_ROOT_CLIMB = "PATH/TO/benchmark/climb_root"

    SELECTED_JSON_FROM_PREV = "PATH/TO/selected_samples.json"
    OUT_ROOT = "PATH/TO/output_dir"

    climb(
        ORI_SAMPLES_JSON,
        MASSIVE_SAMPLES_JSON,
        Y_ROOT_REGMIX,
        Y_ROOT_CLIMB,
        SELECTED_JSON_FROM_PREV,
        OUT_ROOT,
    )
