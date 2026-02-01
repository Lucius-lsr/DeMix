import pandas as pd
from scipy import stats
import numpy as np
import random

BENCHMARKS = {
    'general': ['ARC-e', 'hellaswag', 'piqa', 'siqa', 'winogrande'],
    'code': ['MBPP', 'HUMANEVAL'],
    'math': ['GSM8K', 'MATH'],
}

def get_benchmark_data(checkpoint_result_dir):
    # get your own evaluation results from checkpoint_dir
    # example:
    index = int(checkpoint_result_dir.split('_')[-1])
    return {
        'ARC-e': random.random() + index,
        'hellaswag': random.random() + index,
        'piqa': random.random() + index,
        'siqa': random.random() + index,
        'winogrande': random.random() + index,
        'general_avg': random.random() + index,

        'MBPP': random.random() + index,
        'HUMANEVAL': random.random() + index,
        'code_avg': random.random() + index,

        'GSM8K': random.random() + index,
        'MATH': random.random() + index,
        'math_avg': random.random() + index,
    }



def get_top_spearman(pred_scores, gt_scores, top_k=0.25):
    if (pred_scores is None) or (gt_scores is None):
        return 0.0
    if len(pred_scores) != len(gt_scores) or len(gt_scores) < 2:
        return 0.0

    n = len(gt_scores)
    k = max(2, int(n * top_k)) 
    k = min(k, n)

    gt_arr = np.asarray(gt_scores, dtype=float)
    pred_arr = np.asarray(pred_scores, dtype=float)

    top_idx = np.argsort(gt_arr)[-k:]

    rho, _ = stats.spearmanr(pred_arr[top_idx], gt_arr[top_idx])
    if pd.isna(rho):
        return 0.0
    return float(rho)

def eval(pred_data_dict, gt_data_dict):
    mixture_ids = list(pred_data_dict.keys())
    if not mixture_ids or pred_data_dict[mixture_ids[0]] is None:
        print("No valid prediction data found to evaluate.")
        return

    benchmarks = list(pred_data_dict[mixture_ids[0]].keys())
    
    rho_list = []
    rho_domain_dict = {}

    top_25_rho_list = []
    top_25_rho_domain_dict = {}
    
    maintain_list = []
    maintain_domain_dict = {}
    
    for benchmark in benchmarks:
        gt_scores = []
        pred_scores = []
        
        for mixture_id in mixture_ids:
            if pred_data_dict[mixture_id] and gt_data_dict[mixture_id]:
                gt_scores.append(gt_data_dict[mixture_id].get(benchmark, 0))
                pred_scores.append(pred_data_dict[mixture_id].get(benchmark, 0))
        
        if len(gt_scores) < 2:
            continue

        rho, _ = stats.spearmanr(pred_scores, gt_scores)
        top_25_rho = get_top_spearman(pred_scores, gt_scores)
        
        # 处理 NaN 情况 (例如分数完全一样导致方差为0)
        if pd.isna(rho):
            rho = 0.0

        rho_list.append(rho)
        top_25_rho_list.append(top_25_rho)
        maintain_list.append(sum(pred_scores) / sum(gt_scores))

        if '_avg' in benchmark:
            rho_domain_dict[benchmark] = rho
            top_25_rho_domain_dict[benchmark] = top_25_rho
            maintain_domain_dict[benchmark] = sum(pred_scores) / sum(gt_scores)
        else:
            pass

    rho_domain_dict['avg'] = sum(rho_domain_dict.values()) / len(rho_domain_dict)
    top_25_rho_domain_dict['avg'] = sum(top_25_rho_domain_dict.values()) / len(top_25_rho_domain_dict)
    maintain_domain_dict['avg'] = sum(maintain_domain_dict.values()) / len(maintain_domain_dict)

    return rho_domain_dict, top_25_rho_domain_dict, maintain_domain_dict

if __name__ == "__main__":
    RANGE = range(16)
    GT_DIRS = {str(i): f"PATH_TO_GT_RESULT_DIR_{i}" for i in RANGE}
    MIXTURE_CHECKPOINT_DIRS = {str(i): f"PATH_TO_PROXY_RESULT_DIR_{i}" for i in RANGE}   

    pred_data_dict = {}
    for model_name, path in MIXTURE_CHECKPOINT_DIRS.items():
        data = get_benchmark_data(path)
        if data:
            pred_data_dict[model_name] = data
        else:
            print(f"Warning: Missing Prediction data for {model_name} in {path}")

    gt_data_dict = {}
    for model_name, path in GT_DIRS.items():
        data = get_benchmark_data(path)
        if data:
            gt_data_dict[model_name] = data
        else:
            print(f"Warning: Missing GT data for {model_name} in {path}")

    common_keys = set(pred_data_dict.keys()) & set(gt_data_dict.keys())
    if len(common_keys) < 2:
        raise ValueError(f"Not enough matching data points to calculate correlation")

    pred_filtered = {k: pred_data_dict[k] for k in common_keys}
    gt_filtered = {k: gt_data_dict[k] for k in common_keys}
    rho_domain_dict, rho_25_domain_dict, maintain_domain_dict = eval(pred_filtered, gt_filtered)

    print(rho_domain_dict)
    print(rho_25_domain_dict)
    print(maintain_domain_dict)