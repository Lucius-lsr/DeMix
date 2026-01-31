import json
import os
import yaml
import datetime

MIX_JSON_FILE = "PATH_TO_MIX_JSON"
MIX_NAMES = [f"mix_{i}" for i in range(16)]
SUFFIX = "/checkpoint-12800"
EXP_NAME = "EXP_NAME"
MODEL_SAVE_PATH = f"PATH_TO_MODEL_SAVE_DIR{SUFFIX if SUFFIX else '/final'}"

PURE_MODEL_DICT = {
    "general_target": f"PATH_TO_PURE_MODEL/general_target{SUFFIX}",
    "math_very_high": f"PATH_TO_PURE_MODEL/math_very_high{SUFFIX}",
    "math_high": f"PATH_TO_PURE_MODEL/math_high{SUFFIX}",
    "math_medium": f"PATH_TO_PURE_MODEL/math_medium{SUFFIX}",
    "code_very_high": f"PATH_TO_PURE_MODEL/code_very_high{SUFFIX}",
    "code_high": f"PATH_TO_PURE_MODEL/code_high{SUFFIX}",
    "code_medium": f"PATH_TO_PURE_MODEL/code_medium{SUFFIX}",
}

YAML_SAVE_PATH = os.path.join("PATH_TO_YAML_OUTPUT_DIR", EXP_NAME)
BASE_MODEL_PATH = "PATH_TO_BASE_MODEL"

MERGE_CONFIGS = {
    "linear": {
        "params": {},
        "has_base": False,
        "has_weight": True,
    },
}


def generate_yamls():
    if not os.path.exists(MIX_JSON_FILE):
        print(f"Error: JSON file not found at {MIX_JSON_FILE}")
        return

    if not os.path.exists(YAML_SAVE_PATH):
        os.makedirs(YAML_SAVE_PATH)

    with open(MIX_JSON_FILE, "r", encoding="utf-8") as f:
        mix_data_full = json.load(f)

    current_dt = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    for mix_name, mix_ratios in mix_data_full.items():
        if mix_name not in MIX_NAMES:
            continue

        total_weight = 0.0
        for model_key, ratio_value in mix_ratios.items():
            if model_key in PURE_MODEL_DICT:
                total_weight += float(ratio_value)

        for method_key, config in MERGE_CONFIGS.items():
            models_list = []

            for model_key, ratio_value in mix_ratios.items():
                if model_key not in PURE_MODEL_DICT:
                    continue

                model_path = PURE_MODEL_DICT[model_key]
                model_entry = {"model": model_path}

                if config["has_weight"]:
                    raw_val = float(ratio_value)
                    normalized_val = raw_val / total_weight if total_weight != 0 else 0.0
                    parameters = {"weight": normalized_val}

                    if config["params"]:
                        parameters.update(config["params"])

                    model_entry["parameters"] = parameters

                models_list.append(model_entry)

            actual_merge_method = config.get("method_name_override", method_key)

            yaml_content = {
                "prefix": f"Qwen3-1.7B_{mix_name}_{method_key}",
                "dt": current_dt,
                "save_path": os.path.join(MODEL_SAVE_PATH, mix_name),
                "models": models_list,
                "base_model": BASE_MODEL_PATH if config["has_base"] else None,
                "merge_method": actual_merge_method,
                "dtype": "bfloat16",
            }

            if not config["has_base"]:
                del yaml_content["base_model"]

            file_name = f"Qwen3-1.7B{SUFFIX.replace('/', '_')}_{mix_name}_{method_key}.yaml"
            save_file_path = os.path.join(YAML_SAVE_PATH, file_name)

            with open(save_file_path, "w", encoding="utf-8") as yf:
                yaml.dump(
                    yaml_content,
                    yf,
                    default_flow_style=False,
                    sort_keys=False,
                    allow_unicode=True,
                )

    print(f"Done! All YAML files generated in {YAML_SAVE_PATH}")


if __name__ == "__main__":
    generate_yamls()
