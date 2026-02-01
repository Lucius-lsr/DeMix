import json
import os
import yaml
import datetime

# ================================== settings ==================================
MIX_JSON_FILE = "MIX_JSON_FILE"
PATH_TO_COMPONENT_MODEL_DIR = "PATH_TO_COMPONENT_MODEL_DIR"
MERGED_MODEL_SAVE_ROOT = "MERGED_MODEL_SAVE_ROOT"
EXP_NAME = "merged_30B"
# SUFFIX = "/checkpoint-500" # for 2B token budget
# SUFFIX = "/checkpoint-2500" # for 10B token budget
SUFFIX = "/checkpoint-7500" # for 30B token budget
# SUFFIX = "/checkpoint-12500" # for 50B token budget
PATH_TO_YAML_OUTPUT_DIR = "PATH_TO_YAML_OUTPUT_DIR"
BASE_MODEL_PATH = None # or path_to_base_model if you need to try other merge methods that require a base model
MERGE_CONFIGS = {
    "linear": {
        "params": {},
        "has_base": False,
        "has_weight": True,
    },
    # "multislerp_base": {
    #     "params": {},
    #     "has_base": True,
    #     "has_weight": True,
    #     "method_name_override": "multislerp"
    # },
    # "multislerp": {
    #     "params": {},
    #     "has_base": False,
    #     "has_weight": True
    # },
    # "task_arithmetic": {
    #     "params": {},
    #     "has_base": True,
    #     "has_weight": True
    # },
    # "karcher": {
    #     "params": {},
    #     "has_base": False,
    #     "has_weight": True
    # },
    # "breadcrumbs_ties": {
    #     "params": {"gamma": 0.01, "density": 0.9},
    #     "has_base": True,
    #     "has_weight": True
    # },
    # "breadcrumbs": {
    #     "params": {"gamma": 0.01, "density": 0.9},
    #     "has_base": True,
    #     "has_weight": True
    # },
    # "dare_linear": {
    #     "params": {"density": 0.9},
    #     "has_base": True,
    #     "has_weight": True
    # },
    # "dare_ties": {
    #     "params": {"density": 0.9},
    #     "has_base": True,
    #     "has_weight": True
    # },
    # "della_linear": {
    #     "params": {"density": 0.9, "epsilon": 0.01},
    #     "has_base": True,
    #     "has_weight": True
    # },
    # "della": {
    #     "params": {"density": 0.9, "epsilon": 0.01},
    #     "has_base": True,
    #     "has_weight": True
    # },
    # "ties": {
    #     "params": {"density": 0.9},
    #     "has_base": True,
    #     "has_weight": True
    # }
}

component_model_dict = {
    "general_target": f"{PATH_TO_COMPONENT_MODEL_DIR}/Qwen3-1.7B-Mid_pt_pure_mix_general_target{SUFFIX}",
    "math_very_high": f"{PATH_TO_COMPONENT_MODEL_DIR}/Qwen3-1.7B-Mid_pt_pure_mix_math_very_high{SUFFIX}",
    "math_high": f"{PATH_TO_COMPONENT_MODEL_DIR}/Qwen3-1.7B-Mid_pt_pure_mix_math_high{SUFFIX}",
    "math_medium": f"{PATH_TO_COMPONENT_MODEL_DIR}/Qwen3-1.7B-Mid_pt_pure_mix_math_medium{SUFFIX}",
    "code_very_high": f"{PATH_TO_COMPONENT_MODEL_DIR}/Qwen3-1.7B-Mid_pt_pure_mix_code_very_high{SUFFIX}",
    "code_high": f"{PATH_TO_COMPONENT_MODEL_DIR}/Qwen3-1.7B-Mid_pt_pure_mix_code_high{SUFFIX}",
    "code_medium": f"{PATH_TO_COMPONENT_MODEL_DIR}/Qwen3-1.7B-Mid_pt_pure_mix_code_medium{SUFFIX}",
}
yaml_save_path = os.path.join(PATH_TO_YAML_OUTPUT_DIR, EXP_NAME)
model_save_path = os.path.join(MERGED_MODEL_SAVE_ROOT, EXP_NAME)
mix_names = [f"mix_{i}" for i in range(16)]

def generate_yamls():
    if not os.path.exists(MIX_JSON_FILE):
        print(f"Error: JSON file not found at {MIX_JSON_FILE}")
        return

    if not os.path.exists(yaml_save_path):
        os.makedirs(yaml_save_path)

    with open(MIX_JSON_FILE, "r", encoding="utf-8") as f:
        mix_data_full = json.load(f)

    current_dt = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    for mix_name, mix_ratios in mix_data_full.items():
        if mix_name not in mix_names:
            continue

        total_weight = 0.0
        for model_key, ratio_value in mix_ratios.items():
            if model_key in component_model_dict:
                total_weight += float(ratio_value)

        for method_key, config in MERGE_CONFIGS.items():
            models_list = []

            for model_key, ratio_value in mix_ratios.items():
                if model_key not in component_model_dict:
                    continue

                model_path = component_model_dict[model_key]
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
                "save_path": os.path.join(model_save_path, mix_name),
                "models": models_list,
                "base_model": BASE_MODEL_PATH if config["has_base"] else None,
                "merge_method": actual_merge_method,
                "dtype": "bfloat16",
            }

            if not config["has_base"]:
                del yaml_content["base_model"]

            file_name = f"Merge{SUFFIX.replace('/', '_')}_{mix_name}_{method_key}.yaml"
            save_file_path = os.path.join(yaml_save_path, file_name)

            with open(save_file_path, "w", encoding="utf-8") as yf:
                yaml.dump(
                    yaml_content,
                    yf,
                    default_flow_style=False,
                    sort_keys=False,
                    allow_unicode=True,
                )

    print(f"Done! All YAML files generated in {yaml_save_path}")


if __name__ == "__main__":
    generate_yamls()
