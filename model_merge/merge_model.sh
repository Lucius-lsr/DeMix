#!/bin/bash

CONFIG_FILE=""
SCRIPT_PATH="run_yaml.py"

MERGEKIT_ROOT=$(dirname "$(realpath "$SCRIPT_PATH")")/../..
PYTHONPATH_ADDITION=$(realpath "$MERGEKIT_ROOT")

REMAINING_ARGS=()

show_help() {
    echo "Usage: $0 <config.yml> [mergekit options...]"
    echo ""
    echo "Required:"
    echo "  config.yml    YAML config path (must contain prefix and dt fields)"
    echo ""
    echo "Other args will be passed to run_yaml.py"
    echo ""
    echo "Examples:"
    echo "  $0 config.yml"
    echo "  $0 config.yml --cuda --verbose"
    echo "  $0 config.yml --lazy-unpickle --allow-crimes"
    exit 1
}

read_yaml_field() {
    local config_file="$1"
    local field_name="$2"
    local value=""

    while IFS= read -r line; do
        if [[ "$line" =~ ^${field_name}:[[:space:]]*(.+)$ ]]; then
            value="${BASH_REMATCH[1]}"
            value=$(echo "$value" | sed 's/^[[:space:]]*["'\'']*//; s/["'\'']*[[:space:]]*$//')
            break
        fi
    done < "$config_file"

    echo "$value"
}

extract_model_paths() {
    local config_file="$1"
    local model_paths=()
    local in_models_section=false

    while IFS= read -r line; do
        if [[ "$line" =~ ^models:[[:space:]]*$ ]]; then
            in_models_section=true
            continue
        fi

        if [[ "$line" =~ ^[a-zA-Z] ]] && [ "$in_models_section" = true ]; then
            in_models_section=false
        fi

        if [ "$in_models_section" = true ]; then
            if [[ "$line" =~ ^[[:space:]]*-[[:space:]]*model:[[:space:]]*(.+)$ ]]; then
                path="${BASH_REMATCH[1]}"
            elif [[ "$line" =~ ^[[:space:]]+model:[[:space:]]*(.+)$ ]]; then
                path="${BASH_REMATCH[1]}"
            else
                continue
            fi

            path=$(echo "$path" | sed 's/^[[:space:]]*["'\'']*//; s/["'\'']*[[:space:]]*$//')

            if [[ -n "$path" && "$path" != "null" ]]; then
                model_paths+=("$path")
            fi
        fi
    done < "$config_file"

    printf '%s\n' "${model_paths[@]}"
}

copy_generation_config() {
    local config_file="$1"
    local output_dir="$2"

    local model_paths
    readarray -t model_paths < <(extract_model_paths "$config_file")

    if [ ${#model_paths[@]} -eq 0 ]; then
        return 0
    fi

    local copied=false
    for model_path in "${model_paths[@]}"; do
        if [ ! -d "$model_path" ]; then
            continue
        fi

        local gen_config_file="$model_path/generation_config.json"
        if [ -f "$gen_config_file" ]; then
            if cp "$gen_config_file" "$output_dir/generation_config.json"; then
                copied=true
                break
            fi
        fi
    done

    if [ "$copied" = false ]; then
        return 0
    fi
}

copy_additional_files() {
    local config_file="$1"
    local output_dir="$2"

    local model_paths
    readarray -t model_paths < <(extract_model_paths "$config_file")

    if [ ${#model_paths[@]} -eq 0 ]; then
        return 0
    fi

    local vocab_copied=false
    local chat_template_copied=false

    for model_path in "${model_paths[@]}"; do
        if [ ! -d "$model_path" ]; then
            continue
        fi

        if [ "$vocab_copied" = false ]; then
            local vocab_file="$model_path/vocab.json"
            if [ -f "$vocab_file" ]; then
                if cp "$vocab_file" "$output_dir/vocab.json"; then
                    vocab_copied=true
                fi
            fi
        fi

        if [ "$chat_template_copied" = false ]; then
            local chat_template_file="$model_path/chat_template.jinja"
            if [ -f "$chat_template_file" ]; then
                if cp "$chat_template_file" "$output_dir/chat_template.jinja"; then
                    chat_template_copied=true
                fi
            fi
        fi

        if [ "$vocab_copied" = true ] && [ "$chat_template_copied" = true ]; then
            break
        fi
    done
}

if [ $# -lt 1 ]; then
    show_help
fi

CONFIG_FILE="$1"
shift

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            ;;
        *)
            REMAINING_ARGS+=("$1")
            shift
            ;;
    esac
done

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: config file not found: $CONFIG_FILE"
    exit 1
fi

PREFIX=$(read_yaml_field "$CONFIG_FILE" "prefix")
DT=$(read_yaml_field "$CONFIG_FILE" "dt")
SAVE_PATH=$(read_yaml_field "$CONFIG_FILE" "save_path")

if [ -z "$PREFIX" ]; then
    echo "Error: 'prefix' not found in YAML"
    exit 1
fi

if [ -z "$DT" ]; then
    echo "Error: 'dt' not found in YAML"
    exit 1
fi

if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Error: python script not found: $SCRIPT_PATH"
    exit 1
fi

if [ ! -d "$SAVE_PATH" ]; then
    mkdir -p "$SAVE_PATH"
fi
OUTPUT_DIR="${SAVE_PATH}/Merged_${PREFIX}_${DT}"

CONFIG_ABS_PATH=$(realpath "$CONFIG_FILE")
CONFIG_FILENAME=$(basename "$CONFIG_FILE")

mkdir -p "$OUTPUT_DIR" || exit 1

START_TIMESTAMP=$(date +%s)
START_DATE_STR=$(date "+%Y-%m-%d %H:%M:%S")

PYTHONPATH="$PYTHONPATH_ADDITION${PYTHONPATH:+:$PYTHONPATH}" \
python "$SCRIPT_PATH" "${REMAINING_ARGS[@]}" "$CONFIG_ABS_PATH" "$OUTPUT_DIR"

if [ $? -eq 0 ]; then
    CONFIG_BASENAME="${CONFIG_FILENAME%.*}"
    CONFIG_EXT="${CONFIG_FILENAME##*.}"
    NEW_CONFIG_FILENAME="${CONFIG_BASENAME}_${DT}.${CONFIG_EXT}"
    cp "$CONFIG_ABS_PATH" "$OUTPUT_DIR/$NEW_CONFIG_FILENAME" || true

    copy_generation_config "$CONFIG_ABS_PATH" "$OUTPUT_DIR"
    copy_additional_files "$CONFIG_ABS_PATH" "$OUTPUT_DIR"

    END_TIMESTAMP=$(date +%s)
    END_DATE_STR=$(date "+%Y-%m-%d %H:%M:%S")
    DURATION=$((END_TIMESTAMP - START_TIMESTAMP))
    HOURS=$((DURATION / 3600))
    MINUTES=$(((DURATION % 3600) / 60))
    SECONDS_CALC=$((DURATION % 60))

    TIME_LOG_FILE="$OUTPUT_DIR/time.log"
    {
        echo "========================================="
        echo "Execution time log"
        echo "========================================="
        echo "Task: Merged_${PREFIX}_${DT}"
        echo "Config: $CONFIG_ABS_PATH"
        echo "-----------------------------------------"
        echo "Start: $START_DATE_STR"
        echo "End:   $END_DATE_STR"
        echo "Total: ${HOURS}h ${MINUTES}m ${SECONDS_CALC}s (${DURATION} seconds)"
        echo "========================================="
    } > "$TIME_LOG_FILE"

    exit 0
else
    exit 1
fi
