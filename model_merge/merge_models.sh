# pip install mergekit

cd PATH_TO_PARENT_DIR
YAML_PATH="PATH_TO_YAML_PATH"

ls $YAML_PATH/*.yaml | xargs -n 1 -P 16 -I {} bash merge_model.sh {}