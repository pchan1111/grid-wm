env_name=MsPacman
seed=1
python -u eval.py \
    -env_name "ALE/${env_name}-v5" \
    -run_name "WD1e_3-AttRepRatio1:2-${env_name}-life_done-wm_2L512D8H-100k-seed${seed}" \
    -config_path "config_files/STORM.yaml" \