env_name=ChopperCommand
seed=0
python record_play.py \
  -config_path "config_files/STORM.yaml" \
  -env_name "ALE/${env_name}-v5" \
  -run_name "${env_name}-life_done-wm_2L512D8H-100k-seed${seed}"\
  -output_path "recorded_videos/${env_name}-life_done-wm_2L512D8H-100k-seed${seed}.mp4" \
  -num_videos 1 \
  # -native_resolution \