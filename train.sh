env_name=MsPacman
seed=2
python -u train.py \
    -n "v(sampleDiff)+Anchor(NO)+Distance(KL)+Transformer(NoInput)+ContextLength(8)+ActorCritic(Original)-${env_name}-life_done-wm_2L512D8H-100k-seed${seed}" \
    -seed ${seed} \
    -config_path "config_files/STORM.yaml" \
    -env_name "ALE/${env_name}-v5" \
    -trajectory_path "D_TRAJ/${env_name}.pkl" \
    -record_run 