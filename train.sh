env_name=MsPacman
seed=1
python -u train.py \
    -n "v(sampleDiff)+Anchor(YES+z_detach())+Distance(KL)+Transformer(NoInput)+ContextLength(8)+ActorCritic(Original)+HyperParameter(Original)+ModelSize(+1_inputLayer)-${env_name}-life_done-wm_2L512D8H-100k-seed${seed}" \
    -seed ${seed} \
    -config_path "config_files/STORM.yaml" \
    -env_name "ALE/${env_name}-v5" \
    -trajectory_path "D_TRAJ/${env_name}.pkl" \
    -record_run 