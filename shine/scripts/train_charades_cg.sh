dset_name=charades
ctx_mode=video_tef
v_feat_types=i3d
t_feat_type=clip 
results_root=results
exp_id=charades_cg_i3d

######## data paths
train_path=data/charades_cg_gpt_train.jsonl
eval_path=()
eval_path+=(data/charades_cg_test.jsonl)
eval_path+=(data/charades_cg_novel_composition.jsonl)
eval_path+=(data/charades_cg_novel_word.jsonl)
eval_split_name=val

######## setup video+text features
feat_root= # path/to/your/anet/features

# video features
v_feat_dim=0
v_feat_dirs=()
if [[ ${v_feat_types} == *"slowfast"* ]]; then
  v_feat_dirs+=(${feat_root}/slowfast_features)
  (( v_feat_dim += 2304 ))  # double brackets for arithmetic op, no need to use ${v_feat_dim}
fi
if [[ ${v_feat_types} == *"clip"* ]]; then
  v_feat_dirs+=(${feat_root}/clip_features)
  (( v_feat_dim += 512 ))
fi
if [[ ${v_feat_types} == *"i3d"* ]]; then
  v_feat_dirs+=(${feat_root}/i3d_features)
  (( v_feat_dim += 1024 ))
fi

# text features
if [[ ${t_feat_type} == "clip" ]]; then
  t_feat_dir=${feat_root}/text_features/
  t_feat_dim=512
else
  echo "Wrong arg for t_feat_type."
  exit 1
fi

#### training
bsz=32


PYTHONPATH=$PYTHONPATH:. python shine/train.py \
--dset_name ${dset_name} \
--ctx_mode ${ctx_mode} \
--train_path ${train_path} \
--eval_path ${eval_path[@]} \
--eval_split_name ${eval_split_name} \
--v_feat_dirs ${v_feat_dirs[@]} \
--v_feat_dim ${v_feat_dim} \
--t_feat_dir ${t_feat_dir} \
--t_feat_dim ${t_feat_dim} \
--bsz ${bsz} \
--results_root ${results_root} \
--exp_id ${exp_id} \
${@:1}
