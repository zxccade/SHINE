ckpt_path=$1
eval_split_name=$2
eval_path=()
eval_path+=(data/anet_cg/anet_cg_test.jsonl)
eval_path+=(data/anet_cg/anet_cg_novel_composition.jsonl)
eval_path+=(data/anet_cg/anet_cg_novel_word.jsonl)
echo ${ckpt_path}
echo ${eval_split_name}
echo ${eval_path[@]}
PYTHONPATH=$PYTHONPATH:. python shine/inference.py \
--resume ${ckpt_path} \
--eval_split_name ${eval_split_name} \
--eval_path ${eval_path[@]} \
${@:3}
