# SHINE: Saliency-aware HIerarchical NEgative Ranking for Compositional Temporal Grounding -- ECCV2024

This is the implementation for the paper "SHINE: Saliency-aware HIerarchical NEgative Ranking for Compositional Temporal Grounding" (**ECCV 2024**):  [ArXiv version](https://arxiv.org/abs/2407.05118).

by

Zixu Cheng\*<sup>1</sup>, Yujiang Pu\*<sup>2</sup>, Shaogang Gong<sup>1</sup>, Parisa Kordjamshidi<sup>2</sup>, Yu Kong<sup>2</sup>

<sup>1</sup>Queen Mary University of London, <sup>2</sup> Michigan State University (* Equal Contribution)

## Prerequisites

<b>0. Clone this repo</b>

<b>1. Prepare datasets</b>

<b>Charades-CG</b> : Download I3D feature files for Charades-CG dataset from [VSLNet](https://github.com/26hzhang/VSLNet).

<b>ActivityNet-CG</b> : Download C3D feature files for ActivityNet-CG dataset from [MS-2D-TAN](https://github.com/microsoft/VideoX/tree/master/MS-2D-TAN). 

**Text Features** : We provide our hierarchical negative query features here. (To be uploaded)

<b>2. Install dependencies.</b>

```
conda create -n shine python=3.10
conda activate shine
cd SHINE
pip install -r requirements.txt
```

## Train

**Charades-CG**

1. Add your data and feature path in shine/scripts/train_charades_cg.sh

```
######## setup video+text features
feat_root= # path/to/your/anet/features
```

2. Run the script

```
bash shine/scripts/train_charades_cg.sh
```

**ActivityNet-CG**

1. Add your data and feature path in shine/scripts/train_anet_cg.sh

```
######## setup video+text features
feat_root= # path/to/your/anet/features
```

2. Run the script

```
bash shine/scripts/train_anet_cg.sh --clip_length 1 --saliency_margin 1.0  --max_es_cnt 10 --max_q_l 50 --enc_layers 3 --dec_layers 3 --lr 0.00013 --use_saliency_loss
```

## Evaluation

```
# Evaluate Charades-CG
bash shine/scripts/inference_charades.sh path/to/your/ckpt 'val'
# Evaluate ActivityNet-CG
bash shine/scripts/inference_anet.sh path/to/your/ckpt 'val'
```

We also provide our checkpoints here. (To be uploaded)

## Contributors and Contact

If there are any questions, feel free to contact the authors: Zixu Cheng (zixu.cheng@qmul.ac.uk), and Yujiang Pu (puyujian@msu.edu).

## Acknowledgment

Our implementations are based on [Moment-DETR](https://github.com/jayleicn/moment_detr) and [QD-DETR](https://github.com/wjun0830/QD-DETR). We thank the authors for their awesome open-source contributions.

## LICENSE

The annotation files are transformed from [VISA](https://github.com/YYJMJC/Compositional-Temporal-Grounding) and many parts of the implementations are borrowed from [Moment-DETR](https://github.com/jayleicn/moment_detr) and [QD-DETR](https://github.com/wjun0830/QD-DETR). 
Following, Our codes are also under the [MIT](https://opensource.org/licenses/MIT) license.
