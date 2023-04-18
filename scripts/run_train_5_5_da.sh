#!/usr/bin/env bash
export BACKEND="cp"
if [ "$BACKEND"x = "bert"x ];then
  export LR=1e-5
else
  export LR=5e-6
fi

N=5
K=5

export CKPT_NAME=val-pubmed-$N-$K-$BACKEND.pth.tar

CUDA_VISIBLE_DEVICES="2" python ../train_demo.py \
    --trainN $N --N $N --K $K --Q 1 --dot \
    --val val_pubmed --test val_pubmed --ispubmed True \
    --hidden_size 768 --val_step 1000 --lr $LR \
    --model proto --pretrain_ckpt ../bert-base-uncased \
    --batch_size 4 --save_ckpt ../checkpoint/$CKPT_NAME \
    --cat_entity_rep \
    --backend_model $BACKEND
