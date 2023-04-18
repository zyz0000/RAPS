#!/usr/bin/env bash
export BACKEND="cp"
if [ "$BACKEND"x = "bert"x ];then
  export LR=1e-5
else
  export LR=5e-6
fi

N=10
K=1

export CKPT_NAME=val-wiki-$N-$K-$BACKEND.pth.tar

CUDA_VISIBLE_DEVICES="1" python ../train_demo.py \
    --trainN $N --N $N --K $K --Q 1 --dot \
    --hidden_size 768 --val_step 1000 --lr $LR \
    --pretrain_ckpt ../bert-base-uncased \
    --batch_size 2 --save_ckpt ../checkpoint/$CKPT_NAME \
    --cat_entity_rep \
    --backend_model $BACKEND
