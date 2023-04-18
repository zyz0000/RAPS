#!/usr/bin/env bash
export BACKEND="cp"

N=5
K=5

export CKPT_NAME=val-wiki-$N-$K-$BACKEND.pth.tar

CUDA_VISIBLE_DEVICES="0" python ../train_demo.py \
    --trainN $N --N $N --K $K --Q 1 --dot \
    --hidden_size 768 --val_step 1000 --test val_wiki \
    --batch_size 4 --only_test \
    --load_ckpt ../checkpoint/$CKPT_NAME \
    --pretrain_ckpt ../bert-base-uncased \
    --cat_entity_rep \
    --test_iter 1000 \
    --backend_model $BACKEND