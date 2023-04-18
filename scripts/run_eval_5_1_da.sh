#!/usr/bin/env bash
export BACKEND="bert"

N=5
K=1

export CKPT_NAME=val-pubmed-$N-$K-$BACKEND.pth.tar

CUDA_VISIBLE_DEVICES="1" python ../train_demo.py \
    --trainN $N --N $N --K $K --Q 1 --dot \
    --hidden_size 768 --val_step 1000 --test val_pubmed \
    --batch_size 4 --only_test --ispubmed True \
    --load_ckpt ../checkpoint/$CKPT_NAME \
    --pretrain_ckpt ../bert-base-uncased \
    --cat_entity_rep \
    --test_iter 1000 \
    --backend_model $BACKEND