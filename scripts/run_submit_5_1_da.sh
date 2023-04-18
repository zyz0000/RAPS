#!/usr/bin/env bash
export BACKEND="cp"

N=5
K=1

export CKPT_NAME=val-pubmed-$N-$K-$BACKEND.pth.tar

CUDA_VISIBLE_DEVICES="0" python ../train_demo.py \
    --trainN $N --N $N --K $K --Q 1 --dot \
    --hidden_size 768 --val_step 1000 --test test-pubmed-$N-$K-relid \
    --batch_size 4 --test_online --ispubmed True \
    --load_ckpt ../checkpoint/$CKPT_NAME \
    --pretrain_ckpt ../bert-base-uncased \
    --test_output ../submit-da/pred-$N-$K.json \
    --cat_entity_rep \
    --backend_model $BACKEND