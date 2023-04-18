#!/usr/bin/env bash
export BACKEND="cp"

N=10
K=1

export CKPT_NAME=val-wiki-$N-$K-$BACKEND.pth.tar

CUDA_VISIBLE_DEVICES="1" python ../train_demo.py \
    --trainN $N --N $N --K $K --Q 1 --dot \
    --hidden_size 768 --val_step 1000 --test $N-$K-test-relid \
    --batch_size 4 --test_online \
    --load_ckpt ../checkpoint/$CKPT_NAME \
    --pretrain_ckpt ../bert-base-uncased \
    --test_output ../submit-cp/pred-$N-$K.json \
    --cat_entity_rep \
    --backend_model $BACKEND