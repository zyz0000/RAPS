ModelType=nodropPrototype-nodropRelation-lr-5e-6
#nodropPrototype-dropRelation-lr-1e-5
#dropPrototype-nodropRelation-lr-2e-5
#nodropPrototype-nodropRelation-lr-1e-5
#acl-camera-ready-$N-$K.pth.tar
N=10
K=1

CUDA_VISIBLE_DEVICES="1" python train_demo.py \
    --trainN $N --N $N --K $K --Q 1 --dot \
    --hidden_size 768 --val_step 1000 --test $N-$K-test-relid \
    --batch_size 4 --test_online \
    --load_ckpt ./checkpoint/acl-camera-ready-$N-$K.pth.tar \
    --pretrain_ckpt ./bert-base-uncased \
    --test_output ./submit/pred-$N-$K.json \
    --cat_entity_rep \
    --backend_model bert
