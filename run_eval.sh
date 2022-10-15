ModelType=nodropPrototype-nodropRelation-lr-5e-6
#nodropPrototype-dropRelation-lr-1e-5
#dropPrototype-nodropRelation-lr-2e-5
#nodropPrototype-nodropRelation-lr-1e-5
#acl-camera-ready-$N-$K.pth.tar
N=5
K=1

python train_demo.py \
    --trainN $N --N $N --K $K --Q 1 --dot \
    --hidden_size 768 --val_step 1000 --test val_wiki \
    --batch_size 4 --only_test \
    --load_ckpt ./checkpoint/camery-ready-$N-$K.pth.tar \
    --pretrain_ckpt /data/zyz/practice/RE/HCRP/bert-base-uncased \
    --cat_entity_rep \
    --test_iter 1000 \
    --backend_model bert
