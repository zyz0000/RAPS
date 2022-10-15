LR=1e-5
ModelType=nodropPrototype-nodropRelation-lr-$LR
#nodropPrototype-dropRelation-lr-1e-5
#dropPrototype-nodropRelation-lr-2e-5
#nodropPrototype-nodropRelation-lr-1e-5
#acl-camera-ready-$N-$K.pth.tar
N=5
K=1

CUDA_VISIBLE_DEVICES="1" python train_demo.py \
    --trainN $N --N $N --K $K --Q 1 --dot \
    --hidden_size 768 --val_step 1000 --lr $LR \
    --pretrain_ckpt /data/zyz/practice/RE/HCRP/bert-base-uncased \
    --batch_size 2 --save_ckpt ./checkpoint/camery-ready-$N-$K.pth.tar \
    --cat_entity_rep \
    --backend_model bert
