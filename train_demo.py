import os
import sys
import json
import random
import argparse

import numpy as np

import torch
from torch import optim, nn

from models.proto import Proto
from models.d import Discriminator

from fewshot_re_kit.data_loader import get_loader, get_loader_unsupervised, get_loader_test
from fewshot_re_kit.framework import FewShotREFramework
from fewshot_re_kit.sentence_encoder import BERTSentenceEncoder


def setup_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.random.manual_seed(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default='train_wiki', help='train file')
    parser.add_argument('--val', default='val_wiki', help='val file')
    parser.add_argument('--test', default='5-1-test-relid', help='test file')
    parser.add_argument('--ispubmed', default=False, type=bool, help='FewRel 2.0 or not')
    parser.add_argument('--adv', default=None, help='adv file')
    parser.add_argument('--trainN', default=10, type=int, help='N in train')
    parser.add_argument('--N', default=5, type=int, help='N way')
    parser.add_argument('--K', default=5, type=int, help='K shot')
    parser.add_argument('--Q', default=5, type=int, help='Num of query per class')
    parser.add_argument('--batch_size', default=4, type=int, help='batch size')
    parser.add_argument('--train_iter', default=30000, type=int, help='num of iters in training')
    parser.add_argument('--val_iter', default=1000, type=int, help='num of iters in validation')
    parser.add_argument('--test_iter', default=2500, type=int, help='num of iters in testing')
    parser.add_argument('--val_step', default=2000, type=int, help='val after training how many iters')
    parser.add_argument('--model', default='proto', help='model name')
    parser.add_argument('--max_length', default=128, type=int, help='max length')
    parser.add_argument('--lr', default=-1, type=float, help='learning rate')
    parser.add_argument('--weight_decay', default=1e-5, type=float, help='weight decay')
    parser.add_argument('--dropout', default=0.0, type=float, help='dropout rate')
    parser.add_argument('--grad_iter', default=1, type=int, help='accumulate gradient every x iterations')
    parser.add_argument('--optim', default='adamw', help='sgd / adam / adamw')
    parser.add_argument('--hidden_size', default=230, type=int, help='hidden size')
    parser.add_argument('--load_ckpt', default=None, help='load ckpt')
    parser.add_argument('--save_ckpt', default=None, help='save ckpt')
    parser.add_argument('--only_test', action='store_true', help='only test')
    parser.add_argument('--test_online', action='store_true', help='generate the result for submitting')
    parser.add_argument('--ckpt_name', type=str, default='', help='checkpoint name.')
    parser.add_argument('--seed', default=42, type=int, help='seed')
    
    # only for bert
    parser.add_argument('--pretrain_ckpt', default=None, help='bert / roberta pre-trained checkpoint')
    parser.add_argument('--cat_entity_rep', action='store_true', help='concatenate entity representation as sentence rep')

    # only for prototypical networks
    parser.add_argument('--dot', action='store_true', help='use dot instead of L2 distance for proto')
    
    # experiment
    parser.add_argument('--mask_entity', action='store_true', help='mask entity names')
    parser.add_argument('--use_sgd_for_bert', action='store_true', help='use SGD instead of AdamW for BERT.')
    parser.add_argument('--test_output', default=None, help='test file')
    parser.add_argument('--backend_model', type=str, default='bert', choices=['bert', 'cp'], help='checkpoint name.')

    opt = parser.parse_args()
    trainN = opt.trainN
    N = opt.N
    K = opt.K
    Q = opt.Q
    batch_size = opt.batch_size
    max_length = opt.max_length
    
    print("{}-way-{}-shot Few-Shot Relation Classification".format(N, K))
    print("model: {}, encoder: bert".format(opt.model))
    print("max_length: {}".format(max_length))
    print("learning rate: {}".format(opt.lr))
    print("backend model: {}".format(opt.backend_model))
    
    setup_seed(opt.seed)
    
    pretrain_ckpt = opt.pretrain_ckpt or 'bert-base-uncased'
    sentence_encoder = BERTSentenceEncoder(
                            pretrain_ckpt,
                            max_length,
                            cat_entity_rep=opt.cat_entity_rep,
                            mask_entity=opt.mask_entity,
                            backend_model=opt.backend_model)

    train_data_loader = get_loader(opt.train, sentence_encoder,
                                   N=trainN, K=K, Q=Q, ispubmed=opt.ispubmed, batch_size=batch_size)
    val_data_loader = get_loader(opt.val, sentence_encoder,
                                 N=N, K=K, Q=Q, ispubmed=opt.ispubmed, batch_size=batch_size)
    test_data_loader = get_loader_test(opt.test, sentence_encoder,
                                       N=N, K=K, Q=Q, ispubmed=opt.ispubmed, batch_size=batch_size)

    if opt.optim == 'sgd':
        pytorch_optim = optim.SGD
    elif opt.optim == 'adam':
        pytorch_optim = optim.Adam
    elif opt.optim == 'adamw':
        from transformers import AdamW
        pytorch_optim = AdamW
    else:
        raise NotImplementedError
    if opt.adv:
        adv_data_loader = get_loader_unsupervised(opt.adv, sentence_encoder, N=trainN, K=K, Q=Q, batch_size=batch_size)
        d = Discriminator(opt.hidden_size)
        framework = FewShotREFramework(train_data_loader, val_data_loader, test_data_loader, adv_data_loader, adv=opt.adv, d=d)
    else:
        framework = FewShotREFramework(train_data_loader, val_data_loader, test_data_loader)
        
    prefix = '-'.join(["proto", "bert", opt.train, opt.val, str(N), str(K)])
    if opt.adv is not None:
        prefix += '-adv_' + opt.adv
    if opt.dot:
        prefix += '-dot'
    if opt.cat_entity_rep:
        prefix += '-catentity'
    if len(opt.ckpt_name) > 0:
        prefix += '-' + opt.ckpt_name
    if opt.model == "proto":
        model = Proto(sentence_encoder, dot=opt.dot)
    else:
        raise NotImplementedError
    
    if not os.path.exists('../checkpoint'):
        os.mkdir('../checkpoint')
    ckpt = '../checkpoint/{}.pth.tar'.format(prefix)
    if opt.save_ckpt:
        ckpt = opt.save_ckpt

    if torch.cuda.is_available():
        model.cuda()
    
    if not opt.only_test and (not opt.test_online):
        bert_optim = True
        if opt.lr == -1:
            if bert_optim:
                opt.lr = 2e-5
            else:
                opt.lr = 1e-1
        
        opt.train_iter = opt.train_iter * opt.grad_iter
        framework.train(model, prefix, trainN, N, K, Q,
                        pytorch_optim=pytorch_optim, load_ckpt=opt.load_ckpt, save_ckpt=ckpt,
                        train_iter=opt.train_iter, val_iter=opt.val_iter, val_step=opt.val_step,
                        bert_optim=bert_optim, learning_rate=opt.lr,
                        use_sgd_for_bert=opt.use_sgd_for_bert, grad_iter=opt.grad_iter)
    
    elif opt.test_online:
        print('this is the test online type.')
        ckpt = opt.load_ckpt
        if ckpt is None:
            print("Warning: --load_ckpt is not specified. Will load Hugginface pre-trained checkpoint.")
            ckpt = 'none'
        framework.test(model, batch_size, N, K, Q, opt.test_iter, ckpt=ckpt, test_output=opt.test_output)
    
    else:
        ckpt = opt.load_ckpt
        if ckpt is None:
            print("Warning: --load_ckpt is not specified. Will load Hugginface pre-trained checkpoint.")
            ckpt = 'none'

        acc = framework.eval(model, N, K, Q, opt.test_iter, ckpt=ckpt)
        print("RESULT: %.2f" % (acc * 100))


if __name__ == "__main__":
    main()
