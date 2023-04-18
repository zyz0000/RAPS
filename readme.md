# RAPS
This is the implementation of our paper **RAPS: A Novel Few-Shot Relation Extraction Pipeline with Query-Information Guided Attention and Adaptive Prototype Fusion**. 

### Requirements
- ``python 3.7.13``
- ``PyTorch 1.9.1``
- ``transformers 4.6.0``
- ``numpy 1.21.6``

## Datasets
We experiment our model on two few-shot relation extraction datasets,
 1. [FewRel 1.0, FewRel 2.0 training & validation sets](https://github.com/thunlp/FewRel/tree/master/data)
 2. [FewRel 1.0, FewRel 2.0 test set](https://worksheets.codalab.org/worksheets/0x224557d3a319469c82b0eb2550a2219e)
 
Please download data from the official links and put it under ``./data/``. 

## Pretrained Language Models
1. Download pretrained bert-base-uncased model from HuggingFace (https://huggingface.co/bert-base-uncased) and put the corresponding config files under ``./bert-base-uncased/``.
2. Download the CP pretrained model from https://github.com/thunlp/RE-Context-or-Names/tree/master/pretrain and put the checkpoint under ``./CP/``.

## Training
**FewRel 1.0**
If you want to train a 5-way 1-shot model on FewRel 1.0, run
```bash
cd scripts
bash run_train_5_1.sh
```
In ``run_train_5_1.sh``, you can specify BERT as backend model by ``export BACKEND="bert"``, or CP as backend model by ``export BACKEND="cp"``.

**FewRel 2.0**
If you want to train a 5-way 1-shot model on FewRel 2.0, run
```bash
cd scripts
bash run_train_5_1_da.sh
```
In ``run_train_5_1_da.sh``, you can specify BERT as backend model by ``export BACKEND="bert"``, or CP as backend model by ``export BACKEND="cp"``.

## Evaluation
**FewRel 1.0**
If you want to evaluate a 5-way 1-shot model on FewRel 1.0, run
```bash
cd scripts
bash run_eval_5_1.sh
```
In ``run_eval_5_1.sh``, you can specify BERT as backend model by ``export BACKEND="bert"``, or CP as backend model by ``export BACKEND="cp"``.

**FewRel 2.0**
If you want to evaluate a 5-way 1-shot model on FewRel 2.0, run
```bash
cd scripts
bash run_eval_5_1_da.sh
```
In ``run_eval_5_1_da.sh``, you can specify BERT as backend model by ``export BACKEND="bert"``, or CP as backend model by ``export BACKEND="cp"``.
