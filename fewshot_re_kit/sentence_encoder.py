import os
import math
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from transformers import BertTokenizer, BertModel


class BERTSentenceEncoder(nn.Module):

    def __init__(self, pretrain_path, max_length, cat_entity_rep=False, mask_entity=False, backend_model=None): 
        nn.Module.__init__(self)
        self.bert = BertModel.from_pretrained(pretrain_path)
        
        if backend_model == 'cp':
            print("Load CP model as backbone.")
            ckpt = torch.load("../CP/CP")
            temp = OrderedDict()
            ori_dict = self.bert.state_dict()
            for name, parameter in ckpt["bert-base"].items():
                if name in ori_dict:
                    temp[name] = parameter
            
            ori_dict.update(temp)
            self.bert.load_state_dict(ori_dict)
        
        self.max_length = max_length
        self.max_length_name = 8
        self.tokenizer = BertTokenizer.from_pretrained(pretrain_path)
        self.cat_entity_rep = cat_entity_rep
        self.mask_entity = mask_entity
    
    def forward(self, inputs, cat=True):
        if not self.cat_entity_rep:
            x = self.bert(inputs['word'], attention_mask=inputs['mask'])['pooler_output']
            return x
        else:
            outputs = self.bert(inputs['word'], attention_mask=inputs['mask'])
            if cat:
                tensor_range = torch.arange(inputs['word'].size()[0])  # inputs['word'].shape  [20, 128]
                h_state = outputs['last_hidden_state'][tensor_range, inputs["pos1"]] # h_state.shape [20, 768]
                t_state = outputs['last_hidden_state'][tensor_range, inputs["pos2"]] # [20, 768]
                return h_state, t_state, outputs['last_hidden_state']
            else:
                return outputs['pooler_output'], outputs['last_hidden_state']
    
    def tokenize(self, raw_tokens, pos_head, pos_tail):
        # token -> index
        tokens = ['[CLS]']
        cur_pos = 0
        pos1_in_index = 1
        pos1_end_index = 1
        
        pos2_in_index = 1
        pos2_end_index = 1
        
        for token in raw_tokens:
            token = token.lower()
            if cur_pos == pos_head[0]: #if current position is the head position of the entity, insert '[unused0]'.
                tokens.append('[unused0]')
                pos1_in_index = len(tokens)
            if cur_pos == pos_tail[0]:
                tokens.append('[unused1]')
                pos2_in_index = len(tokens)
            if self.mask_entity and ((pos_head[0] <= cur_pos and cur_pos <= pos_head[-1]) or (pos_tail[0] <= cur_pos and cur_pos <= pos_tail[-1])):
                tokens += ['[unused4]']
            else:
                tokens += self.tokenizer.tokenize(token)
            if cur_pos == pos_head[-1]:
                tokens.append('[unused2]')
                pos1_end_index = len(tokens)
                
            if cur_pos == pos_tail[-1]:
                tokens.append('[unused3]')
                pos2_end_index = len(tokens)
                
            cur_pos += 1
            ## the operation above does like: insert '[unused0]','[unused2]' before and after the head entity; insert '[unused1]', '[unused3]' before and after the tail entity
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        
        # padding
        while len(indexed_tokens) < self.max_length:
            indexed_tokens.append(0)
        indexed_tokens = indexed_tokens[:self.max_length]
        
        # pos
        pos1 = np.zeros(self.max_length, dtype=np.int32)
        pos2 = np.zeros(self.max_length, dtype=np.int32)
        for i in range(self.max_length):
            pos1[i] = i - pos1_in_index + self.max_length
            pos2[i] = i - pos2_in_index + self.max_length

        # mask
        mask = np.zeros(self.max_length, dtype=np.int32)
        mask[:len(tokens)] = 1

        pos1_in_index = min(self.max_length, pos1_in_index)
        pos2_in_index = min(self.max_length, pos2_in_index)
        
        pos1_end_index = min(self.max_length, pos1_end_index)
        pos2_end_index = min(self.max_length, pos2_end_index)

        return indexed_tokens, pos1_in_index - 1, pos2_in_index - 1, mask, len(indexed_tokens), pos1_end_index - 1, pos2_end_index - 1 #these positions are exactly the position of four special charaters
        
    ##TODO tokenize relation name and description
    def tokenize_rel(self, raw_tokens):
        # token -> index
        tokens = ['[CLS]']
        name, description = raw_tokens
        for token in name.split(' '):
            token = token.lower()
            tokens += self.tokenizer.tokenize(token)
        tokens.append('[SEP]')
        for token in description.split(' '):
            token = token.lower()
            tokens += self.tokenizer.tokenize(token)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)

        # padding
        while len(indexed_tokens) < self.max_length:
            indexed_tokens.append(0)
        indexed_tokens = indexed_tokens[:self.max_length]

        # mask
        mask = np.zeros(self.max_length, dtype=np.int32)
        mask[:len(tokens)] = 1

        return indexed_tokens, mask

    def tokenize_name(self, name):
        # for FewRel 2.0
        # token -> index
        tokens = ['[CLS]']
        for token in name.split('_'):
            token = token.lower()
            tokens += self.tokenizer.tokenize(token)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)

        # padding
        while len(indexed_tokens) < self.max_length_name:
            indexed_tokens.append(0)
        indexed_tokens = indexed_tokens[:self.max_length_name]

        # mask
        mask = np.zeros(self.max_length_name, dtype=np.int32)
        mask[:len(tokens)] = 1

        return indexed_tokens, mask
