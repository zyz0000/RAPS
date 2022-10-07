import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import BertTokenizer, BertModel

'''
def entity_conv(h_state, t_state, cls_state, pooling):
    h_state_mean = h_state.mean(dim=0).unsqueeze(0).transpose(0, 1).unsqueeze(-1)
    t_state_mean = t_state.mean(dim=0).unsqueeze(0).transpose(0, 1).unsqueeze(-1)
    h_state_conv = pooling(F.conv1d(cls_state, h_state_mean)).squeeze(-1)
    t_state_conv = pooling(F.conv1d(cls_state, t_state_mean)).squeeze(-1)
    state = torch.cat((h_state_conv, t_state_conv), -1)
    return state
'''


class BERTSentenceEncoder(nn.Module):

    def __init__(self, pretrain_path, max_length, num_kernels, path):
        nn.Module.__init__(self)
        self.bert = BertModel.from_pretrained(pretrain_path)
        if path is not None and path != "None":
            self.bert.load_state_dict(torch.load(path)["bert-base"])
            print("We load " + path + " to train!")
        self.max_length = max_length
        self.max_length_name = 8
        emb_dim = self.bert.pooler.dense.out_features
        self.num_kernels = num_kernels
        self.tokenizer = BertTokenizer.from_pretrained(pretrain_path)
        self.pool = nn.AvgPool1d(emb_dim - num_kernels + 1)

    def forward(self, inputs, cat=True):
        outputs = self.bert(inputs['word'], attention_mask=inputs['mask'])
        if cat:
            tensor_range = torch.arange(inputs['word'].size()[0])
            cls_state = outputs[0][tensor_range, torch.zeros_like(inputs["pos1"])].unsqueeze(1)
            h_state = outputs[0][tensor_range, inputs["pos1"]]
            t_state = outputs[0][tensor_range, inputs["pos2"]]
            h_state_mean = h_state.mean(dim=0).unsqueeze(0).transpose(0, 1).unsqueeze(-1)
            t_state_mean = t_state.mean(dim=0).unsqueeze(0).transpose(0, 1).unsqueeze(-1)
            kernel_size = (1, 1, self.num_kernels)
            h_state_mean = h_state_mean.repeat(*kernel_size)
            t_state_mean = t_state_mean.repeat(*kernel_size)
            h_state_conv = self.pool(F.conv1d(cls_state, h_state_mean)).squeeze(-1)
            t_state_conv = self.pool(F.conv1d(cls_state, t_state_mean)).squeeze(-1)
            state = torch.cat((h_state_conv, t_state_conv), -1)
            return state, outputs[0]
        else:
            return outputs[1], outputs[0]

    def tokenize(self, raw_tokens, pos_head, pos_tail):
        # token -> index
        tokens = ['[CLS]']
        cur_pos = 0
        pos1_in_index = 0
        pos2_in_index = 0
        for token in raw_tokens:
            token = token.lower()
            if cur_pos == pos_head[0]:
                tokens.append('[unused0]')
                pos1_in_index = len(tokens)
            if cur_pos == pos_tail[0]:
                tokens.append('[unused1]')
                pos2_in_index = len(tokens)
            tokens += self.tokenizer.tokenize(token)
            if cur_pos == pos_head[-1]:
                tokens.append('[unused2]')
            if cur_pos == pos_tail[-1]:
                tokens.append('[unused3]')
            cur_pos += 1

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
        
        if pos1_in_index == 0:
            pos1_in_index = 1
        if pos2_in_index == 0:
            pos2_in_index = 1
        pos1_in_index = min(self.max_length, pos1_in_index)
        pos2_in_index = min(self.max_length, pos2_in_index)
    
        return indexed_tokens, pos1_in_index - 1, pos2_in_index - 1, mask

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
