import os
import json
import random
import numpy as np
import torch
import torch.utils.data as data


class FewRelDataset(data.Dataset):
    def __init__(self, name, encoder, N, K, Q, root, ispubmed=False):
        self.root = root
        path = os.path.join(root, name + ".json")
        
        pid2name = 'pid2name'
        pid2name_path = os.path.join(root, pid2name + ".json")
        
        if not os.path.exists(path):
            print("[ERROR] Data file does not exist!")
            assert(0)
        self.json_data = json.load(open(path))
        self.pid2name = json.load(open(pid2name_path))
        self.classes = list(self.json_data.keys())
        self.N = N
        self.K = K
        self.Q = Q
        self.ispubmed = ispubmed
        self.encoder = encoder

    def __getraw__(self, item):
        word, pos1, pos2, mask, lens, pos1_end, pos2_end = self.encoder.tokenize(item['tokens'], item['h'][2][0], item['t'][2][0])
        return word, pos1, pos2, mask, lens, pos1_end, pos2_end

    def __additem__(self, d, word, pos1, pos2, mask, lens, pos1_end, pos2_end):
        d['word'].append(word)
        d['pos1'].append(pos1)
        d['pos2'].append(pos2)
        d['mask'].append(mask)
        d['lens'].append(lens)
        d['pos1_end'].append(pos1_end)
        d['pos2_end'].append(pos2_end)
        
    def __getrel__(self, item):
        word, mask = self.encoder.tokenize_rel(item)
        return word, mask

    def __getname__(self, name):
        word, mask = self.encoder.tokenize_name(name)
        return word, mask
    
    def __getitem__(self, index):
        target_classes = random.sample(self.classes, self.N)
        relation_set = {'word': [], 'mask': []}
        support_set = {'word': [], 'pos1': [], 'pos2': [], 'mask': [], 'lens':[], 'pos1_end':[], 'pos2_end': []}
        query_set = {'word': [], 'pos1': [], 'pos2': [], 'mask': [], 'lens':[], 'pos1_end': [], 'pos2_end': []}
        query_label = []

        for i, class_name in enumerate(target_classes):
            if self.ispubmed:
                if class_name in self.pid2name.keys():
                    name, _ = self.pid2name[class_name]
                    rel_text, rel_text_mask = self.__getname__(name)
                else:
                    rel_text, rel_text_mask = self.__getname__(class_name)
            else:
                rel_text, rel_text_mask = self.__getrel__(self.pid2name[class_name])

            rel_text, rel_text_mask = torch.tensor(rel_text).long(), torch.tensor(rel_text_mask).long()
            relation_set['word'].append(rel_text)
            relation_set['mask'].append(rel_text_mask)
        
            indices = np.random.choice(
                    list(range(len(self.json_data[class_name]))), 
                    self.K + self.Q, False)
            count = 0
            for j in indices:
                word, pos1, pos2, mask, lens, pos1_end, pos2_end = self.__getraw__(self.json_data[class_name][j])
                word = torch.tensor(word).long()
                pos1 = torch.tensor(pos1).long()
                pos2 = torch.tensor(pos2).long()
                mask = torch.tensor(mask).long()
                lens = torch.tensor(lens).long()
                pos1_end = torch.tensor(pos1_end).long()
                pos2_end = torch.tensor(pos2_end).long()
                if count < self.K:
                    self.__additem__(support_set, word, pos1, pos2, mask, lens, pos1_end, pos2_end)
                else:
                    self.__additem__(query_set, word, pos1, pos2, mask, lens, pos1_end, pos2_end)
                count += 1

            query_label += [i] * self.Q

        return support_set, query_set, query_label, relation_set  #separate support and query -- no pair
    
    def __len__(self):
        return 1000000000


def collate_fn(data):
    batch_support = {'word': [], 'pos1': [], 'pos2': [], 'mask': [], 'lens': [], 'pos1_end': [], 'pos2_end': []}
    batch_query = {'word': [], 'pos1': [], 'pos2': [], 'mask': [], 'lens': [], 'pos1_end': [], 'pos2_end': []}
    batch_relation = {'word': [], 'mask': []}
    batch_label = []
    support_sets, query_sets, query_labels, relation_sets = zip(*data)
    for i in range(len(support_sets)):
        for k in support_sets[i]:
            batch_support[k] += support_sets[i][k]
        for k in query_sets[i]:
            batch_query[k] += query_sets[i][k]
        for k in relation_sets[i]:
            batch_relation[k] += relation_sets[i][k]
        batch_label += query_labels[i]
    
    for k in batch_support:
        batch_support[k] = torch.stack(batch_support[k], 0)
    for k in batch_query:
        batch_query[k] = torch.stack(batch_query[k], 0)
    for k in batch_relation:
        batch_relation[k] = torch.stack(batch_relation[k], 0)
    batch_label = torch.tensor(batch_label)
    return batch_support, batch_query, batch_label, batch_relation


def get_loader(name, encoder, N, K, Q, batch_size, 
               num_workers=8, collate_fn=collate_fn, ispubmed=False, root='../data'):
    dataset = FewRelDataset(name, encoder, N, K, Q, root, ispubmed)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  pin_memory=True,
                                  num_workers=num_workers,
                                  collate_fn=collate_fn)
    return iter(data_loader)


class FewRelUnsupervisedDataset(data.Dataset):
    def __init__(self, name, encoder, N, K, Q, root):
        self.root = root
        path = os.path.join(root, name + ".json")
        if not os.path.exists(path):
            print("[ERROR] Data file does not exist!")
            assert(0)
        self.json_data = json.load(open(path))
        self.N = N
        self.K = K
        self.Q = Q
        self.encoder = encoder

    def __getraw__(self, item):
        word, pos1, pos2, mask, lens, pos1_end, pos2_end = self.encoder.tokenize(item['tokens'], item['h'][2][0], item['t'][2][0])
        return word, pos1, pos2, mask, lens, pos1_end, pos2_end

    def __additem__(self, d, word, pos1, pos2, mask, lens, pos1_end, pos2_end):
        d['word'].append(word)
        d['pos1'].append(pos1)
        d['pos2'].append(pos2)
        d['mask'].append(mask)
        d['lens'].append(lens)
        d['pos1_end'].append(pos1_end)
        d['pos2_end'].append(pos2_end)

    def __getitem__(self, index):
        total = self.N * self.K
        support_set = {'word': [], 'pos1': [], 'pos2': [], 'mask': [], 'lens': [], 'pos1_end': [], 'pos2_end': []}

        indices = np.random.choice(list(range(len(self.json_data))), total, False)
        for j in indices:
            word, pos1, pos2, mask, lens, pos1_end, pos2_end = self.__getraw__(self.json_data[j])
            word = torch.tensor(word).long()
            pos1 = torch.tensor(pos1).long()
            pos2 = torch.tensor(pos2).long()
            mask = torch.tensor(mask).long()
            lens = torch.tensor(lens).long()
            pos1_end = torch.tensor(pos1_end).long()
            pos2_end = torch.tensor(pos2_end).long()
            self.__additem__(support_set, word, pos1, pos2, mask, lens, pos1_end, pos2_end)

        return support_set
    
    def __len__(self):
        return 1000000000
    

def collate_fn_unsupervised(data):
    batch_support = {'word': [], 'pos1': [], 'pos2': [], 'mask': [], 'lens': [], 'pos1_end': [], 'pos2_end': []}
    support_sets = data
    for i in range(len(support_sets)):
        for k in support_sets[i]:
            batch_support[k] += support_sets[i][k]
    for k in batch_support:
        batch_support[k] = torch.stack(batch_support[k], 0)
    return batch_support


def get_loader_unsupervised(name, encoder, N, K, Q, batch_size, 
                            num_workers=8, collate_fn=collate_fn_unsupervised, root='../data'):
    dataset = FewRelUnsupervisedDataset(name, encoder, N, K, Q, root)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  pin_memory=True,
                                  num_workers=num_workers,
                                  collate_fn=collate_fn)
    return iter(data_loader)


class FewRelTestDataset(data.Dataset):
    def __init__(self, name, encoder, N, K, Q, root, ispubmed=False):
        self.root = root
        path = os.path.join(root, name + ".json")
        
        pid2name = 'pid2name'
        pid2name_path = os.path.join(root, pid2name + ".json")
        
        if not os.path.exists(path):
            print("[ERROR] Data file does not exist!")
            assert(0)
        self.json_data = json.load(open(path))
        self.pid2name = json.load(open(pid2name_path))
        self.N = N
        self.K = K
        self.Q = Q
        self.ispubmed = ispubmed
        self.encoder = encoder

    def __getraw__(self, item):
        word, pos1, pos2, mask, lens, pos1_end, pos2_end = self.encoder.tokenize(item['tokens'], item['h'][2][0], item['t'][2][0])
        return word, pos1, pos2, mask, lens, pos1_end, pos2_end

    def __additem__(self, d, word, pos1, pos2, mask, lens, pos1_end, pos2_end):
        d['word'].append(word)
        d['pos1'].append(pos1)
        d['pos2'].append(pos2)
        d['mask'].append(mask)
        d['lens'].append(lens)
        d['pos1_end'].append(pos1_end)
        d['pos2_end'].append(pos2_end)

    def __getrel__(self, item):
        word, mask = self.encoder.tokenize_rel(item)
        return word, mask

    def __getname__(self, name):
        word, mask = self.encoder.tokenize_name(name)
        return word, mask

    def __getitem__(self, index):
        relation_set = {'word': [], 'mask': []}
        support_set = {'word': [], 'pos1': [], 'pos2': [], 'mask': [], 'lens': [], 'pos1_end': [], 'pos2_end': []}
        query_set = {'word': [], 'pos1': [], 'pos2': [], 'mask': [], 'lens': [], 'pos1_end': [], 'pos2_end': []}

        data = self.json_data[index]
        support_set_my = data['meta_train']
        rel_set = data['relation']
        
        for idx, j in enumerate(support_set_my):
            rel = rel_set[idx]
            if self.ispubmed:
                if rel in self.pid2name.keys():
                    name, _ = self.pid2name[rel]
                    rel_text, rel_text_mask = self.__getrel__(name)
                else:
                    rel_text, rel_text_mask = self.__getname__(rel)
            else:
                rel_text, rel_text_mask = self.__getrel__(self.pid2name[rel_set[idx]])

            rel_text, rel_text_mask = torch.tensor(rel_text).long(), torch.tensor(rel_text_mask).long()
            relation_set['word'].append(rel_text)
            relation_set['mask'].append(rel_text_mask)
            
            for i in j:
                word, pos1, pos2, mask, lens, pos1_end, pos2_end = self.__getraw__(i)
                word = torch.tensor(word).long()
                pos1 = torch.tensor(pos1).long()
                pos2 = torch.tensor(pos2).long()
                mask = torch.tensor(mask).long()
                lens = torch.tensor(lens).long()
                pos1_end = torch.tensor(pos1_end).long()
                pos2_end = torch.tensor(pos2_end).long()
                self.__additem__(support_set, word, pos1, pos2, mask, lens, pos1_end, pos2_end)

            query_set_my = data['meta_test']
        
            word, pos1, pos2, mask, lens, pos1_end, pos2_end = self.__getraw__(query_set_my)
            word = torch.tensor(word).long()
            pos1 = torch.tensor(pos1).long()
            pos2 = torch.tensor(pos2).long()
            mask = torch.tensor(mask).long()
            lens = torch.tensor(lens).long()
            pos1_end = torch.tensor(pos1_end).long()
            pos2_end = torch.tensor(pos2_end).long()
            self.__additem__(query_set, word, pos1, pos2, mask, lens, pos1_end, pos2_end)

        return support_set, query_set, relation_set  #separate support and query -- no pair
    
    def __len__(self):
        return 1000000000


def collate_fn_test(data):
    batch_support = {'word': [], 'pos1': [], 'pos2': [], 'mask': [], 'lens': [], 'pos1_end': [], 'pos2_end': []}
    batch_query = {'word': [], 'pos1': [], 'pos2': [], 'mask': [], 'lens': [], 'pos1_end': [], 'pos2_end': []}
    batch_relation = {'word': [], 'mask': []}
    support_sets, query_sets, relation_sets = zip(*data)
    for i in range(len(support_sets)):
        for k in support_sets[i]:
            batch_support[k] += support_sets[i][k]
        for k in query_sets[i]:
            batch_query[k] += query_sets[i][k]
        for k in relation_sets[i]:
            batch_relation[k] += relation_sets[i][k]
    
    for k in batch_support:
        batch_support[k] = torch.stack(batch_support[k], 0)
    for k in batch_query:
        batch_query[k] = torch.stack(batch_query[k], 0)
    for k in batch_relation:
        batch_relation[k] = torch.stack(batch_relation[k], 0)
        
    return batch_support, batch_query, batch_relation


def get_loader_test(name, encoder, N, K, Q, batch_size,
                    num_workers=8, collate_fn=collate_fn_test, ispubmed=False, root='../data'):
    dataset = FewRelTestDataset(name, encoder, N, K, Q, root, ispubmed)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  pin_memory=True,
                                  num_workers=num_workers,
                                  collate_fn=collate_fn)
    return iter(data_loader)
