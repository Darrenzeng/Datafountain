import numpy as np
import os
import json
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import pandas as pd
import random
import copy
from torch.utils.data import Dataset, DataLoader, random_split

class MyDataset(Dataset):
    def __init__(self, file, is_train):
        self.data, self.labels, self.is_train = [], [], is_train
        self.vocabs, self.vocab_num = set(), 0

        data = pd.read_csv('data/{}.csv'.format(file))

        for idx in tqdm(range(len(data))):
            seq1 = self.to_index(data.iloc[idx].text)
            self.data.append(seq1)
            self.vocabs = self.vocabs.union(seq1)  #比vocab_num少104个
            self.vocab_num = max([self.vocab_num]+seq1)
            if is_train:
                self.labels.append(int(data.iloc[idx].label))
            else:
                self.labels.append(0)
            
        self.vocab_num += 1#从0开始数的
        print('Vocab number:', self.vocab_num)
    
    def to_index(self, seq):
        # [PAD], [unused1]...[unused99], [UNK], [CLS], [SEP], [MASK]
        # 0, 1..99, 100, 101, 102, 103
        seq = [int(v)+104 for v in seq.strip().split(' ')]
        return seq
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = {'seq':self.data[idx], 'label':self.labels[idx]}
        # item = {'seq1': self.data[idx][0], 'seq2': self.data[idx][1], 'label': self.labels[idx]}
        return item

class MyCollation:
    def __init__(self, config, is_train):
        self.config = config
        self.is_train = is_train
    #mask的设置, 传入一个文本,然后制作mask预测数据
    def mask(self, seq):
        res, label_res = [], [] #res就是被mask后的文本[1110, ...232, 103, ...,321, 103]。label_res记录需要被mask(包含随机替代以及不变)的词(记录原词)，不要被mask的位置都为0:[0, 0, 0, 238, 0, 127, 0, 0, 0, 0, 133, 321, 265]
        for token in seq:
            p = random.random()
            if self.is_train and p < 0.15:#遮蔽15%
                p /= 0.15
                label_res.append(token)
                if p < 0.8:
                    res.append(103)  # [MASK]，被mask的词，以103（mask的对应token）来取代
                elif p < 0.9:
                    res.append(random.sample(self.config.vocabs, 1)[0])#%10随机替换
                else:
                    res.append(token)#10%不变
            else:
                res.append(token)
                label_res.append(0)
        return res, label_res
    
    def __call__(self, data):#从get_ittem_获得一个batch的数据[{'seq1': [...], 'seq2': [...], 'label': 0},{...}]
        inputs, segs, mask_labels, cls_labels = [], [], [], []
        max_len = 0 # 0
        for datum in data:#!!!!!!!!!!!!!!!!!!!!!!
            max_len = max(max_len, len(datum['seq'])+3) #+3是加上[cls],[sep],[unk]
        for datum in data:
            seq, label_seq = self.mask(datum['seq'])#获得被mask后的文本[1110, ...232, 103, ...,321, 103]，及对应的标记[0, 0, 0, 238, 0, 127, 0, 0, 0, 0, 133, 321, 265]
            # seq2, label_seq2 = self.mask(datum['seq2'])
            if not self.is_train or random.randint(0, 1):
                input = [101]+seq+[102]
                seg = [0]*(len(seq)+2)
                mask_label = [0]+label_seq+[0]
            else:##########!!!!!!!!!!!!!!!!!!!!!!
                input = [101]+seq+[102] #制作input_id：cls+text(被mask的)+sep+text(被mask的)+sep
                seg = [0]*(len(seq)+2) #seg_id
                mask_label = [0]+label_seq+[0] #是否被mask的标记
            input += [0]*(max_len-len(input))#做填补
            seg += [1]*(max_len-len(seg))
            mask_label += [0]*(max_len-len(mask_label))
            inputs.append(input)
            segs.append(seg)
            mask_labels.append(mask_label)
            ####修改label
            cls_labels.append(datum['label'])#原数据的label也记录下来
        inputs = torch.tensor(inputs, dtype=torch.long).to(self.config.device) #[4, 23]
        segs = torch.tensor(segs, dtype=torch.long).to(self.config.device)
        res = {'inputs': inputs, 'segs': segs, 'mask_labels': mask_labels, 'cls_labels': cls_labels}
        return res

class InfiniteDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.iterator = super().__iter__()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.iterator) #获得一个batch的数据
        except StopIteration:
            self.iterator = super().__iter__()
            batch = next(self.iterator)
        return batch

class MyDataLoader:
    def __init__(self, config):
        self.train = MyDataset(config.train, True) #19015个词+， 获得测试集的self.data, self.label，self.vocabs, self.vocab_num
        self.test = MyDataset(config.test, False) #9322个词， 同理{'seq1': [105, 106, 107, 108, 109, 110, 111], 'seq2': [112, 113, 114, 108, 115], 'label': 0}
        config.vocabs = self.train.vocabs.union(self.test.vocabs) #union可以理解为去重，词表集合: 20600个词
        config.vocab_num = max(self.train.vocab_num, self.test.vocab_num, 35000) #测试集，训练集中的词集合与25000， 得到未知的token？？？
        print('Unknown token number in test:', len(self.test.vocabs-self.train.vocabs)) #测试集中有的词是训练集中没有的？：1585
        self.config = config
        self.fn_train = MyCollation(config, True)
        self.fn_eval = MyCollation(config, False)
    
    def get_train(self):
        n = len(self.train)
        d1, d2 = int(n*0.9), n-int(n*0.9) #拆出训练集和验证集
        train, valid = random_split(self.train, [d1, d2])
        [test] = random_split(self.test, [len(self.test)]) #获得测试集
        valid_unlabel = []
        for datum in valid:#给验证集添加上的假标签,覆盖真标签
            datum_new = copy.deepcopy(datum)
            # datum_new['label'] = 0  ################### -1
            valid_unlabel.append(datum_new)
        train += test+valid_unlabel # {'seq1': [239, 240, 236, 917, 202, 248, 2733], 'seq2': [239, 240, 236, 2745, 2771, 123, 312, 1117], 'label': 0}
        train = InfiniteDataLoader(train, self.config.batch_size(True), shuffle=True, collate_fn=self.fn_train) #{'inputs': tensor([[101, 13...), 'segs': tensor([[0, 0,...), 'mask_labels': [[...], ...], 'cls_labels': [0, 0, 1,...]}
        valid = DataLoader(valid, self.config.batch_size(False), shuffle=False, collate_fn=self.fn_eval)
        return train, valid
    
    def get_all(self):
        data = DataLoader(self.train+self.test, self.config.batch_size(False), shuffle=False, collate_fn=self.fn_eval)
        return data
    
    def get_predict(self):
        data = DataLoader(self.test, self.config.batch_size(False), shuffle=False, collate_fn=self.fn_eval)
        return data