import argparse
from data_loader import MyDataLoader
from processor import Processor
from config import Config
import pickle
import os
import torch
import random
import numpy as np
import pandas as pd
import re

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True

def process_text(document):
    #删除标点符号
    text = str(document)
    text = text.replace("，", '')
    text = text.replace('！', '')
    # 用单个空格替换多个空格
    text = re.sub(r'\s+', ' ', text, flags=re.I)
    return text


def label_process():
    file = '/home/zyf/Summer_game2021/Datafountain/simple_bert_pretrain/data/'
    data_train = pd.read_csv(os.path.join(file, 'datagrand_2021_train.csv'))
    data_test = pd.read_csv(os.path.join(file, 'datagrand_2021_test.csv'))
    #先将label转换
    id2label = list(data_train['label'].unique())
    label2id = {id2label[i]: i for i in range(len(id2label))}
    data_train['label'] = data_train['label'].apply(lambda x: label2id[x])

    data_train['text'] = data_train['text'].apply(lambda x: process_text(x))
    rest_data_train = data_train[-1:]
    data_train = pd.concat([data_train, rest_data_train], axis=0)
    # data_train = data_train[:1000]#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    data_test['text'] = data_test['text'].apply(lambda x: process_text(x))
    # data_test = data_test[:1000]

    data_train.to_csv(os.path.join(file, 'datagrand_2021_train_processed.csv'), index=False)
    data_test.to_csv(os.path.join(file, 'datagrand_2021_test_processed.csv'), index=False)

    data_test = data_test.drop(['text'], inplace=False, axis=1)
    return id2label, data_test

    
def main():
    parser = argparse.ArgumentParser(description='Text_match')
    parser.add_argument('-train', type=str, default='datagrand_2021_train_processed', choices=['datagrand_2021_train_processed'])
    parser.add_argument('-test', type=str, default='datagrand_2021_test_processed', choices=['datagrand_2021_test_processed'])
    parser.add_argument('-lr0', type=float, default=3e-4)
    parser.add_argument('-lr', type=float, default=5e-5)
    parser.add_argument('-bs', type=int, default=32) #32
    parser.add_argument('-mask_w', type=float, default=1.5)
    parser.add_argument('-seed', type=int, default=0)
    parser.add_argument('-gpu', type=str, default='cuda:0')
    args = parser.parse_args()
    set_seed(args.seed)
    config = Config(args.train, args.test, args.lr0, args.lr, args.bs, args.mask_w, args.seed, args.gpu)
    #标签转化
    id2label, data_test = label_process()
    data_loader = MyDataLoader(config)
    processor = Processor(data_loader, config) #初始化，训练，评估等的设置
    processor.train() #训练得到模型train_testA_0.0003_5e-05_32_1.5_0.pth
    processor.extract_feature()
    processor.predict(id2label, data_test)

if __name__ == '__main__':
    main()