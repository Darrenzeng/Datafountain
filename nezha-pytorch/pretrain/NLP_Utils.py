import random
import json
from pandas.core import indexing
import transformers as _
from transformers1 import BertTokenizer
import torch
from torch.utils.data import Dataset,DataLoader
import numpy as np
from itertools import chain
import os
import pandas as pd
import re
from tqdm import tqdm
tqdm.pandas()

def writeToJsonFile(path: str, obj):
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False,indent=0))
def readFromJsonFile(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.loads(f.read())

def loadData(path):
    allData=[]
    with open(path,"r") as f:
        j = 0
        for line in f:
            line=line.strip().split(',')
            if j == 0:
                j += 1
                continue
            if len(line)==0:#防止空行
                break
            if len(line)==3:#训练集
                a,b,label=line
                b=b.split(' ')
            else:#测试集，直接转为id形式
                a,b,label= line[0], line[1], -1
                b=b.split(' ')
            allData.append([b, int(label)]) #做成text+label添加
            j+=1
    # allData = allData[:500]################测试用
    return allData

def calNegPos(ls):#计算正负比例
    posNum,negNum=0,0
    for i in ls:
        if i[2]==0:
            negNum+=1
        elif i[2]==1:
            posNum+=1
    posNum=1 if posNum==0 else posNum
    return negNum,posNum,round(negNum/posNum,4)

def preprocess_text(document):

    # 删除逗号, 脱敏数据中最大值为30357
    text = str(document)
    text = text.replace('，', '35001')
    text = text.replace('！', '35002')
    text = text.replace('？', '35003')
    text = text.replace('。', '35004')
    # text = text.replace('17281', '')
    # 用单个空格替换多个空格
    text = re.sub(r'\s+', ' ', text, flags=re.I)

    return text


train_clean = '/home/zyf/Summer_game2021/Datafountain/datasets/train_clean_add4.csv'
test_clean = '/home/zyf/Summer_game2021/Datafountain/datasets/test_clean_add4.csv'
# aug_clean = '/home/zyf/Summer_game2021/Datafountain/datasets/aug_clean1.csv'



if not os.path.exists(train_clean):
    train_df = pd.read_csv('/home/zyf/Summer_game2021/Datafountain/datasets/datagrand_2021_train.csv')
    test_df = pd.read_csv('/home/zyf/Summer_game2021/Datafountain/datasets/datagrand_2021_test.csv')
    # train_df = train_df[:500]###############


    train_df["text"] = train_df["text"].progress_apply(lambda x: preprocess_text(x))
    id2label = list(train_df['label'].unique())
    label2id = {id2label[i]: i for i in range(len(id2label))}
    train_df["label"] = train_df["label"].map(label2id)
    
    # test_df = test_df[:300] ############
    test_df["text"] = test_df["text"].progress_apply(lambda x: preprocess_text(x))

    #额外的语料库
    # aug_file = '/home/zyf/Summer_game2021/Datafountain/pytorch_chinese_lm_pretrain/datasets/extra_train.txt'
    # with open(aug_file, 'r') as f:
    #     data = f.readlines()
    # augdata_dict = {}
    # text_list = []
    # label_list = []
    # id_list = []
    # idx = 0
    # for lines in tqdm(data):
    #     if lines == '\n':
    #         continue
    #     text = lines.strip()
    #     id_list.append(idx)
    #     idx += 1
    #     text_list.append(text)
    #     label_list.append(-1)
    # augdata_dict['id'] = id_list
    # augdata_dict['text'] = text_list
    # augdata_dict['label'] = label_list
    # augdata_df = pd.DataFrame(augdata_dict)
    # augdata_df["text"] = augdata_df["text"].progress_apply(lambda x: preprocess_text(x))

    # augdata_df.to_csv(aug_clean, index=False)
    train_df.to_csv(train_clean, index=False)
    test_df.to_csv(test_clean,index = False)

# allData = loadData(train_clean) + loadData(test_clean) + loadData(aug_clean)
allData = loadData(train_clean) + loadData(test_clean)
# testA_data = loadData(test_clean)
# testB_data = loadData('/tcdata/gaiic_track3_round1_testB_20210317.tsv')
random.shuffle(allData)

train_data=allData#全量
valid_data=allData[-1000:] #取两万出来作为验证集---->预训练
print("训练集样本数量：", len(train_data))

def paddingList(ls:list,val,returnTensor=False):
    ls=ls[:]#不要改变了原list尺寸
    maxLen=max([len(i) for i in ls])
    for i in range(len(ls)):
        ls[i]=ls[i]+[val]*(maxLen-len(ls[i]))
    return torch.tensor(ls, device='cuda:1') if returnTensor else ls

def truncate(a:list, maxLen):
    maxLen -= 3#空留给cls sep sep
    assert maxLen>=0
    #一共就a超长与否，b超长与否，组合的四种情况
    if len(a)>maxLen:#需要截断
        # 尾截断
        # a=a[:maxLen]
        # 首截断
        # a = a[maxLen-len(a):]
        # 首尾截断
        outlen = (len(a)-maxLen) #前后位置比输出多的词量
        headid = int(outlen/2)  #首部空出一半， 尾部也需要空出一半
        a = a[headid:headid-outlen]

    return a

class MLM_Data(Dataset):
    #传入句子对列表
    def __init__(self, textLs:list, maxLen:int, tk:BertTokenizer):
        super().__init__()
        self.data=textLs
        self.maxLen=maxLen
        self.tk=tk
        self.spNum=len(tk.all_special_tokens) #['[PAD]', '[CLS]', '[UNK]', '[MASK]', '[SEP]']
        self.tkNum=tk.vocab_size

    def __len__(self):
        return len(self.data)

    def random_mask(self,text_ids):
        input_ids, output_ids = [], []
        rands = np.random.random(len(text_ids)) #生成一些随机数
        idx=0
        mask_p = 0.5
        while idx<len(rands): #遍历所有词，看是否需要mask。设置每个词被mask的概率，如果是n_mask,就需要将相连的几个词的mask概率调为可mask
            if rands[idx]<mask_p:#需要mask
                ngram=np.random.choice([1,2,3], p=[0.7,0.2,0.1])#若要mask，进行随机进行1，2，3的mask，x_gram mask的概率
                if ngram==3 and len(rands)<7:#太大的gram不要应用于过短文本
                    ngram=2
                if ngram==2 and len(rands)<4:
                    ngram=1
                L=idx+1 #因为从0开始数
                R=idx+ngram#最终需要mask的右边界（开）
                while L<R and L<len(rands):
#                     rands[L]=np.random.random()*0.15#强制mask
                    rands[L]=np.random.random()*0.5#强制mask
                    L+=1
                idx=R
                if idx<len(rands):
                    rands[idx]=1#禁止mask片段的下一个token被mask，防止一大片连续mask
            idx+=1

        for r, i in zip(rands, text_ids):
            if r < mask_p * 0.8:#mask
                input_ids.append(self.tk.mask_token_id)
                output_ids.append(i)#mask预测自己
            elif r < mask_p * 0.9: #不变
                input_ids.append(i)
                output_ids.append(i)
            elif r < mask_p: #随机替换
                input_ids.append(np.random.randint(self.spNum,self.tkNum))
                output_ids.append(i)  #随机的一个词预测自己，随机词不会从特殊符号中选取，有小概率抽到自己
            else:  #不要mask
                input_ids.append(i)
                output_ids.append(-100)

        return input_ids, output_ids

    #耗时操作在此进行，可用上多进程
    def __getitem__(self, item):
        text1, _ = self.data[item]#预处理，mask等操作

        text1=truncate(text1,self.maxLen)
        text1_ids = self.tk.convert_tokens_to_ids(text1) #文本转化为机器识别的token
        
        text1_ids, out1_ids = self.random_mask(text1_ids)#添加mask预测， text1_ids是做好的mask文本，out1_ids是标记

        input_ids = [self.tk.cls_token_id] + text1_ids + [self.tk.sep_token_id]#拼接:cls+句子+sep
        token_type_ids=[0]*(len(text1_ids)+2)
        labels = [-100] + out1_ids + [-100] 
        assert len(input_ids)==len(token_type_ids)==len(labels)
        return {'input_ids':input_ids,'token_type_ids':token_type_ids,'labels':labels}

    @classmethod
    def collate(cls, batch):
        input_ids=[i['input_ids'] for i in batch]
        token_type_ids=[i['token_type_ids'] for i in batch]
        labels=[i['labels'] for i in batch]
        input_ids=paddingList(input_ids,0,returnTensor=True)
        token_type_ids=paddingList(token_type_ids,0,returnTensor=True)
        labels=paddingList(labels,-100,returnTensor=True)
        attention_mask=(input_ids!=0) #增加attention_mask
        return {'input_ids':input_ids,'token_type_ids':token_type_ids
                ,'attention_mask':attention_mask,'labels':labels}


unionList=lambda ls:list(chain(*ls))#定义一个匿名函数，按元素拼接
splitList=lambda x,bs:[x[i:i+bs] for i in range(0,len(x),bs)]#按bs切分


#sortBsNum：原序列按多少个bs块为单位排序，可用来增强随机性
#比如如果每次打乱后都全体一起排序，那每次都是一样的
def blockShuffle(data:list,bs:int,sortBsNum,key):
    random.shuffle(data)#先打乱
    tail=len(data)%bs#计算碎片长度
    tail=[] if tail == 0 else data[-tail:]
    data=data[:len(data)-len(tail)] #去除碎片数据
    assert len(data)%bs==0#剩下的一定能被bs整除
    sortBsNum=len(data)//bs if sortBsNum is None else sortBsNum#为None就是整体排序
    data=splitList(data,sortBsNum*bs)
    data=[sorted(i,key=key,reverse=True) for i in data]#每个大块进行降排序
    data=unionList(data)
    data=splitList(data,bs)#最后，按bs分块
    random.shuffle(data)#块间打乱
    data=unionList(data)+tail
    return data
from torch.utils.data.dataloader import _SingleProcessDataLoaderIter,_MultiProcessingDataLoaderIter

#每轮迭代重新分块shuffle数据的DataLoader
class blockShuffleDataLoader(DataLoader):
    def __init__(self, dataset: Dataset, sortBsNum, key, **kwargs):
        assert isinstance(dataset.data,list)#需要有list类型的data属性
        super().__init__(dataset,**kwargs)#父类的参数传过去
        self.sortBsNum=sortBsNum 
        self.key=key

    def __iter__(self):
        #分块shuffle
        self.dataset.data=blockShuffle(self.dataset.data, self.batch_size, self.sortBsNum,self.key)
        if self.num_workers == 0:
            return _SingleProcessDataLoaderIter(self)
        else:
            return _MultiProcessingDataLoaderIter(self)
