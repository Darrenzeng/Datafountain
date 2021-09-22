import pandas as pd
import directory
from tqdm import tqdm
import pickle
import re
import torch
from run_config import RunConfig
from torch.utils import data
import gensim
# def preprocess_text(document):
#     # 删除符号
#     text = str(document)
#     for punct in "，":
#         text = text.replace(punct, '')
#     for punct in '?!！.,"#&$%\()*+-:;<=>@[\\]^_`{|}~“”‘’':
#         text = text.replace(punct, ' ')
#     for punct in '\n':
#         text = text.replace(punct, ' ')
#     # 删除所有单个字符
#     # text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)
#     # 从开头删除单个字符
#     # text = re.sub(r'\^[a-zA-Z]\s+', ' ', text)
#     # 用单个空格替换多个空格
#     # text = re.sub(r'\s+', ' ', text, flags=re.I)
#     # 转换为小写
#     # text = text.lower()
#     # 词形还原
#     # tokens = text.split()
#     # tokens = [stemmer.lemmatize(word) for word in tokens]
#     # 去停用词
#     # tokens = [word for word in tokens if word not in en_stop]
#     # preprocessed_text = ' '.join(tokens)
#     return text
def preprocess_text(document):

    # 删除逗号
    text = str(document)
    text = text.replace('，','')
    text = text.replace('！', '')
    # 用单个空格替换多个空格
    text = re.sub(r'\s+', ' ', text, flags=re.I)

    return text

#处理训练集
def train_pro(train_set_path):
    if train_set_path != directory.TRAIN_SET_PATH:
        raise ValueError('Train_set_path error.')

    train_df = pd.read_csv(train_set_path)

    train_df.drop(['id'], axis=1, inplace=True)

    train_df['text'] = [i.strip('|').strip() for i in train_df['text'].values]
    train_df['label'] = [i.strip('|').strip() for i in train_df['label'].values]

    train_num = len(train_df)
    #替换特殊符号
    train_df['text'] = train_df['text'].apply(lambda x: preprocess_text(x))
    #制作标签
    id2label = list(train_df['label'].unique())
    label2id = {id2label[i]: i for i in range(len(id2label))}
    with open('./model/id2label.pickle', 'wb') as f:
        pickle.dump(id2label, f)
    with open('./model/label2id.pickle', 'wb') as f:
        pickle.dump(label2id, f)
        
    train_df['label'] = train_df['label'].apply(lambda x: label2id[x])
    train_df['text'] = train_df['text'].apply(lambda x: [int(word) for word in x.split()])

    # for train_idx in tqdm(range(train_num)):

    #     des = train_df.loc[train_idx, 'description']
    #     des = [int(word) for word in des.split()]
    #     train_df.loc[train_idx, 'description'] = des
        # train_df.iloc[train_idx]['description'] = des

    return train_df

#处理测试集
def test_pro(test_set_path):
    
    test_df = pd.read_csv(test_set_path)
#     test_df.columns = ['report_ID', 'description']
    # test_df.drop(['id'], axis=1, inplace=True)
#     test_df['text'] = [i.strip('|').strip() for i in test_df['text'].values]
    test_df['text'] = test_df['text'].apply(lambda x: preprocess_text(x))
    test_df['text'] = [i for i in test_df['text'].values]
    
    test_num = len(test_df)
    # test_df['text'] = test_df['text'].apply(lambda x: [int(word) for word in x.split()])
    
    test_dataset = []
    for i in tqdm(range(len(test_df))):
        test_dict = {}
        test_dict['id'] = test_df.loc[i, 'id']
        test_dict['text'] = test_df.loc[i, 'text']
        test_dict['label'] = -1
        test_dataset.append(test_dict)
        
    #生成dataloader
    test_dataloader, test_torchdata = get_dataloader(RunConfig(), test_dataset, mode='test')

    return test_dataloader#原来的test只有文本内容



class DataSet(data.Dataset):
    def __init__(self, args, data, mode='train'):
        self.data = data
        self.mode = mode
        self.w2v_model = gensim.models.word2vec.Word2Vec.load("./embedding/w2v.model")
        self.dataset = self.get_data(args, self.data, self.mode)

    def get_data(self, args, data, mode):
        dataset = []
        global s
        for data_li in tqdm(data):
            text = data_li['text'].split(' ')
            text = [self.w2v_model.wv.key_to_index[s] +1 if s in self.w2v_model.wv else 0 for s in text]
            if len(text) < args.seq_len:
                text += [0] * (args.seq_len - len(text))
            else:
                text = text[:args.seq_len]
            label = data_li['label']
            dataset_dict = {'text': text, 'label': label}
            dataset.append(dataset_dict)
        return dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        text = torch.tensor(data['text'])
        if self.mode == 'test':
            return text
        else:
            label = torch.tensor(data['label'])
            return text, label


def get_dataloader(args, dataset, mode):
    torchdata = DataSet(args, dataset, mode=mode)
    if mode == 'train':
        dataloader = torch.utils.data.DataLoader(
            torchdata, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    elif mode == 'test':
        dataloader = torch.utils.data.DataLoader(
            torchdata, batch_size=args.val_batch_size, shuffle=False, num_workers=4, drop_last=False)
    elif mode == 'valid':
        dataloader = torch.utils.data.DataLoader(
            torchdata, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=True)
    return dataloader, torchdata