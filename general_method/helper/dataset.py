from run_config import RunConfig
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
import gensim

class TextDataset(Dataset):
    def __init__(self, df, idx):#传入训练集，训练集idx
        super().__init__()
        self.run_config = RunConfig()
        self.df = df.loc[idx, :].reset_index(drop=True)
        self.description = df['text'].values
        self.labels = df['label'].values

    @staticmethod
    def get_dummy(classes):
        """
        标签转为0/1向量
        """
        label = [0] * 35
        label[classes] = 1
        
        # if classes == '':
        #     return label
        # else:
        #     temp = [int(i) for i in classes.split(' ')]

        #     for i in temp:
        #         label[i] = 1


        return label

    def des_padding(self, des_list):
        """
        截断文本，少的用858填充，多的直接截断
        """
        des_len = len(des_list)
        if des_len > self.run_config.seq_len:
            des = des_list[:self.run_config.seq_len]
        else:
            des = des_list + [858] * (self.run_config.seq_len - des_len)
        return des

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        des = self.description[idx]
        label = self.labels[idx]
        padding_des = self.des_padding(des)
        label = self.get_dummy(label)

        return np.array(padding_des), np.array(label)


class DataSet(Dataset):
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