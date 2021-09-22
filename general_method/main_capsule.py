import pandas as pd
import numpy as np
import collections
import re
import torch
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import log_loss, roc_auc_score, f1_score
import warnings
import torch.nn as nn
from tqdm import tqdm
import random
import gensim
import argparse
from torchcontrib.optim import SWA
import os
import logging
from torch.utils import data
from torch import nn
import torch.nn.functional as F
from torch.optim import *
from adversarial_model import FGM
torch.set_printoptions(edgeitems=768)
warnings.filterwarnings("ignore")
np.set_printoptions(threshold=np.inf)
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
                                                                                                                                                        
def basic_setting(SEED, DEVICE):
    random.seed(SEED)
    os.environ["PYTHONHASHSEED"] = str(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if DEVICE != 'cpu':
        torch.cuda.manual_seed(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def data_process():
    train_data = pd.read_csv('/home/zyf/Summer game2021/Datafountain/datasets/datagrand_2021_train.csv')
    test_data = pd.read_csv('/home/zyf/Summer game2021/Datafountain/datasets/datagrand_2021_test.csv') 

    id2label = list(train_data['label'].unique())
    label2id = {id2label[i]: i for i in range(len(id2label))}
    y_train = np.zeros((len(train_data), len(id2label)), dtype=np.int8)

    all_sentences = pd.concat([train_data['text'], test_data['text']]).reset_index(drop=True)
    all_sentences.drop_duplicates().reset_index(drop=True, inplace=True)
    all_sentences = all_sentences.apply(lambda x: x.split(' ')).tolist()
    if not os.path.exists('/home/zyf/Summer game2021/Datafountain/competition-baseline-main/全球人工智能技术创新大赛/赛道一/embedding/w2v.model'):
        w2v_model = gensim.models.word2vec.Word2Vec(
            all_sentences, sg=1, vector_size=300, window=7, min_count=1, negative=3, sample=0.001, hs=1, seed=452)
        w2v_model.save('/home/zyf/Summer game2021/Datafountain/competition-baseline-main/全球人工智能技术创新大赛/赛道一/embedding/w2v.model')
    else:
        w2v_model = gensim.models.word2vec.Word2Vec.load(
            "/home/zyf/Summer game2021/Datafountain/competition-baseline-main/全球人工智能技术创新大赛/赛道一/embedding/w2v.model")

    if not os.path.exists('/home/zyf/Summer game2021/Datafountain/competition-baseline-main/全球人工智能技术创新大赛/赛道一/embedding/fasttext.model'):
        fasttext_model = gensim.models.FastText(
            all_sentences, seed=452, vector_size=100, min_count=1, epochs=20, window=2)
        fasttext_model.save('/home/zyf/Summer game2021/Datafountain/competition-baseline-main/全球人工智能技术创新大赛/赛道一/embedding/fasttext.model')
    else:
        fasttext_model = gensim.models.word2vec.Word2Vec.load(
            "/home/zyf/Summer game2021/Datafountain/competition-baseline-main/全球人工智能技术创新大赛/赛道一/embedding/fasttext.model")
    train_dataset = []
    for i in tqdm(range(len(train_data))):
        train_dict = {}
        train_dict['id'] = train_data.loc[i, 'id']
        train_dict['text'] = train_data.loc[i, 'text']
        y_train[i][label2id[train_data.loc[i, 'label']]] = 1
        train_dict['label'] = y_train[i]
        train_dataset.append(train_dict)
    test_dataset = []
    for i in tqdm(range(len(test_data))):
        test_dict = {}
        test_dict['id'] = test_data.loc[i, 'id']
        test_dict['text'] = test_data.loc[i, 'text']
        test_dict['label'] = -1
        test_dataset.append(test_dict)
    return test_data, train_dataset, test_dataset, w2v_model, fasttext_model, id2label

class DataSet(data.Dataset):
    def __init__(self, args, data, mode='train'):
        self.data = data
        self.mode = mode
        self.dataset = self.get_data(self.data,self.mode)
        
    def get_data(self, data, mode):
        dataset = []
        global s
        for data_li in tqdm(data):
            text = data_li['text'].split(' ')
            text = [w2v_model.wv.key_to_index[s]+1 if s in w2v_model.wv else 0 for s in text]
            if len(text) < args.MAX_LEN:
                text += [0] * (args.MAX_LEN - len(text))
            else:
                text = text[:args.MAX_LEN]
            label = data_li['label']
            dataset_dict = {'text':text, 'label':label}
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
        dataloader = torch.utils.data.DataLoader(torchdata, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    elif mode == 'test':
        dataloader = torch.utils.data.DataLoader(torchdata, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)
    elif mode == 'valid':
        dataloader = torch.utils.data.DataLoader(torchdata, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=True)#drop_last是否丢弃最后不足一个batch_size的数据
    return dataloader, torchdata

class CyclicLR(object):
    def __init__(self, optimizer, base_lr=1e-3, max_lr=6e-3,
                 step_size=2000, mode='triangular', gamma=1.,
                 scale_fn=None, scale_mode='cycle', last_batch_iteration=-1):

        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        if isinstance(base_lr, list) or isinstance(base_lr, tuple):
            if len(base_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} base_lr, got {}".format(
                    len(optimizer.param_groups), len(base_lr)))
            self.base_lrs = list(base_lr)
        else:
            self.base_lrs = [base_lr] * len(optimizer.param_groups)

        if isinstance(max_lr, list) or isinstance(max_lr, tuple):
            if len(max_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} max_lr, got {}".format(
                    len(optimizer.param_groups), len(max_lr)))
            self.max_lrs = list(max_lr)
        else:
            self.max_lrs = [max_lr] * len(optimizer.param_groups)

        self.step_size = step_size

        if mode not in ['triangular', 'triangular2', 'exp_range'] \
                and scale_fn is None:
            raise ValueError('mode is invalid and scale_fn is None')

        self.mode = mode
        self.gamma = gamma

        if scale_fn is None:
            if self.mode == 'triangular':
                self.scale_fn = self._triangular_scale_fn
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = self._triangular2_scale_fn
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = self._exp_range_scale_fn
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode

        self.batch_step(last_batch_iteration + 1)
        self.last_batch_iteration = last_batch_iteration

    def batch_step(self, batch_iteration=None):
        if batch_iteration is None:
            batch_iteration = self.last_batch_iteration + 1
        self.last_batch_iteration = batch_iteration
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

    def _triangular_scale_fn(self, x):
        return 1.

    def _triangular2_scale_fn(self, x):
        return 1 / (2. ** (x - 1))

    def _exp_range_scale_fn(self, x):
        return self.gamma**(x)

    def get_lr(self):
        step_size = float(self.step_size)
        cycle = np.floor(1 + self.last_batch_iteration / (2 * step_size))
        x = np.abs(self.last_batch_iteration / step_size - 2 * cycle + 1)

        lrs = []
        param_lrs = zip(self.optimizer.param_groups, self.base_lrs, self.max_lrs)
        for param_group, base_lr, max_lr in param_lrs:
            base_height = (max_lr - base_lr) * np.maximum(0, (1 - x))
            if self.scale_mode == 'cycle':
                lr = base_lr + base_height * self.scale_fn(cycle)
            else:
                lr = base_lr + base_height * self.scale_fn(self.last_batch_iteration)
            lrs.append(lr)
        return lrs
class Caps_Layer(nn.Module):#胶囊层
    def __init__(self, input_dim_capsule, num_capsule=5, dim_capsule=5, \
                 routings=4, kernel_size=(9, 1), share_weights=True,
                 activation='default', **kwargs):
        super(Caps_Layer, self).__init__(**kwargs)
        self.T_epsilon = 1e-7
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = 4
        self.kernel_size = kernel_size  # 暂时没用到
        self.share_weights = share_weights
        if activation == 'default':
            self.activation = self.squash
        else:
            self.activation = nn.ReLU(inplace=True)

        if self.share_weights:
            self.W = nn.Parameter(
                nn.init.xavier_normal_(torch.empty(1, input_dim_capsule, self.num_capsule * self.dim_capsule)))#[1, 256, 25]
        else:
            self.W = nn.Parameter(
                torch.randn(args.batch_size, input_dim_capsule, self.num_capsule * self.dim_capsule))  # 64即batch_size

    def forward(self, x):

        if self.share_weights:
            u_hat_vecs = torch.matmul(x, self.W) #[16, 100, 256]-->[16, 100, 25]
        else:
            print('add later')

        batch_size = x.size(0)
        input_num_capsule = x.size(1) #输入是100维度
        u_hat_vecs = u_hat_vecs.view((batch_size, input_num_capsule,  #[16, 100, 25] --> [16, 100, 5, 5]
                                      self.num_capsule, self.dim_capsule))
        u_hat_vecs = u_hat_vecs.permute(0, 2, 1, 3)  # 交换维度，即转成(batch_size,num_capsule,input_num_capsule,dim_capsule)  [16, 5, 100, 5]
        b = torch.zeros_like(u_hat_vecs[:, :, :, 0])  # (batch_size,num_capsule,input_num_capsule)[16, 5, 100]

        for i in range(self.routings):#允许的跳数:4
            b = b.permute(0, 2, 1)#[16,100,5]
            c = F.softmax(b, dim=2)
            c = c.permute(0, 2, 1)#[16, 100, 5]-->[16, 5, 100] !这是经过softmax的
            b = b.permute(0, 2, 1)#[16, 100, 5]-->[16, 5, 100]
            outputs = self.activation(torch.einsum('bij,bijk->bik', (c, u_hat_vecs)))  #先用einsum运算[16, 5, 100]与[16, 5, 100, 5] -->  [16, 5, 5]     batch matrix multiplication批量矩阵乘法
            # outputs shape (batch_size, num_capsule, dim_capsule)#(batch, 胶囊数量， 胶囊维度)
            if i < self.routings - 1:
                b = torch.einsum('bik,bijk->bij', (outputs, u_hat_vecs))  # 矩阵乘法：[16, 5, 5]*[16, 5, 100, 5]-->[16,5,100] batch matrix multiplication
        return outputs  # (batch_size, num_capsule, dim_capsule)[16, 5, 5]

    # text version of squash, slight different from original one
    def squash(self, x, axis=-1): #x:[16,5,5]
        s_squared_norm = (x ** 2).sum(axis, keepdim=True) #[16,5,5] --> [16, 5, 1]
        scale = torch.sqrt(s_squared_norm + self.T_epsilon)
        return x / scale
    
class Attention(nn.Module):
    def __init__(self, feature_dim, step_dim, bias=True, **kwargs):
        super(Attention, self).__init__(**kwargs)
        
        self.supports_masking = True

        self.bias = bias
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        self.features_dim = 0
        
        weight = torch.zeros(feature_dim, 1)
        nn.init.xavier_uniform_(weight)
        self.weight = nn.Parameter(weight)
        
        if bias:
            self.b = nn.Parameter(torch.zeros(step_dim))
        
    def forward(self, x, mask=None):#x:[16, 100, 256]
        feature_dim = self.feature_dim#256
        step_dim = self.step_dim#100

        eij = torch.mm(#
            x.contiguous().view(-1, feature_dim), #contiguous通常与view配套使用，用于变化维度[16, 100, 256]-->[1600, 256]
            self.weight
        ).view(-1, step_dim)#[1600, 256]*[256,1] --> [1600, 1] -->[16, 100]
        
        if self.bias:
            eij = eij + self.b
            
        eij = torch.tanh(eij)
        a = torch.exp(eij)#里面的值取对数[16, 100]
        
        if mask is not None:
            a = a * mask

        a = a / torch.sum(a, 1, keepdim=True) + 1e-10  #[16, 100]/[16, 1] + 很大的负数1x10负10次方-->[16,100]

        weighted_input = x * torch.unsqueeze(a, -1) #[16, 100, 256] * [16,100,1] -- > [16, 100, 256] 
        return torch.sum(weighted_input, 1)
#定义打印日志
def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    logger.removeHandler(sh)
    return logger

#定义模型    


class NeuralNet(nn.Module):
    def __init__(self,args, vocab_size, embedding_dim, embeddings=None):
        super(NeuralNet, self).__init__()
        self.num_classes = 35
        fc_layer = 256
        hidden_size = 128
        self.MAX_LEN = args.MAX_LEN
        self.Num_capsule = 5
        self.Dim_capsule = 5
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if embeddings:
            w2v_model = gensim.models.word2vec.Word2Vec.load(
                "/home/zyf/Summer game2021/Datafountain/competition-baseline-main/全球人工智能技术创新大赛/赛道一/embedding/w2v.model").wv
            fasttext_model = gensim.models.word2vec.Word2Vec.load(
                "/home/zyf/Summer game2021/Datafountain/competition-baseline-main/全球人工智能技术创新大赛/赛道一/embedding/fasttext.model").wv
            w2v_embed_matrix = w2v_model.vectors
            fasttext_embed_matrix = fasttext_model.vectors
#             embed_matrix = w2v_embed_matrix
            embed_matrix = np.concatenate(
                [w2v_embed_matrix, fasttext_embed_matrix], axis=1)
            oov_embed = np.zeros((1, embed_matrix.shape[1]))
            embed_matrix = torch.from_numpy(
                np.vstack((oov_embed, embed_matrix)))
            self.embedding.weight.data.copy_(embed_matrix)
            self.embedding.weight.requires_grad = False

        self.embedding_dropout = nn.Dropout2d(0.1)
        self.lstm = nn.GRU(embedding_dim, hidden_size, 2,
                           bidirectional=True, batch_first=True)
        self.gru = nn.GRU(hidden_size * 2, hidden_size, 2,
                          bidirectional=True, batch_first=True)
        self.tdbn = nn.BatchNorm2d(1) #块标准化的目的就是让传输的数据合理的分布，加速训练的过程,输入是：4维数据(N,C,H,W)
        self.lstm_attention = Attention(hidden_size * 2, self.MAX_LEN)
        self.gru_attention = Attention(hidden_size * 2, self.MAX_LEN)
        self.bn = nn.BatchNorm1d(fc_layer)
        self.linear = nn.Linear(hidden_size*8+1, fc_layer)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.output = nn.Linear(fc_layer, self.num_classes)
        self.lincaps = nn.Linear(self.Num_capsule *self.Dim_capsule, 1)
        self.caps_layer = Caps_Layer(hidden_size*2)

    def forward(self, x, label=None):
        #x:[16, 100]
        #         Capsule(num_capsule=10, dim_capsule=10, routings=4, share_weights=True)(x)
        h_embedding = self.embedding(x) #[16, 100] -> [16, 100, 400]
        h_embedding = torch.squeeze(
            self.embedding_dropout(torch.unsqueeze(h_embedding, 0)))#[16, 100, 400]-->[1, 16, 100, 400]-->[16, 100, 400]
        h_embedding = self.tdbn(h_embedding.unsqueeze(1)).squeeze(1)#归一化处理
        h_lstm, _ = self.lstm(h_embedding) #[16, 100, 400] --> [16, 100, 256]
        h_gru, _ = self.gru(h_lstm) #[16, 100, 400] --> [16, 100, 256]

        ##Capsule Layer
        content3 = self.caps_layer(h_gru)#通过胶囊网络[16, 100, 256] --> [16, 5, 5]
        content3 = self.dropout(content3)
        batch_size = content3.size(0)
        content3 = content3.view(batch_size, -1)#[16, 5, 5]-->[16, 25]
        content3 = self.relu(self.lincaps(content3))#[16, 25]-->[16, 1]

        ##Attention Layer
        h_lstm_atten = self.lstm_attention(h_lstm) #[16, 100, 256] --> [16, 256]
        h_gru_atten = self.gru_attention(h_gru) #[16, 100, 256] --> [16, 256]

        # global average pooling
        avg_pool = torch.mean(h_gru, 1) # [16, 100, 256] --> [16, 256]
        # global max pooling
        max_pool, _ = torch.max(h_gru, 1) #[16, 100, 256] --> [16, 256]

        conc = torch.cat((h_lstm_atten, h_gru_atten,  # [16, 1025]
                         content3, avg_pool, max_pool), 1)
        conc = self.relu(self.linear(conc)) # [16, 1025]-->[16, 256]
        conc = self.bn(conc)
        out = self.dropout(self.output(conc))#[16, 256]-->[16, 35]
        if label is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(out.view(-1, self.num_classes).float(),  #计算loss
                            label.view(-1, self.num_classes).float())
            return loss
        else:
            return out

def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds, average='macro')
    return acc, f1

loss_fun = nn.BCEWithLogitsLoss()
def validation_funtion(model, valid_dataloader, valid_torchdata, mode='valid'):
    model.eval()
    pred_list = []
    labels_list = []
    if mode == 'valid':
        for i, (description, label) in enumerate(tqdm(valid_dataloader)):
            output = model(description.to(DEVICE))
            pred_list.extend(output.sigmoid().detach().cpu().numpy())#同时记录预测标签，以及正确标签
            labels_list.extend(label.detach().cpu().numpy())

        labels_arr = np.array(labels_list)#tensor转为numpy
        pred_arr = np.array(pred_list)
        labels = np.argmax(labels_arr, axis=1)#numpy.argmax(array, axis) 用于返回一个numpy数组中最大值的索引值。当一组中同时出现几个最大值时，返回第一个最大值的索引值
        pred = np.argmax(pred_arr, axis=1)
        acc, f1 = acc_and_f1(pred, labels)

        loss = loss_fun(torch.FloatTensor(labels_arr),
                        torch.FloatTensor(pred_arr))
        return acc, f1, loss
    else:
        for i, (description) in enumerate(tqdm(valid_dataloader)):
            output = model(description.to(DEVICE))
            pred_list += output.sigmoid().detach().cpu().numpy().tolist()
        return pred_list

                            
def train(args, model, train_dataloader, valid_dataloader, valid_torchdata, epochs, model_num, early_stop=None):
#     ema = EMA(model, 0.999)
#     ema.register()
    param_optimizer = list(model.named_parameters())
    embed_pa = ['embedding.weight']
    optimizer_grouped_parameters = [{'params': [p for n, p in param_optimizer if not any(nd in n for nd in embed_pa)]},
                                    {'params': model.embedding.parameters(), 'lr': 5e-5}]
    optimizer = AdamW(optimizer_grouped_parameters, lr=1e-3,amsgrad=True, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=3, T_mult=2, eta_min=1e-5, last_epoch=-1)
#     scheduler = CyclicLR(optimizer, base_lr=1e-3, max_lr=3e-3,
#                step_size=30, mode='exp_range',
#                gamma=0.99994)
#     opt = SWA(optimizer, swa_start=100, swa_freq=5, swa_lr=1e-4)
    total_loss = []
    train_loss = []
    best_val_loss = np.inf
    best_acc = np.inf
    best_f1 = np.inf
    best_loss = np.inf
    no_improve = 0
    for epoch in range(epochs):
        model.train()
        fgm = FGM(model)
        bar = tqdm(train_dataloader)
        for i, (description, label) in enumerate(bar):
            optimizer.zero_grad()
            output = model(description.to(DEVICE), label.to(DEVICE))
            loss = output
            loss.backward()
            train_loss.append(loss.item())

            #对抗训练
            fgm.attack() # 在embedding上添加对抗扰动
            loss_adv = model(description.to(DEVICE), label.to(DEVICE))
            loss_ad = loss_adv
            loss_ad.backward()# 反向传播，并在正常的grad基础上，累加对抗训练的梯度
            fgm.restore()# 恢复embedding参数
            
            scheduler.step(epochs + i / len(train_dataloader))
#             scheduler.batch_step()
            optimizer.step()
#             ema.update()
            bar.set_postfix(tloss=np.array(train_loss).mean())
#         opt.swap_swa_sgd()
#         ema.apply_shadow()
        acc, f1, val_loss = validation_funtion(
            model, valid_dataloader, valid_torchdata, 'valid')
#         ema.restore()
        print('train_loss: {:.5f}, val_loss: {:.5f}, acc: {:.5f}, f1: {:.5f}\n'.format(
            train_loss[-1], val_loss, acc, f1))
        if early_stop:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_acc = acc
                best_f1 = f1
                best_loss = train_loss[-1]
#                 ema.apply_shadow()
                torch.save(model.state_dict(),'/home/zyf/Summer game2021/Datafountain/competition-baseline-main/全球人工智能技术创新大赛/赛道一/learned_model/{}_model_{}.bin'.format(args.NAME, model_num))
#                 ema.restore()
                model_num += 1
            else:
                no_improve += 1
            if no_improve == early_stop:
                model_num += 1
                break
            if epoch == epochs-1:
                model_num += 1
        else:
            if epoch >= epochs-1:
                torch.save(model.state_dict(),'/home/zyf/Summer game2021/Datafountain/competition-baseline-main/全球人工智能技术创新大赛/赛道一/learned_model/{}_model_{}.bin'.format(args.NAME, model_num))
                model_num += 1
    return best_val_loss, best_acc, best_f1, best_loss, model_num


def run(args, train_dataset, w2v_model, fasttext_model):
    kf = StratifiedKFold(n_splits=args.FOLD, shuffle=True, random_state=args.SEED)
    best_mlogloss = []
    best_acc = []
    best_f1 = []
    best_loss = []
    model_num = args.model_num
    model = NeuralNet(args, w2v_model.wv.vectors.shape[0]+1, w2v_model.wv.vectors.shape[1]+fasttext_model.wv.vectors.shape[1], embeddings=True)
    model.to(DEVICE)
    for i, (train_index, test_index) in enumerate(kf.split(np.arange(len(train_dataset)), [-1]*(len(train_dataset)))):
        print(str(i+1), '-'*50)
        tra = [train_dataset[index] for index in train_index]
        val = [train_dataset[index] for index in test_index]
        print(len(tra))
        print(len(val))
        train_dataloader, train_torchdata = get_dataloader(args,tra, mode='train')
        valid_dataloader, valid_torchdata = get_dataloader(args,val, mode='valid')
    
        mlogloss, acc, f1, loss, model_n = train(args, model, train_dataloader,
                                    valid_dataloader,
                                    valid_torchdata,
                                    args.epochs,
                                    model_num,
                                    early_stop=5)
        torch.cuda.empty_cache()
        best_mlogloss.append(mlogloss)
        best_acc.append(acc)
        best_f1.append(f1)
        best_loss.append(loss)
    for i in range(args.FOLD):
        print('- 第{}折中，best mlogloss: {}   best acc: {}  best f1:{}  best loss: {}'.format(i+1, best_mlogloss[i], best_acc[i], best_f1[i], best_loss[i]))
    return model_n

def get_submit(args, test_data, test_dataset, id2label, model_num):
    model = NeuralNet(args, w2v_model.wv.vectors.shape[0]+1,w2v_model.wv.vectors.shape[1]+fasttext_model.wv.vectors.shape[1],embeddings=True)
    model.to(DEVICE)
    test_preds_total = []
    test_dataloader, test_torchdata = get_dataloader(args, test_dataset, mode='test')
    for i in range(1,model_num):
        model.load_state_dict(torch.load('/home/zyf/Summer game2021/Datafountain/competition-baseline-main/全球人工智能技术创新大赛/赛道一/learned_model/{}_model_{}.bin'.format(args.NAME, i)))
        test_pred_results = validation_funtion(model, test_dataloader, test_torchdata, 'test')
        test_preds_total.append(test_pred_results)
    test_preds_merge = np.sum(test_preds_total, axis=0) / (model_num-1)
    test_pre_tensor = torch.tensor(test_preds_merge)
    test_pre = torch.max(test_pre_tensor,1)[1]

    pred_labels = [id2label[i] for i in test_pre]
    submit_file = '/home/zyf/Summer game2021/Datafountain/submits/submit.csv'

    pd.DataFrame({"id": test_data['id'], "label": pred_labels}).to_csv(
        submit_file, index=False)


def arg_setting():
    parser = argparse.ArgumentParser()
    parser.add_argument('--NAME', default='capsuleNet', type=str, help="")
    parser.add_argument('--MAX_LEN', default=100, type=int,
                        help='max length of sentence')
    parser.add_argument('--batch_size', default=16, type=int, help='')
    parser.add_argument('--SEED', default=9797, type=int, help='')
    parser.add_argument('--FOLD', default=5, type=int, help="k fold")
    parser.add_argument('--epochs', default=10, type=int, help="")
    parser.add_argument('--model_num', default=1, type=int, help='')

    args = parser.parse_args()
    return args

if __name__=='__main__':
    args = arg_setting() #设置基本参数
    basic_setting(args.SEED, DEVICE) #设置随机种子
    test_data, train_dataset, test_dataset, w2v_model, fasttext_model, id2label = data_process()
    #开始训练
    model_num = run(args, train_dataset, w2v_model, fasttext_model)
    #获得提交结果文件
    get_submit(args, test_data, test_dataset, id2label, model_num)
