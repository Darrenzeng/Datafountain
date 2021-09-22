import pandas as pd
import numpy as np
import collections
import re
import torch
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import log_loss, roc_auc_score
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
from sklearn.metrics import f1_score
from adversarial_model import FGM, PGD
from torch import nn
import torch.nn.functional as F
from torch.optim import *
from model.han import HAN, HanConfig
from run_config import RunConfig
import pickle
import sys
# sys.path.append("/home/zyf/Summer game2021/Datafountain/competition-baseline-main/全球人工智能技术创新大赛/赛道一")
torch.set_printoptions(edgeitems=768)
warnings.filterwarnings("ignore")
np.set_printoptions(threshold=np.inf)
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

SEED =2021

random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if DEVICE != 'cpu':
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def data_process():
    train_data = pd.read_csv(
        "./datasets/datagrand_2021_train.csv")
    test_data = pd.read_csv(
        "./datasets/datagrand_2021_test.csv")

    id2label = list(train_data["label"].unique())
    label2id = {id2label[i]: i for i in range(len(id2label))}
    with open('./model/id2label.pkl', 'wb') as f:
        pickle.dump(id2label, f)
    with open('./model/label2id.pkl', 'wb') as f:
        pickle.dump(label2id, f)

    y_train = np.zeros((len(train_data), len(id2label)), dtype=np.int8)

    all_sentences = pd.concat(
        [train_data['text'], test_data['text']]).reset_index(drop=True)

    all_sentences.drop_duplicates().reset_index(drop=True, inplace=True)
    all_sentences = all_sentences.apply(lambda x: x.split(' ')).tolist()
    #获得训练w2v模型
    if not os.path.exists('./embedding/w2v.model'):
        w2v_model = gensim.models.word2vec.Word2Vec(
            all_sentences, sg=1, vector_size=300, window=7, min_count=1, negative=3, sample=0.001, hs=1, seed=452)
        w2v_model.save('./embedding/w2v.model')
    else:
        w2v_model = gensim.models.word2vec.Word2Vec.load(
            "./embedding/w2v.model")
    #获得fasttext模型
    if not os.path.exists('./embedding/fasttext.model'):
        fasttext_model = gensim.models.FastText(
            all_sentences, seed=452, vector_size=100, min_count=1, epochs=20, window=2)
        fasttext_model.save('./embedding/fasttext.model')
    else:
        fasttext_model = gensim.models.word2vec.Word2Vec.load(
            "./embedding/fasttext.model")
    train_dataset = []

    ylabel = []
    for i in tqdm(range(len(train_data))):
        train_dict = {}
        train_dict['id'] = train_data.loc[i, 'id']
        train_dict['text'] = train_data.loc[i, 'text']
        y_train[i][label2id[train_data.loc[i, 'label']]] = 1
        train_dict['label'] = y_train[i]

        ylabel.append(train_data.loc[i, 'label'])
        train_dataset.append(train_dict)
    test_dataset = []
    for i in tqdm(range(len(test_data))):
        test_dict = {}
        test_dict['id'] = test_data.loc[i, 'id']
        test_dict['text'] = test_data.loc[i, 'text']
        test_dict['label'] = -1
        test_dataset.append(test_dict)
    return test_data, train_dataset, test_dataset, w2v_model, fasttext_model, id2label, ylabel


class DataSet(data.Dataset):
    def __init__(self, args, data, mode='train'):
        self.data = data
        self.mode = mode
        self.w2v_model = gensim.models.word2vec.Word2Vec.load("./embedding/w2v.model")
        self.dataset = self.get_data(self.data, self.mode)

    def get_data(self, data, mode):
        dataset = []
        global s
        for data_li in tqdm(data):
            text = data_li['text'].split(' ')
            text = [self.w2v_model.wv.key_to_index[s] +1 if s in self.w2v_model.wv else 0 for s in text]
            if len(text) < args.MAX_LEN:
                text += [0] * (args.MAX_LEN - len(text))
            else:
                text = text[:args.MAX_LEN]
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
            torchdata, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=False)
    elif mode == 'valid':
        dataloader = torch.utils.data.DataLoader(
            torchdata, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=True)
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
        param_lrs = zip(self.optimizer.param_groups,
                        self.base_lrs, self.max_lrs)
        for param_group, base_lr, max_lr in param_lrs:
            base_height = (max_lr - base_lr) * np.maximum(0, (1 - x))
            if self.scale_mode == 'cycle':
                lr = base_lr + base_height * self.scale_fn(cycle)
            else:
                lr = base_lr + base_height * \
                    self.scale_fn(self.last_batch_iteration)
            lrs.append(lr)
        return lrs


class Caps_Layer(nn.Module):  # 胶囊层
    def __init__(self, input_dim_capsule, num_capsule=5, dim_capsule=5,
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
                nn.init.xavier_normal_(torch.empty(1, input_dim_capsule, self.num_capsule * self.dim_capsule)))
        else:
            self.W = nn.Parameter(
                torch.randn(args.batch_size, input_dim_capsule, self.num_capsule * self.dim_capsule))  # 64即batch_size

    def forward(self, x):

        if self.share_weights:
            # [16, 100, 256]-->[16, 100, 25]
            u_hat_vecs = torch.matmul(x, self.W)
        else:
            print('add later')

        batch_size = x.size(0)
        input_num_capsule = x.size(1)  # 输入是100维度
        u_hat_vecs = u_hat_vecs.view((batch_size, input_num_capsule,  # [16, 100, 25] --> [16, 100, 5, 5]
                                      self.num_capsule, self.dim_capsule))
        # 交换维度，即转成(batch_size,num_capsule,input_num_capsule,dim_capsule)  [16, 5, 100, 5]
        u_hat_vecs = u_hat_vecs.permute(0, 2, 1, 3)
        # (batch_size,num_capsule,input_num_capsule)[16, 5, 100]
        b = torch.zeros_like(u_hat_vecs[:, :, :, 0])

        for i in range(self.routings):
            b = b.permute(0, 2, 1)
            c = F.softmax(b, dim=2)
            c = c.permute(0, 2, 1)
            b = b.permute(0, 2, 1)
            # batch matrix multiplication
            outputs = self.activation(torch.einsum(
                'bij,bijk->bik', (c, u_hat_vecs)))
            # outputs shape (batch_size, num_capsule, dim_capsule)
            if i < self.routings - 1:
                # batch matrix multiplication
                b = torch.einsum('bik,bijk->bij', (outputs, u_hat_vecs))
        return outputs  # (batch_size, num_capsule, dim_capsule)[16, 5, 5]

    # text version of squash, slight different from original one
    def squash(self, x, axis=-1):
        s_squared_norm = (x ** 2).sum(axis, keepdim=True)
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

    def forward(self, x, mask=None):
        feature_dim = self.feature_dim
        step_dim = self.step_dim

        eij = torch.mm(
            x.contiguous().view(-1, feature_dim),
            self.weight
        ).view(-1, step_dim)

        if self.bias:
            eij = eij + self.b

        eij = torch.tanh(eij)
        a = torch.exp(eij)

        if mask is not None:
            a = a * mask

        a = a / torch.sum(a, 1, keepdim=True) + 1e-10

        weighted_input = x * torch.unsqueeze(a, -1)  # [16, 100, 256]
        return torch.sum(weighted_input, 1)

#定义模型
class SelfAttention(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SelfAttention, self).__init__()
        self.W = nn.Linear(input_size, hidden_size, True)
        self.u = nn.Linear(hidden_size, 1)

    def forward(self, x):
        u = torch.tanh(self.W(x))
        a = F.softmax(self.u(u), dim=1)
        x = a.mul(x).sum(1)
        return x

loss_fun = nn.BCEWithLogitsLoss()



def cal_macro_f1(y_true, y_pred):
    score = f1_score(y_true, y_pred, average='macro')
    return score
def validation_funtion(model, valid_dataloader, valid_torchdata, mode='valid'):
    model.eval()
    pred_list = []
    labels_list = []
    val_loss = []
    y_preds = []
    y_trues = []
    with torch.no_grad():
        if mode == 'valid':
            for i, (description, label) in enumerate(tqdm(valid_dataloader)):
                output = model(description.to(DEVICE))
                loss = loss_fun(output.sigmoid(), label.float().to(DEVICE))
                y_pred = torch.max(output.sigmoid(), 1)[1].cpu().tolist()
                y_label = torch.max(label, 1)[1].cpu().tolist()
                y_trues.extend(y_label)
                y_preds.extend(y_pred)
                val_loss.append(loss.item())

            auc = cal_macro_f1(y_trues, y_preds)

            return auc, np.mean(val_loss)
        else:
            for i, (description) in enumerate(tqdm(valid_dataloader)):
                output = model(description.to(DEVICE))
                pred_list += output.sigmoid().detach().cpu().numpy().tolist()
            return pred_list


def train(args, model, train_dataloader, valid_dataloader, valid_torchdata, epochs, model_num, early_stop=None):
    #     ema = EMA(model, 0.999)
    #     ema.register()
    param_optimizer = list(model.named_parameters())
    embed_pa = ['embed.weight']
    optimizer_grouped_parameters = [{'params': [p for n, p in param_optimizer if not any(nd in n for nd in embed_pa)]},
                                    {'params': model.embed.parameters(), 'lr': 2e-5}]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr,amsgrad=True, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=3, T_mult=2, eta_min=1e-5, last_epoch=-1)
#     scheduler = CyclicLR(optimizer, base_lr=1e-3, max_lr=3e-3,
#                step_size=30, mode='exp_range',
#                gamma=0.99994)
#     opt = SWA(optimizer, swa_start=100, swa_freq=5, swa_lr=1e-4)
   
    best_val_loss = np.inf
    best_f1 = np.inf
    best_loss = np.inf
    no_improve = 0

#     for param in model.named_parameters():
#         print(param[0])

    for epoch in range(epochs):
        model.train()
        train_loss = []
        if epoch > 2:
            for param in model.named_parameters():
                if param[0] == 'embed.weight':
                    param[1].requires_grad = True
                    break

        # fgm = FGM(model, emb_name="embed.weight")
#         pgd = PGD(model, emb_name="embed.weight")
        bar = tqdm(train_dataloader)
        for i, (description, label) in enumerate(bar):
            optimizer.zero_grad()
            output = model(description.to(DEVICE), label.to(DEVICE))
            loss = output
            loss.backward()
            train_loss.append(loss.item())
####################################################
#打开这下面两个中的一个即可
##################################################
            #fgm对抗训练
            # fgm.attack() # 在embedding上添加对抗扰动
            # loss_adv = model(description.to(DEVICE), lab·el.to(DEVICE))
            # # loss_adv = loss_adv.mean()
            # loss_adv.backward()# 反向传播，并在正常的grad基础上，累加对抗训练的梯度
            # fgm.restore()# 恢复embedding参数

            # pgd对抗训练
#             K=3
#             pgd.backup_grad()
#             for t in range(K):
#                 pgd.attack(is_first_attack=(t==0)) # 在embedding上添加对抗扰动, first attack时备份param.processor
#                 if t != K-1:
#                     model.zero_grad()
#                 else:
#                     pgd.restore_grad()
#                 loss_adv = model(description.to(DEVICE), label.to(DEVICE))
# #                 loss_adv = loss_adv.mean()
#                 loss_adv.backward() # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
#             pgd.restore() # 恢复embedding参数

            scheduler.step(epochs + i / len(train_dataloader))
#             scheduler.batch_step()
            optimizer.step()
#             ema.update()
            # bar.set_postfix(tloss=np.array(train_loss).mean())
#         opt.swap_swa_sgd()
#         ema.apply_shadow()
        f1, val_loss = validation_funtion(model, valid_dataloader, valid_torchdata, 'valid')
#         ema.restore()
        print('Epoch:[{}/{}] train_loss: {:.5f}, val_loss: {:.5f},f1-score: {:.5f}\n'.format(epoch,epochs,np.mean(train_loss), val_loss, f1))
        if early_stop:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_f1 = f1
                best_loss = np.mean(train_loss)
#                 ema.apply_shadow()
                torch.save(model.state_dict(), './model/saved/{}_model_{}.bin'.format(args.NAME, model_num))#保存模型
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
                torch.save(
                    model.state_dict(), './model/saved/{}_model_{}.bin'.format(args.NAME, model_num))#保存模型
                model_num += 1
    return best_val_loss, best_f1, best_loss, model_num

def run(args, train_dataset, w2v_model, fasttext_model, ylabel):
    kf = StratifiedKFold(n_splits=args.FOLD, shuffle=True,random_state=SEED)
    best_mlogloss = []
    best_f1 = []
    best_loss = []
    model_num = 1

    for i, (train_index, test_index) in enumerate(kf.split(np.arange(len(train_dataset)), ylabel)):
        print(str(i+1), '-'*50)
        tra = [train_dataset[index] for index in train_index]
        val = [train_dataset[index] for index in test_index]
        print(len(tra))
        print(len(val))
        #将w2v和fasttext模型的向量进行拼接
        run_config = RunConfig()#加载训练参数，如epoch, batch_size
        model_config = HanConfig()
        model = HAN(run_config, model_config)
        model.to(DEVICE)
        train_dataloader, train_torchdata = get_dataloader(
            args, tra, mode='train')
        valid_dataloader, valid_torchdata = get_dataloader(
            args, val, mode='valid')

        mlogloss, f1, loss, model_n = train(args, model, train_dataloader,
                                             valid_dataloader,
                                             valid_torchdata,
                                             args.epochs,
                                             model_num,
                                             early_stop=args.early_step)
        torch.cuda.empty_cache()
        best_mlogloss.append(mlogloss)
        best_f1.append(f1)
        best_loss.append(loss)

    for i in range(args.FOLD):
        print('- 第{}折中，best valloss: {}   best f1-socre: {}   best trainloss: {}'.format(i +1, best_mlogloss[i], best_f1[i], best_loss[i]))
    return model_n


def get_submit(args, test_data, test_dataset, id2label, model_num):
    run_config = RunConfig()#加载训练参数，如epoch, batch_size
    model_config = HanConfig()
    model = HAN(run_config, model_config)
    model.to(DEVICE)
    test_preds_total = []
    test_dataloader, test_torchdata = get_dataloader(
        args, test_dataset, mode='test')
    for i in range(1, model_num):
        model.load_state_dict(torch.load(
            './model/saved/{}_model_{}.bin'.format(args.NAME, i)))
        test_pred_results = validation_funtion(
            model, test_dataloader, test_torchdata, 'test')
        test_preds_total.append(test_pred_results)
    test_preds_merge = np.sum(test_preds_total, axis=0) / (model_num-1)
    test_pre_tensor = torch.tensor(test_preds_merge)
    test_pre = torch.max(test_pre_tensor, 1)[1]

    pred_labels = [id2label[i] for i in test_pre]
    submit_file = '/home/zyf/Summer game2021/Datafountain/submits/submit_HAN_base1.csv'
    # submit_file = "./submit/submit_{}.csv".format(args.NAME)

    pd.DataFrame({"id": test_data['id'], "label": pred_labels}).to_csv(
        submit_file, index=False)


def arg_setting():
    parser = argparse.ArgumentParser()
    parser.add_argument('--NAME', default='han', type=str, help="")
    parser.add_argument('--MAX_LEN', default=100, type=int,
                        help='max length of sentence')
    parser.add_argument('--batch_size', default=32, type=int, help='')
    parser.add_argument('--FOLD', default=10, type=int, help="k fold")
    parser.add_argument('--epochs', default=10, type=int, help="")
    parser.add_argument('--early_step', default=5, type=int, help="")
    parser.add_argument('--lr', default=1e-3, type=float, help="")
    args = parser.parse_args()
    return args



if __name__ == '__main__':
    args = arg_setting()  # 设置基本参数
    test_data, train_dataset, test_dataset, w2v_model, fasttext_model, id2label, ylabel = data_process()
    #开始训练
    model_num = run(args, train_dataset, w2v_model, fasttext_model, ylabel)
    #获得提交结果文件
    get_submit(args, test_data, test_dataset, id2label, model_num-1)
