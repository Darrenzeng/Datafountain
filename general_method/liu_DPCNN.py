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
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from utils.spatial_dropout import SpatialDropout
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
from torch import nn
import torch.nn.functional as F
from torch.optim import *
from adversarial_model import FGM, PGD
from utils.init_net import init_network
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
    # train_data = pd.read_csv("/media/mgege007/winType/DaGuan/data/datagrand_2021_train.csv")
    train_data = pd.read_csv("/media/mgege007/winType/DaGuan/data/pseudo_train_data.csv")
    # train_data1 = pd.read_csv(
    #     "/media/mgege007/winType/DaGuan/data/newdataRandDelete.csv", index_col=0)
    # train_data2 = pd.read_csv(
    #     "/media/mgege007/winType/DaGuan/data/newdataRandSwap.csv", index_col=0)
    # train_data3 = pd.read_csv(
    #     "/media/mgege007/winType/DaGuan/data/newdataReversed.csv", index_col=0)
    # train_data = pd.concat(
    #     [train_data1, train_data2[int(len(train_data2)/2):], train_data3[int(len(train_data3)/2):]]).reset_index(drop=True)
    test_data = pd.read_csv(
        "/media/mgege007/winType/DaGuan/data/datagrand_2021_test.csv")
    id2label = list(train_data['label'].unique())
    label2id = {id2label[i]: i for i in range(len(id2label))}
    y_train = np.zeros((len(train_data), len(id2label)), dtype=np.int8)

    all_sentences = pd.concat(
        [train_data['text'], test_data['text']]).reset_index(drop=True)
    all_sentences.drop_duplicates().reset_index(drop=True, inplace=True)
    all_sentences = all_sentences.apply(lambda x: x.split(' ')).tolist()
    if not os.path.exists('./embedding/w2v.model'):
        w2v_model = gensim.models.word2vec.Word2Vec(
            all_sentences, sg=1, vector_size=300, window=7, min_count=1, negative=3, sample=0.001, hs=1, seed=452)
        w2v_model.save('./embedding/w2v.model')
    else:
        w2v_model = gensim.models.word2vec.Word2Vec.load(
            "./embedding/w2v.model")

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
        train_dict['text'] = train_data.loc[i, 'text']
        y_train[i][label2id[train_data.loc[i, 'label']]] = 1
        train_dict['label'] = y_train[i]
        # ylabel.append(y_train[i])
        ylabel.append(train_data.loc[i, 'label'])
        train_dataset.append(train_dict)
    test_dataset = []
    for i in tqdm(range(len(test_data))):
        test_dict = {}
        test_dict['text'] = test_data.loc[i, 'text']
        test_dict['label'] = -1
        test_dataset.append(test_dict)
    return test_data, train_dataset, test_dataset, w2v_model, fasttext_model, id2label, ylabel


class DataSet(data.Dataset):
    def __init__(self, args, data, mode='train'):
        self.data = data
        self.mode = mode
        self.dataset = self.get_data(self.data, self.mode)

    def get_data(self, data, mode):
        dataset = []
        global s
        for data_li in tqdm(data):
            text = data_li['text'].split(' ')
            text = [w2v_model.wv.key_to_index[s] +
                    1 if s in w2v_model.wv else 0 for s in text]
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
            torchdata, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    elif mode == 'test':
        dataloader = torch.utils.data.DataLoader(
            torchdata, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)
    elif mode == 'valid':
        dataloader = torch.utils.data.DataLoader(
            torchdata, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=True)
    return dataloader, torchdata

# 定义模型


class DPCNN(nn.Module):
    def __init__(self, args, vocab_size, embedding_dim, embeddings=None):
        super(DPCNN, self).__init__()
        self.dropout = 0.5  # 随机失活
        self.require_improvement = 1000  # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = 35  # 类别数
        self.learning_rate = 1e-3  # 学习率
        self.num_filters = 250  # 卷积核数量(channels数)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        hidden_size = 128
        # 字向量维度
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        if embeddings:
            w2v_model = gensim.models.word2vec.Word2Vec.load(
                "./embedding/w2v.model").wv
            fasttext_model = gensim.models.word2vec.Word2Vec.load(
                "./embedding/fasttext.model"
            ).wv
            w2v_embed_matrix = w2v_model.vectors
            fasttext_embed_matrix = fasttext_model.vectors
            #             embed_matrix = w2v_embed_matrix
            embed_matrix = np.concatenate(
                [w2v_embed_matrix, fasttext_embed_matrix], axis=1
            )
            oov_embed = np.zeros((1, embed_matrix.shape[1]))
            embed_matrix = torch.from_numpy(
                np.vstack((oov_embed, embed_matrix)))
            self.embedding.weight.data.copy_(embed_matrix)
            self.embedding.weight.requires_grad = False

        self.spatial_dropout = SpatialDropout(drop_prob=0.5)

        self.conv_region = nn.Conv2d(
            1, self.num_filters, (3, self.embedding_dim))

        self.conv = nn.Conv2d(self.num_filters, self.num_filters, (3, 1))

        self.max_pool = nn.MaxPool2d(kernel_size=(3, 1), stride=2)

        self.padding1 = nn.ZeroPad2d((0, 0, 1, 1))

        self.padding2 = nn.ZeroPad2d((0, 0, 0, 1))

        self.relu = nn.ReLU()
        # 全连接方式二选一
        self.fc = nn.Linear(self.num_filters, self.num_classes)

    def forward(self, x, label=None):
        x = self.embedding(x)
        x = self.spatial_dropout(x)
        x = x.unsqueeze(1)
        x = self.conv_region(x)
        x = self.padding1(x)
        x = self.relu(x)
        x = self.conv(x)
        x = self.padding1(x)
        x = self.relu(x)
        x = self.conv(x)
        while x.size()[2] > 2:
            x = self._block(x)
        x = x.squeeze(-1)
        x = x.squeeze(-1)
        out = self.fc(x)
        if label is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(out.view(-1, self.num_classes).float(),
                            label.view(-1, self.num_classes).float())
            return loss
        else:
            return out

    def _block(self, x):
        x = self.padding2(x)

        px = self.max_pool(x)

        x = self.padding1(px)

        x = F.relu(x)

        x = self.conv(x)

        x = self.padding1(x)

        x = F.relu(x)

        x = self.conv(x)

        x = x + px

        return x


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

loss_fun = nn.BCEWithLogitsLoss()


def validation_funtion(model, valid_dataloader, valid_torchdata, mode='valid'):
    model.eval()
    pred_list = []
    labels_list = []
    if mode == 'valid':
        for i, (description, label) in enumerate(tqdm(valid_dataloader)):
            output = model(description.to(DEVICE))
            pred_list.extend(output.sigmoid().detach().cpu().numpy())
            labels_list.extend(label.detach().cpu().numpy())
        labels_arr = np.array(labels_list)
        pred_arr = np.array(pred_list)
        labels = np.argmax(labels_arr, axis=1)
        pred = np.argmax(pred_arr, axis=1)
        auc = f1_score(labels, pred, average='macro')
        loss = loss_fun(torch.FloatTensor(labels_arr),
                        torch.FloatTensor(pred_arr))
        return auc, loss
    else:
        for i, (description) in enumerate(tqdm(valid_dataloader)):
            output = model(description.to(DEVICE))
            pred_list += output.sigmoid().detach().cpu().numpy().tolist()
        return pred_list


def train(args, fold, model, train_dataloader, valid_dataloader, valid_torchdata, epochs, model_num, early_stop=None):
    #     ema = EMA(model, 0.999)
    #     ema.register()
    param_optimizer = list(model.named_parameters())
    embed_pa = ['embedding.weight']
    optimizer_grouped_parameters = [{'params': [p for n, p in param_optimizer if not any(nd in n for nd in embed_pa)]},
                                    {'params': model.embedding.parameters(), 'lr': 5e-5}]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.lr, amsgrad=True, weight_decay=5e-4)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-5, last_epoch=-1)
#     scheduler = CyclicLR(optimizer, base_lr=1e-3, max_lr=3e-3,
#                step_size=30, mode='exp_range',
#                gamma=0.99994)
#     opt = SWA(optimizer, swa_start=100, swa_freq=5, swa_lr=1e-4)
    total_loss = []
    train_loss = []
    best_val_loss = np.inf
    best_auc = np.inf
    best_loss = np.inf
    no_improve = 0
    for epoch in range(epochs):
        model.train()
#         fgm = FGM(model)
        bar = tqdm(train_dataloader)
        for i, (description, label) in enumerate(bar):
            optimizer.zero_grad()
            output = model(description.to(DEVICE), label.to(DEVICE))
            loss = output
            loss.backward()
            train_loss.append(loss.item())

            scheduler.step()
            # scheduler.step(epochs + i / len(train_dataloader))
#             scheduler.batch_step()
            optimizer.step()

        auc, val_loss = validation_funtion(
            model, valid_dataloader, valid_torchdata, 'valid')
        print('Epoch:[{}/{}] train_loss: {:.5f}, val_loss: {:.5f},f1-score: {:.5f}\n'.format(
            epoch+1, epochs, np.mean(train_loss), val_loss, auc))

        if early_stop:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_auc = auc
                best_loss = train_loss[-1]
#                 ema.apply_shadow()
                torch.save(
                    model.state_dict(), './saved/{}_model_{}.bin'.format(args.NAME, model_num))
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
                    model.state_dict(), './saved/{}_model_{}.bin'.format(args.NAME, model_num))
                model_num += 1
    torch.save(model.state_dict(),
               './model/{}_model_{}.bin'.format(args.NAME, fold))
    return best_val_loss, best_auc, best_loss, model_num

# 初始网络


def run(args, train_dataset, w2v_model, fasttext_model, ylabel):
    kf = StratifiedKFold(n_splits=args.FOLD, shuffle=True, random_state=2021)
    # kf = MultilabelStratifiedKFold(n_splits=args.FOLD, shuffle=True, random_state=2021)
    best_mlogloss = []
    best_auc = []
    best_loss = []
    model_num = 1
    for i, (train_index, test_index) in enumerate(kf.split(np.arange(len(train_dataset)), ylabel)):
        model = DPCNN(
            args, w2v_model.wv.vectors.shape[0]+1, w2v_model.wv.vectors.shape[1]+fasttext_model.wv.vectors.shape[1], embeddings=True)
        #init_network(model)
        model.to(DEVICE)
        print(str(i+1), '-'*50)
        tra = [train_dataset[index] for index in train_index]
        val = [train_dataset[index] for index in test_index]
        print(len(tra))
        print(len(val))
        train_dataloader, train_torchdata = get_dataloader(
            args, tra, mode='train')
        valid_dataloader, valid_torchdata = get_dataloader(
            args, val, mode='valid')

        mlogloss, auc, loss, model_n = train(args, i, model, train_dataloader,
                                             valid_dataloader,
                                             valid_torchdata,
                                             args.epochs,
                                             model_num,
                                             early_stop=args.early_stop)
        torch.cuda.empty_cache()
        best_mlogloss.append(mlogloss)
        best_auc.append(auc)
        best_loss.append(loss)
        best_auc.append(auc)
    for i in range(args.FOLD):
        print('- 第{}折中，best mlogloss: {}   best auc: {}   best loss: {}'.format(i +
              1, best_mlogloss[i], best_auc[i], best_loss[i]))
    return model_n


def get_submit(args, test_data, test_dataset, id2label, model_num):
    model = DPCNN(
        args, w2v_model.wv.vectors.shape[0]+1, w2v_model.wv.vectors.shape[1]+fasttext_model.wv.vectors.shape[1], embeddings=True)
    model.to(DEVICE)
    test_preds_total = []
    test_dataloader, test_torchdata = get_dataloader(
        args, test_dataset, mode='test')
    for i in range(0, args.FOLD):
        model.load_state_dict(torch.load(
            './model/{}_model_{}.bin'.format(args.NAME, i)))
        test_pred_results = validation_funtion(
            model, test_dataloader, test_torchdata, 'test')
        test_preds_total.append(test_pred_results)
    test_preds_merge = np.sum(test_preds_total, axis=0) / (model_num-1)
    test_pre_tensor = torch.tensor(test_preds_merge)
    test_pre = torch.max(test_pre_tensor, 1)[1]

    pred_labels = [id2label[i] for i in test_pre]
    # submit_file = '/home/zyf/Summer game2021/Datafountain/submits/submit.csv'
    submit_file = "./submit/submit_{}.csv".format(args.NAME)

    pd.DataFrame({"id": test_data['id'], "label": pred_labels}).to_csv(
        submit_file, index=False)


def arg_setting():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--NAME', default='DPCNN_epoch40_10fold_20earlystop_pseudo_18395', type=str, help="")
    parser.add_argument('--MAX_LEN', default=100, type=int,
                        help='max length of sentence')
    parser.add_argument('--batch_size', default=32, type=int, help='')
    parser.add_argument('--SEED', default=9797, type=int, help='')
    parser.add_argument('--FOLD', default=10, type=int, help="k fold")
    parser.add_argument('--epochs', default=40, type=int, help="")
    parser.add_argument('--early_stop', default=20, type=int, help="")
    parser.add_argument('--lr', default=1e-3, type=float, help="")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = arg_setting()  # 设置基本参数
    basic_setting(args.SEED, DEVICE)  # 设置随机种子
    test_data, train_dataset, test_dataset, w2v_model, fasttext_model, id2label, ylabel = data_process()
    #开始训练
    model_num = run(args, train_dataset, w2v_model, fasttext_model, ylabel)
    #获得提交结果文件
    get_submit(args, test_data, test_dataset, id2label, model_num)
