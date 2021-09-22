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
from itertools import repeat
from model.rcnn import TextRCNN, TextRCNNConfig
from run_config import RunConfig

# from utils.spatial_dropout import SpatialDropout
import warnings
import torch.nn as nn
from tqdm import tqdm
import random
import gensim
import argparse
from torchcontrib.optim import SWA
import os
import logging
from apex import amp
from torch.utils import data
from sklearn.metrics import f1_score
from torch import nn
import torch.nn.functional as F
from torch.optim import *
import pickle
torch.set_printoptions(edgeitems=768)
warnings.filterwarnings("ignore")
np.set_printoptions(threshold=np.inf)
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

SEED = 2021

random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if DEVICE != "cpu":
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def data_process():
    train_data = pd.read_csv("./datasets/datagrand_2021_train.csv")
    test_data = pd.read_csv("./datasets/datagrand_2021_test.csv")
    
    with open('./model/id2label.pkl', 'rb') as f:
        id2label = pickle.load(f)
    with open('./model/label2id.pkl', 'rb') as f:
        label2id = pickle.load(f)
    
    y_train = np.zeros((len(train_data), len(id2label)), dtype=np.int8)

    all_sentences = pd.concat([train_data["text"], test_data["text"]]).reset_index(
        drop=True
    )
    all_sentences.drop_duplicates().reset_index(drop=True, inplace=True)
    all_sentences = all_sentences.apply(lambda x: x.split(" ")).tolist()
    if not os.path.exists("./embedding/w2v.model"):
        w2v_model = gensim.models.word2vec.Word2Vec(
            all_sentences,
            sg=1,
            vector_size=300,
            window=7,
            min_count=1,
            negative=3,
            sample=0.001,
            hs=1,
            seed=452,
        )
        w2v_model.save("./embedding/w2v.model")
    else:
        w2v_model = gensim.models.word2vec.Word2Vec.load("./embedding/w2v.model")

    if not os.path.exists("./embedding/fasttext.model"):
        fasttext_model = gensim.models.FastText(
            all_sentences, seed=452, vector_size=100, min_count=1, epochs=20, window=2
        )
        fasttext_model.save("./embedding/fasttext.model")
    else:
        fasttext_model = gensim.models.word2vec.Word2Vec.load(
            "./embedding/fasttext.model"
        )
    train_dataset = []
    ###################################
    # 增加一个List ylabel 存储训练集的类别
    ###################################
    ylabel = []
    for i in tqdm(range(len(train_data))):
        train_dict = {}
        train_dict["id"] = train_data.loc[i, "id"]
        train_dict["text"] = train_data.loc[i, "text"]
        y_train[i][label2id[train_data.loc[i, "label"]]] = 1
        train_dict["label"] = y_train[i]
        ###################################
        # 增加一个List ylabel 存储训练集的类别
        ###################################
        ylabel.append(train_data.loc[i, "label"])
        train_dataset.append(train_dict)
    test_dataset = []
    for i in tqdm(range(len(test_data))):
        test_dict = {}
        test_dict["id"] = test_data.loc[i, "id"]
        test_dict["text"] = test_data.loc[i, "text"]
        test_dict["label"] = -1
        test_dataset.append(test_dict)
    return (
        test_data,
        train_dataset,
        test_dataset,
        w2v_model,
        fasttext_model,
        id2label,
        ylabel,
    )


class DataSet(data.Dataset):
    def __init__(self, args, data, mode="train"):
        self.data = data
        self.mode = mode
        self.w2v_model = gensim.models.word2vec.Word2Vec.load("./embedding/w2v.model")
        self.dataset = self.get_data(self.data, self.mode)

    def get_data(self, data, mode):
        dataset = []
        global s
        for data_li in tqdm(data):
            text = data_li["text"].split(" ")
            text = [
                self.w2v_model.wv.key_to_index[s] + 1 if s in self.w2v_model.wv else 0
                for s in text
            ]
            if len(text) < args.MAX_LEN:
                text += [0] * (args.MAX_LEN - len(text))
            else:
                text = text[: args.MAX_LEN]
            label = data_li["label"]
            dataset_dict = {"text": text, "label": label}
            dataset.append(dataset_dict)
        return dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        text = torch.tensor(data["text"])
        if self.mode == "test":
            return text
        else:
            label = torch.tensor(data["label"])
            return text, label


def get_dataloader(args, dataset, mode):
    torchdata = DataSet(args, dataset, mode=mode)
    if mode == "train":
        dataloader = torch.utils.data.DataLoader(
            torchdata,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            drop_last=True,
            pin_memory=True,
        )
    elif mode == "test":
        dataloader = torch.utils.data.DataLoader(
            torchdata,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            drop_last=False,
            pin_memory=True,
        )
    elif mode == "valid":
        dataloader = torch.utils.data.DataLoader(
            torchdata,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            drop_last=True,
            pin_memory=True,
        )
    return dataloader, torchdata


class CyclicLR(object):
    def __init__(
        self,
        optimizer,
        base_lr=1e-3,
        max_lr=6e-3,
        step_size=2000,
        mode="triangular",
        gamma=1.0,
        scale_fn=None,
        scale_mode="cycle",
        last_batch_iteration=-1,
    ):

        if not isinstance(optimizer, Optimizer):
            raise TypeError("{} is not an Optimizer".format(type(optimizer).__name__))
        self.optimizer = optimizer

        if isinstance(base_lr, list) or isinstance(base_lr, tuple):
            if len(base_lr) != len(optimizer.param_groups):
                raise ValueError(
                    "expected {} base_lr, got {}".format(
                        len(optimizer.param_groups), len(base_lr)
                    )
                )
            self.base_lrs = list(base_lr)
        else:
            self.base_lrs = [base_lr] * len(optimizer.param_groups)

        if isinstance(max_lr, list) or isinstance(max_lr, tuple):
            if len(max_lr) != len(optimizer.param_groups):
                raise ValueError(
                    "expected {} max_lr, got {}".format(
                        len(optimizer.param_groups), len(max_lr)
                    )
                )
            self.max_lrs = list(max_lr)
        else:
            self.max_lrs = [max_lr] * len(optimizer.param_groups)

        self.step_size = step_size

        if mode not in ["triangular", "triangular2", "exp_range"] and scale_fn is None:
            raise ValueError("mode is invalid and scale_fn is None")

        self.mode = mode
        self.gamma = gamma

        if scale_fn is None:
            if self.mode == "triangular":
                self.scale_fn = self._triangular_scale_fn
                self.scale_mode = "cycle"
            elif self.mode == "triangular2":
                self.scale_fn = self._triangular2_scale_fn
                self.scale_mode = "cycle"
            elif self.mode == "exp_range":
                self.scale_fn = self._exp_range_scale_fn
                self.scale_mode = "iterations"
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
            param_group["lr"] = lr

    def _triangular_scale_fn(self, x):
        return 1.0

    def _triangular2_scale_fn(self, x):
        return 1 / (2.0 ** (x - 1))

    def _exp_range_scale_fn(self, x):
        return self.gamma ** (x)

    def get_lr(self):
        step_size = float(self.step_size)
        cycle = np.floor(1 + self.last_batch_iteration / (2 * step_size))
        x = np.abs(self.last_batch_iteration / step_size - 2 * cycle + 1)

        lrs = []
        param_lrs = zip(self.optimizer.param_groups, self.base_lrs, self.max_lrs)
        for param_group, base_lr, max_lr in param_lrs:
            base_height = (max_lr - base_lr) * np.maximum(0, (1 - x))
            if self.scale_mode == "cycle":
                lr = base_lr + base_height * self.scale_fn(cycle)
            else:
                lr = base_lr + base_height * self.scale_fn(self.last_batch_iteration)
            lrs.append(lr)
        return lrs


# 定义模型

loss_fun = nn.BCEWithLogitsLoss()


def cal_macro_f1(y_true, y_pred):
    score = f1_score(y_true, y_pred, average="macro")
    return score


def validation_funtion(model, valid_dataloader, valid_torchdata, mode="valid"):
    model.eval()
    pred_list = []
    labels_list = []
    val_loss = []
    y_preds = []
    y_trues = []
    with torch.no_grad():
        if mode == "valid":
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


def train(
    args,
    model,
    train_dataloader,
    valid_dataloader,
    valid_torchdata,
    epochs,
    model_num,
    early_stop=None,
):
    #     ema = EMA(model, 0.999)
    #     ema.register()
    param_optimizer = list(model.named_parameters())
    embed_pa = ["embed.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in embed_pa)
            ]
        },
        {"params": model.embed.parameters(), "lr": 2e-5},
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters, lr=args.lr, amsgrad=True, weight_decay=5e-4
    )
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=3, T_mult=2, eta_min=1e-5, last_epoch=-1
    )
    #     scheduler = CyclicLR(optimizer, base_lr=1e-3, max_lr=3e-3,
    #                step_size=30, mode='exp_range',
    #                gamma=0.99994)
    #     opt = SWA(optimizer, swa_start=100, swa_freq=5, swa_lr=1e-4)

    best_val_loss = np.inf
    best_auc = np.inf
    best_loss = np.inf
    no_improve = 0
    for epoch in range(epochs):
        model.train()
        train_loss = []
        if epoch > 2:
            for param in model.named_parameters():
                if param[0] == "embed.weight":
                    param[1].requires_grad = True
                    break
        #         fgm = FGM(model)
        bar = tqdm(train_dataloader)
        for i, (description, label) in enumerate(bar):
            optimizer.zero_grad()
            output = model(description.to(DEVICE), label.to(DEVICE))
            loss = output
            #             loss.backward()
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            train_loss.append(loss.item())

            #             fgm.attack()
            #             loss_adv = model(describe.to(DEVICE), label.to(DEVICE))
            #             loss_ad = loss_adv
            #             loss_ad.backward()
            #             fgm.restore()

            scheduler.step(epochs + i / len(train_dataloader))
            #             scheduler.batch_step()
            optimizer.step()
        #             ema.update()
        # bar.set_postfix(tloss=np.array(train_loss).mean())
        #         opt.swap_swa_sgd()
        #         ema.apply_shadow()
        auc, val_loss = validation_funtion(
            model, valid_dataloader, valid_torchdata, "valid"
        )
        #         ema.restore()
        print(
            "Epoch:[{}/{}] train_loss: {:.5f}, val_loss: {:.5f},f1-score: {:.5f}\n".format(
                epoch, epochs, np.mean(train_loss), val_loss, auc
            )
        )
        if early_stop:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_auc = auc
                best_loss = np.mean(train_loss)
                #                 ema.apply_shadow()
                torch.save(
                    model.state_dict(),
                    "./model/saved/{}_model_{}.bin".format(args.NAME, model_num),
                )
                #                 ema.restore()
                model_num += 1
            else:
                no_improve += 1
            if no_improve == early_stop:
                model_num += 1
                break
            if epoch == epochs - 1:
                model_num += 1
        else:
            if epoch >= epochs - 1:
                torch.save(
                    model.state_dict(),
                    "./model/saved/{}_model_{}.bin".format(args.NAME, model_num),
                )
                model_num += 1
    return best_val_loss, best_auc, best_loss, model_num


###################################
# 增加了一个参数 ylabel
###################################


def run(args, train_dataset, w2v_model, fasttext_model, ylabel):
    kf = StratifiedKFold(n_splits=args.FOLD, shuffle=True, random_state=SEED)
    best_mlogloss = []
    best_auc = []
    best_loss = []
    model_num = 1
    ###################################
    # 增加了一个参数 ylabel,用在此处
    ###################################
    for i, (train_index, test_index) in enumerate(
        kf.split(np.arange(len(train_dataset)), ylabel)
    ):
        print(str(i + 1), "-" * 50)
        tra = [train_dataset[index] for index in train_index]
        val = [train_dataset[index] for index in test_index]
        print(len(tra))
        print(len(val))
        train_dataloader, train_torchdata = get_dataloader(args, tra, mode="train")
        valid_dataloader, valid_torchdata = get_dataloader(args, val, mode="valid")
        run_config = RunConfig()#加载训练参数，如epoch, batch_size
        model_config = TextRCNNConfig()
        model = TextRCNN(run_config, model_config)
        model.to(DEVICE)
        mlogloss, auc, loss, model_n = train(
            args,
            model,
            train_dataloader,
            valid_dataloader,
            valid_torchdata,
            args.epochs,
            model_num,
            early_stop=args.early_step,
        )
        torch.cuda.empty_cache()
        best_mlogloss.append(mlogloss)
        best_auc.append(auc)
        best_loss.append(loss)
    for i in range(args.FOLD):
        print(
            "- 第{}折中，best valloss: {}   best f1-socre: {}   best trainloss: {}".format(
                i + 1, best_mlogloss[i], best_auc[i], best_loss[i]
            )
        )
    return model_n


def get_submit(args, test_data, test_dataset, id2label, model_num):
    model = TextRCNN(
        args,
        w2v_model.wv.vectors.shape[0] + 1,
        w2v_model.wv.vectors.shape[1] + fasttext_model.wv.vectors.shape[1],
        embeddings=True,
    )
    model.to(DEVICE)
    test_preds_total = []
    test_dataloader, test_torchdata = get_dataloader(args, test_dataset, mode="test")
    for i in range(1, model_num):
        model.load_state_dict(
            torch.load("./saved/{}_model_{}.bin".format(args.NAME, i))
        )
        test_pred_results = validation_funtion(
            model, test_dataloader, test_torchdata, "test"
        )
        test_preds_total.append(test_pred_results)
    test_preds_merge = np.sum(test_preds_total, axis=0) / (model_num - 1)
    test_pre_tensor = torch.tensor(test_preds_merge)
    test_pre = torch.max(test_pre_tensor, 1)[1]

    pred_labels = [id2label[i] for i in test_pre]
    # submit_file = '/home/zyf/Summer game2021/Datafountain/submits/submit.csv'
    submit_file = "./submit/submit_{}.csv".format(args.NAME)

    pd.DataFrame({"id": test_data["id"], "label": pred_labels}).to_csv(
        submit_file, index=False
    )


def arg_setting():
    parser = argparse.ArgumentParser()
    parser.add_argument("--NAME", default="rcnn", type=str, help="")
    parser.add_argument(
        "--MAX_LEN", default=100, type=int, help="max length of sentence"
    )
    parser.add_argument("--batch_size", default=32, type=int, help="")
    parser.add_argument("--FOLD", default=10, type=int, help="k fold")
    parser.add_argument("--epochs", default=10, type=int, help="")
    parser.add_argument("--early_step", default=5, type=int, help="")
    parser.add_argument("--lr", default=1e-3, type=float, help="")
    args = parser.parse_args(args=[])
    return args


if __name__ == "__main__":
    args = arg_setting()  # 设置基本参数
    ###################################
    # 增加一个返回值 ylabel
    ###################################
    (
        test_data,
        train_dataset,
        test_dataset,
        w2v_model,
        fasttext_model,
        id2label,
        ylabel,
    ) = data_process()
    # 开始训练
    ###################################
    # 增加一个参数ylabel
    ###################################
    model_num = run(args, train_dataset, w2v_model, fasttext_model, ylabel)
    # 获得提交结果文件
    # get_submit(args, test_data, test_dataset, id2label, model_num)
