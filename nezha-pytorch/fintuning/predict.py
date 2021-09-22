from NEZHA.modeling_nezha import *
from tqdm import tqdm, trange
import numpy as np
import pandas as pd
import logging
import torch
import random
import os
from torch import nn, optim
from transformers import BertTokenizer, AdamW, BertModel, BertPreTrainedModel, BertConfig, \
    get_linear_schedule_with_warmup, XLNetModel, XLNetTokenizer, XLNetConfig
from transformers.optimization import get_linear_schedule_with_warmup
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import mean_absolute_error, accuracy_score, f1_score, roc_auc_score
from model import *
from utils import *
import time


MODEL_CLASSES = {
    'BertForClass': BertForClass,
    'BertLastCls': BertLastCls,
    'BertLastTwoCls': BertLastTwoCls,
    'BertLastTwoClsPooler': BertLastTwoClsPooler,
    'BertLastTwoEmbeddings': BertLastTwoEmbeddings,
    'BertLastTwoEmbeddingsPooler': BertLastTwoEmbeddingsPooler,
    'BertLastFourCls': BertLastFourCls,
    'BertLastFourClsPooler': BertLastFourClsPooler,
    'BertLastFourEmbeddings': BertLastFourEmbeddings,
    'BertLastFourEmbeddingsPooler': BertLastFourEmbeddingsPooler,
    'BertDynCls': BertDynCls,
    'BertDynEmbeddings': BertDynEmbeddings,
    'BertRNN': BertRNN,
    'BertCNN': BertCNN,
    'BertRCNN': BertRCNN,
    'XLNet': XLNet,
    'Electra': Electra,
    'NEZHA': NEZHA,

}


class Config:
    def __init__(self):
        # 预训练模型路径
        self.modelId = 2
        self.model = "NEZHA"
        self.Stratification = False
        self.model_path = '/home/zyf/Summer_game2021/Datafountain/liu_nezha-pytorch/pretrain/nezha_output/add4_token_mlm5_920ep/'

        self.num_class = 35
        self.dropout = 0.2
        self.MAX_LEN = 100
        self.epoch = 3
        self.learn_rate = 4e-5
        self.normal_lr = 1e-4
        self.batch_size = 32
        self.k_fold = 5
        self.seed = 42

        self.device = torch.device('cuda:1')
        # self.device = torch.device('cpu')

        self.focalloss = False
        self.pgd = False
        self.fgm = False


def submit(pred, test_df, id2label):
    test_preds_merge = np.sum(pred, axis=0) / (pred.shape[0])
    test_pre_tensor = torch.tensor(test_preds_merge)
    test_pre = torch.max(test_pre_tensor, 1)[1]
    pred_labels = [id2label[i] for i in test_pre]
    SUBMISSION_DIR = "submit"
    if not os.path.exists(SUBMISSION_DIR):
        os.makedirs(SUBMISSION_DIR)
    Name = "Nezha_add4_mlm5_5kfold_most"
    submit_file = SUBMISSION_DIR+"/submit_{}.csv".format(Name)

    pd.DataFrame({"id": test_df['id'], "label": pred_labels}).to_csv(
        submit_file, index=False)



config = Config()
os.environ['PYTHONHASHSEED'] = '0'  # 消除hash算法的随机性
random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)

train_clean = '/home/zyf/Summer_game2021/Datafountain/datasets/datagrand_2021_train.csv'
test_clean = '/home/zyf/Summer_game2021/Datafountain/datasets/test_clean.csv'
train = pd.read_csv(train_clean)
test = pd.read_csv(test_clean)
id2label = list(train['label'].unique())
label2id = {id2label[i]: i for i in range(len(id2label))}
test_dataset = []
for i in tqdm(range(len(test))):
    test_dict = {}
    test_dict['text'] = test.loc[i, 'text']
    test_dict['label'] = [-1]*35
    test_dataset.append(test_dict)
test_D = data_generator(test_dataset, config)
model_pre = []
for fold in tqdm(range(config.k_fold)):
    PATH = './models_add4_mlm0.5/bert_{}.pth'.format(fold)
    # model = MODEL_CLASSES[config.model](config).to(config.device)
#     model.load_state_dict(torch.load(PATH)
    model =  torch.load(PATH)
    model = model.to(config.device)
    model.eval()
    with torch.no_grad():
        y_p = []
        y_l = []
        val_y = []
        train_logit = None
        for input_ids, input_masks, segment_ids, labels in tqdm(test_D, disable=True):

            y_pred = model(input_ids, input_masks, segment_ids)
            y_pred = F.softmax(y_pred)
            y_pred = y_pred.detach().to("cpu").numpy()
            if train_logit is None:
                train_logit = y_pred
            else:
                train_logit = np.vstack((train_logit, y_pred))
        model_pre.append(train_logit)
        

submit(np.array(model_pre), test, id2label)

