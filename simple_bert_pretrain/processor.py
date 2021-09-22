import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers.optimization import get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, roc_auc_score
import os
import json
import numpy as np
import tqdm
import random
import transformers
from model.model import Model
import time
import pickle
import sys
import copy

class Processor(object):
    def __init__(self, data_loader, config):
        self.data_loader = data_loader
        self.config = config
    
    def bce_loss(self, outputs, labels, finally_output):
        labels = torch.tensor(labels, dtype=torch.long).to(self.config.device)
        # loss_fn = nn.CrossEntropyLoss().to(self.config.device)
        # # loss = loss_fn(finally_output, labels)
        loss = F.cross_entropy(finally_output, labels)
        # loss = F.binary_cross_entropy_with_logits(outputs, labels, labels>=0)
        return loss
    
    def ce_loss(self, outputs, labels):
        labels = torch.tensor(labels, dtype=torch.long).to(self.config.device)
        loss = F.cross_entropy(outputs.transpose(1, 2), labels, ignore_index=0)
        return loss
    
    def train_one_step(self, batch, pretrain):
        cls_outputs, mask_outputs, final_out_cls, final_output_pooler = self.model(batch) #将batch数据放入模型, 获得cls:[128], mask输出:[128, 23, 25000]  25000是所有token
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!需要解决
        cls_loss = self.bce_loss(cls_outputs, batch['cls_labels'], final_out_cls) #cls的预测结果,除了真正的标签，还有我们自己设置的加标签
        mask_loss = self.ce_loss(mask_outputs, batch['mask_labels']) #mask的预测结果
        if pretrain:
            loss = mask_loss
        else:
            # loss = cls_loss + self.config.mask_w*mask_loss
            loss = cls_loss
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
        return loss.item(), cls_loss.item(), mask_loss.item()
    
    def eval_one_step(self, batch):
        with torch.no_grad():
            cls_outputs, mask_outputs, final_out_cls, final_output_pooler = self.model(batch)
            loss = self.bce_loss(cls_outputs, batch['cls_labels'], final_out_cls).item()#使用真正的标签进行loss传播
            outputs = torch.sigmoid(cls_outputs).detach().cpu().numpy() #output只取cls来做的预测
        return outputs, loss, final_output_pooler
    #######！！！！！！！！！！！！！！！！！！！！！！
    def evaluate(self, data, flag):
        self.model.eval()
        trues, preds = [], []
        eval_loss = 0
        eval_tqdm = tqdm.tqdm(data, total=len(data))
        eval_tqdm.set_description('eval_loss: {:.4f}'.format(0))
        for batch in eval_tqdm:
            outputs, loss, finally_output = self.eval_one_step(batch)
            pre_for_f1 = torch.max(finally_output, 1)[1].data.cpu().numpy()
            trues.extend(batch['cls_labels'])
            preds.extend(pre_for_f1)

            # for j in range(len(outputs)):
            #     true = batch['cls_labels'][j]#一个一个数据的装？
            #     pred = outputs[j]
            #     trues.append(true)
            #     preds.append(pred)
            eval_loss += loss
            eval_tqdm.set_description('eval_loss: {:.4f}'.format(loss))
        eval_loss /= len(data)

        f1 = f1_score(trues, preds, average='micro')
        # self.model.train()#????这里是做什么训练？？？这下面没有训练，只是在做指标测评
        if trues:
            pairs = list(zip(trues, preds))
            pairs.sort(key=lambda x: x[1])
            rank_sum, pos_num, neg_num = 0, 0, 0
            for i, pair in enumerate(pairs):
                if pair[0] == 1:
                    pos_num += 1
                    rank_sum += i
                else:
                    neg_num += 1
            auc = (rank_sum-pos_num*(pos_num+1)//2)/(pos_num*neg_num)
            # trues, preds = np.array(trues), np.array(preds)>0.5

            # f1 = f1_score(trues, preds, average='micro')
            print('Average {} loss: {:.4f}, auc: {:.4f}, f1: {:.4f}.'.format(flag, eval_loss, auc, f1))
        else:
            auc = f1 = None
        score = {'auc': auc, 'f1': f1}
        return eval_loss, score
    
    def init(self):
        self.model = Model(self.config) #encoder+linear（cls）+linear(mask)
        no_decay = ['bias', 'LayerNorm.weight']
        self.optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]#打印参数量
        print('model parameters number: {}.'.format(sum(p.numel() for p in self.model.parameters() if p.requires_grad)))
        self.model.to(self.config.device)
    
    def train(self):
        print('Train starts:')
        #判断！如果微调文件已经存在，那么就直接结束第二阶段
        if os.path.exists(self.config.store_path()): #'./result/model_states/train_testA_0.0003_5e-05_32_1.5_0.pth'
            print('Train done.')
            return
        with open(self.config.store_path(), 'w') as f:#防止没有训练完前
            f.write('!')
        train, valid = self.data_loader.get_train()#获得训练数据，验证数据
        train_iter = iter(train)
        print('Train batch size {}, eval batch size {}.'.format(self.config.batch_size(True), self.config.batch_size(False)))
        print('Batch number: train {}, valid {}.'.format(len(train), len(valid)))
        if not os.path.exists(self.config.pretrain_path()):
            print('Stage 1:')
            self.init()#初始化--> 定义模型，参数。以及将新的文本词加入到模型的token中
            self.optimizer = optim.AdamW(self.optimizer_grouped_parameters, lr=self.config.learning_rate(1))
            self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=self.config.warmup_steps0, num_training_steps=self.config.training_steps0)
            print('total epochs: {}, warmup steps: {}, training steps: {}'.format(self.config.training_steps0/len(train), self.config.warmup_steps0, self.config.training_steps0))
            for i, p in enumerate(self.model.encoder.bert.parameters()):
                if i > 0:
                    p.requires_grad = False#embedding层不梯度更新
            min_train_loss, epoch, global_steps = 1e16, 0, 0
            try:
                while global_steps < self.config.training_steps0:
                    epoch += 1
                    train_mask_loss = 0.0
                    train_tqdm = tqdm.tqdm(range(len(train)))
                    train_tqdm.set_description('Epoch {} | train_mask_loss: {:.4f}'.format(epoch, 0))#设置动态变化的loss
                    for steps in train_tqdm:  #遍历所有的batch
                        batch = next(train_iter) #每次出来一个batch的数据
                        loss, cls_loss, mask_loss = self.train_one_step(batch, True) #训练返回综合loss， 真实标签的loss，mask标签的loss
                        train_mask_loss += mask_loss
                        train_tqdm.set_description('Epoch {} | train_mask_loss: {:.4f}'.format(epoch, mask_loss))
                    steps += 1
                    global_steps += steps
                    train_mask_loss /= steps
                    print('Average train_mask_loss: {:.4f}.'.format(train_mask_loss))
                    if train_mask_loss < min_train_loss:
                        min_train_loss = train_mask_loss
                        word_embeddings = copy.deepcopy(self.model.encoder.bert.embeddings.word_embeddings.state_dict())
                        mask_fc = copy.deepcopy(self.model.mask_fc.state_dict())
            except KeyboardInterrupt:
                train_tqdm.close()
                print('Exiting from training early.')
                os.remove(self.config.store_path())
                return
            #存放训练好的embedding
            with open(self.config.pretrain_path(), 'wb') as f:
                torch.save([word_embeddings, mask_fc], f)
            for i, p in enumerate(self.model.encoder.bert.parameters()):
                if i > 0:
                    p.requires_grad = True #设置用于第二阶段的梯度是否回传
        
        
        print('Stage 2:')
        #阶段2，进行下游任务训练微调预测
        self.init()
        with open(self.config.pretrain_path(), 'rb') as f:#打开训练好的模型参数及全连接层参数
            [word_embeddings, mask_fc] = torch.load(f)   #mask_fc是全连接层的权重
        self.model.encoder.bert.embeddings.word_embeddings.load_state_dict(word_embeddings)##加载模型参数
        #self.model.mask_fc.load_state_dict(mask_fc)
        max_valid_auc, epoch, global_steps = 0.0, 0, 0
        best_scores = {}
        self.optimizer = optim.AdamW(self.optimizer_grouped_parameters, lr=self.config.learning_rate(2))
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=self.config.warmup_steps, num_training_steps=self.config.training_steps)
        print('total epochs: {}, warmup steps: {}, training steps: {}'.format(self.config.training_steps/len(train), self.config.warmup_steps, self.config.training_steps))
        try:
            while global_steps < self.config.training_steps:
                epoch += 1
                train_loss, train_cls_loss, train_mask_loss = 0.0, 0.0, 0.0
                train_tqdm = tqdm.tqdm(range(len(train)))
                train_tqdm.set_description('Epoch {} | train_loss: {:.4f}'.format(epoch, 0))
                for steps in train_tqdm:
                    batch = next(train_iter)
                    loss, cls_loss, mask_loss = self.train_one_step(batch, False)
                    train_loss += loss
                    train_cls_loss += cls_loss
                    train_mask_loss += mask_loss
                    train_tqdm.set_description('Epoch {} | train_loss: {:.4f}'.format(epoch, loss))
                steps += 1
                global_steps += steps
                print('Average train_loss: {:.4f}, train_cls_loss: {:.4f}, train_mask_loss: {:.4f}.'.format(train_loss/steps, train_cls_loss/steps, train_mask_loss/steps))
                valid_loss, scores = self.evaluate(valid, 'valid')#完整训练完一个epoch后进行验证评估
                if scores['auc'] > max_valid_auc:#以auc值为指标
                    max_valid_auc = scores['auc']
                    best_scores = copy.deepcopy(scores)
                    best_para = copy.deepcopy(self.model.state_dict())
        except KeyboardInterrupt:
            train_tqdm.close()
            print('Exiting from training early.')
            os.remove(self.config.store_path())
            return
        #####训练完成，记录下
        print('Train finished, max valid auc {:.4f}, stop at epoch {}.'.format(max_valid_auc, epoch))
        with open(self.config.store_path(), 'wb') as f:
            torch.save(best_para, f)#保存最好的测试模型
        result_path = self.config.result_path()

        with open(result_path, 'a', encoding='utf-8') as f:
            obj = self.config.parameter_info()
            obj.update(best_scores)
            f.write(json.dumps(obj)+'\n')
    
    def extract_feature(self):#第三阶段？
        print('Extract feature:')

        self.model = Model(self.config)
        self.model.to(self.config.device)
        #打印特征量
        print('model parameters number: {}.'.format(sum(p.numel() for p in self.model.parameters() if p.requires_grad)))
        file = self.config.store_path()
        if not os.path.exists(file):
            return
        with open(file, 'rb') as f:
            best_para = torch.load(f)
        #加载前面第二阶段微调训练好的模型
        self.model.load_state_dict(best_para)
        self.model.eval()
        data = self.data_loader.get_all()#含inputs, segs, mask_labels, cls_labels,即得到模型的输入
        extract_tqdm = tqdm.tqdm(data, total=len(data))
        features = []
        for batch in extract_tqdm:
            outputs, loss, outputs_for_predict = self.eval_one_step(batch)
            features.append(self.model.cls_h.cpu().numpy())#取cls的隐藏状态向量
        features = np.concatenate(features, 0) #(2000, 768)
        np.save(self.config.feature_path(), features)
    ############预测生成提交文件部分
    def predict(self, id2label, data_test):
        print('Predict starts:')
        self.model = Model(self.config)
        self.model.to(self.config.device)
        print('model parameters number: {}.'.format(sum(p.numel() for p in self.model.parameters() if p.requires_grad)))
        predicts = []
        for seed in range(100):
            file = self.config.store_path(seed=seed)
            if not os.path.exists(file):
                continue
            print('Ensemble id:', seed)
            with open(file, 'rb') as f:
                best_para = torch.load(f)
            self.model.load_state_dict(best_para)#加载模型
            self.model.eval()
            data = self.data_loader.get_predict()
            predict_tqdm = tqdm.tqdm(data, total=len(data))
            predict = []
            predict_for_submit = []
            for batch in predict_tqdm:
                outputs, loss, outputs_for_predict = self.eval_one_step(batch)
                for j in range(len(outputs)):
                    predict.append(outputs[j])
                #转化为标签来存储
                out = torch.max(outputs_for_predict, 1)[1]
                predict_for_submit.extend(out.data.cpu().numpy())
            predicts.append(predict)
        predicts = np.mean(np.array(predicts), 0).tolist()
        
        predict_submit = [id2label[label] for label in predict_for_submit]
        data_test['label'] = predict_submit
        data_test.to_csv(self.config.prediction_path(), index=False)
        # with open(self.config.prediction_path(), 'w') as f:
        #     f.write('\n'.join([str(v) for v in data_test]))