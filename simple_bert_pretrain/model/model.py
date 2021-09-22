import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.bert_encoder import BERTEncoder

class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        #普通的bertencode
        self.encoder = BERTEncoder(config)
        self.cls_fc = nn.Linear(config.hidden_dim, 1) #针对cls
        self.mask_fc = nn.Linear(config.hidden_dim, config.vocab_num) #[768, 25000]
        self.linear = nn.Linear(config.hidden_dim, config.num_class)
    
    def forward(self, batch):
        
        h, pooler = self.encoder(batch)  # [batch_size, seq_len, hiden_dim],    [batch_size, hiden_dim]
        self.cls_h = h[:, 0, :] #[4, 23, 768]  --> [4, 1, 768]只取cls作为输出
        cls_outputs = self.cls_fc(self.cls_h).squeeze(-1) #[4, 768] --> [4]
        mask_outputs = self.mask_fc(h) #[4, 23, 768] --> [4, 23, 25000]
        final_out_cls = self.linear(self.cls_h.squeeze()) #[4, 768]-->[4, 35]#取cls来做
        final_output_pooler = self.linear(pooler)  #[4, 768]-->[4, 35]取句子级别来做
        return cls_outputs, mask_outputs, final_out_cls, final_output_pooler