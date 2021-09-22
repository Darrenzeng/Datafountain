import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import BertModel, BertConfig

class BERTEncoder(nn.Module):        
    def __init__(self, config):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        self.bert.resize_token_embeddings(104) #向transformers bert中新增字符的方法，这里是新增[PAD], [unused1]...[unused99], [UNK], [CLS], [SEP], [MASK]
        self.bert.resize_token_embeddings(config.vocab_num) #新增文本中25000个token
    
    def forward(self, batch):
        x, y = batch['inputs'], batch['segs']
        outputs = self.bert(x, attention_mask=(x>0), token_type_ids=y) #last_hidden_state[4, 23, 768], pooler_output[4, 768]
        h = outputs.last_hidden_state
        pooler = outputs.pooler_output #[32, 768]
        return h, pooler