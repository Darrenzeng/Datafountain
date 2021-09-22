import torch
import torch.nn as nn
import gensim
import torch.nn.functional as F
import numpy as np

class HanConfig(object):
    def __init__(self):
        self.dropout = 0.5
        self.learning_rate = 5e-4
        self.num_filters = 200

        self.num_classes = 35
        self.hidden_size_gru = 256
        self.hidden_size_att = 512
        self.hidden_size = 128
        self.embedding_dim = 400 #w2v和fasttext拼接后的维度

        # 训练过程中是否冻结对词向量的更新
        self.freeze = True

class HAN(nn.Module):
    def __init__(self, run_config, model_config): #model_config可以等价于args
#     def __init__(self,args, vocab_size, embedding_dim, embeddings=None):
        super(HAN, self).__init__()
        self.num_classes = model_config.num_classes
        hidden_size_gru = model_config.hidden_size_gru
        hidden_size_att = model_config.hidden_size_att
        hidden_size = model_config.hidden_size
        self.seq_len = run_config.seq_len
        
        
        #加载训练好的两个模型
        w2v_model = gensim.models.word2vec.Word2Vec.load("./embedding/w2v.model").wv
        vocab_size = w2v_model.vectors.shape[0]+1  #3457

        self.embed = nn.Embedding(vocab_size, model_config.embedding_dim) #词汇长度是w2v训练后的长度, 维度是w2v和fasttext拼接后的维度
        #使用单一模型来进行预测
        # weight_vector = PretrainVector().load_pretrained_vec(vec_type='concat')

        fasttext_model = gensim.models.word2vec.Word2Vec.load("./embedding/fasttext.model").wv
        #获得两个模型的向量
        w2v_embed_matrix = w2v_model.vectors #(3456, 300)
        fasttext_embed_matrix = fasttext_model.vectors #(3456, 100)
#       embed_matrix = w2v_embed_matrix
        #向量拼接
        embed_matrix = np.concatenate([w2v_embed_matrix, fasttext_embed_matrix], axis=1) #(3456, 400)

        oov_embed = np.zeros((1, embed_matrix.shape[1]))
        embed_matrix = torch.from_numpy(np.vstack((oov_embed, embed_matrix))) #加入unk， [3457, 400]

        self.embed.weight.data.copy_(embed_matrix)
        self.embed.weight.requires_grad = False
        
        
        self.gru1 = nn.GRU(model_config.embedding_dim, hidden_size_gru,
                           bidirectional=True, batch_first=True)
        self.att1 = SelfAttention(hidden_size_gru * 2, hidden_size_att)
        self.gru2 = nn.GRU(hidden_size_att, hidden_size_gru,
                           bidirectional=True, batch_first=True)
        self.att2 = SelfAttention(hidden_size_gru * 2, hidden_size_att)
        self.tdfc = nn.Linear(model_config.embedding_dim, model_config.embedding_dim)
        # self.tdfc = nn.Linear(400, 400)
        self.tdbn = nn.BatchNorm2d(1)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size_att, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size,self.num_classes)
        )
        self.dropout = nn.Dropout(0.4)

    def forward(self, x, label=None):
        # [32, 100]
        x = x.view(x.size(0) * self.seq_len, -1).contiguous() #[3200, 1] #对
        x = self.dropout(self.embed(x)) #[3200, 1, 400]
        x = self.tdfc(x)
        x = x.unsqueeze(1) #[3200, 1, 1, 400]

        x = self.tdbn(x).squeeze(1) #[3200, 1, 400]
        x, _ = self.gru1(x) #[3200, 1, 512]
        x = self.att1(x)  #[3200, 512]
        x = x.view(x.size(0) // self.seq_len,
                   self.seq_len, -1).contiguous()  #[32, 100, 512]
        x, _ = self.gru2(x)
        x = self.att2(x)
        out = self.dropout(self.fc(x))
        if label is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(out.view(-1, self.num_classes).float(),
                            label.view(-1, self.num_classes).float())
            return loss
        else:
            return out


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