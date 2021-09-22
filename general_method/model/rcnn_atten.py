
import torch
import torch.nn as nn
import gensim
import torch.nn.functional as F
import numpy as np
from itertools import repeat

class TextRCNNAttnConfig(object):
    def __init__(self):
        self.dropout = 0.4
        self.learning_rate = 5e-4
        self.num_filters = 200

        self.hidden_size1 = 128
        self.hidden_size2 = 64
        self.num_layers = 2
        self.num_classes = 35
        self.hidden_size = 128
        self.embedding_dim = 400 #w2v和fasttext拼接后的维度

        # 训练过程中是否冻结对词向量的更新
        self.freeze = True

class TextRCNNAttn(nn.Module):
    def __init__(self, run_config, model_config):
        super(TextRCNNAttn, self).__init__()
        self.hidden_size1 = model_config.hidden_size1
        self.hidden_size2 = model_config.hidden_size2
        self.num_layers = model_config.num_layers
        self.dropout = model_config.dropout
        self.learning_rate = model_config.learning_rate
        self.freeze = True  # 训练过程中是否冻结对词向量的更新
        self.seq_len = run_config.seq_len
        self.num_classes=model_config.num_classes
        self.embedding_dim = model_config.embedding_dim


        w2v_model = gensim.models.word2vec.Word2Vec.load("./embedding/w2v.model").wv
        fasttext_model = gensim.models.word2vec.Word2Vec.load("./embedding/fasttext.model").wv
        self.vocab_size = w2v_model.vectors.shape[0]+1
        # 字向量维度
        self.embed = nn.Embedding(self.vocab_size, self.embedding_dim)

        w2v_embed_matrix = w2v_model.vectors
        fasttext_embed_matrix = fasttext_model.vectors
        #             embed_matrix = w2v_embed_matrix
        embed_matrix = np.concatenate(
            [w2v_embed_matrix, fasttext_embed_matrix], axis=1
        )
        oov_embed = np.zeros((1, embed_matrix.shape[1]))
        embed_matrix = torch.from_numpy(np.vstack((oov_embed, embed_matrix)))
        self.embed.weight.data.copy_(embed_matrix)
        self.embed.weight.requires_grad = False

        self.spatial_dropout = SpatialDropout(drop_prob=0.5)
        self.lstm = nn.GRU(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_size1,
            num_layers=self.num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=self.dropout,
        )
        self.tanh = nn.Tanh()
        self.w = nn.Parameter(torch.zeros(self.hidden_size1 * 2))
        self.fc1 = nn.Linear(
            self.hidden_size1 * 2 + self.embedding_dim, self.hidden_size2
        )
        self.maxpool = nn.MaxPool1d(self.seq_len)
        self.fc2 = nn.Linear(self.hidden_size2, self.num_classes)
        self._init_parameters()

    def forward(self, x,label=None):
        embed = self.embed(x)
        spatial_embed = self.spatial_dropout(embed)
        H, _ = self.lstm(spatial_embed)
        M = self.tanh(H)
        alpha = F.softmax(torch.matmul(M, self.w), dim=1).unsqueeze(-1)
        out = H * alpha
        out = torch.cat((embed, out), 2)
        out = F.relu(out)
        out = out.permute(0, 2, 1)
        out = self.maxpool(out).squeeze(-1)
        out = self.fc1(out)
        out = self.fc2(out)
        if label is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(out.view(-1, self.num_classes).float(),
                            label.view(-1, self.num_classes).float())
            return loss
        else:
            return out

    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.kaiming_normal_(p)
            else:
                nn.init.constant_(p, 0)

class SpatialDropout(nn.Module):
    def __init__(self, drop_prob):
        super(SpatialDropout, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, inputs):
        output = inputs.clone()
        if not self.training or self.drop_prob == 0:
            return inputs
        else:
            noise = self._make_noise(inputs)
            if self.drop_prob == 1:
                noise.fill_(0)
            else:
                noise.bernoulli_(1 - self.drop_prob).div_(1 - self.drop_prob)
            noise = noise.expand_as(inputs)
            output.mul_(noise)
        return output

    @staticmethod
    def _make_noise(inputs):
        return inputs.new().resize_(
            inputs.size(0), *repeat(1, inputs.dim() - 2), inputs.size(2)
        )