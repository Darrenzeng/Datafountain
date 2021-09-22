from pretrain.word_to_vec import WordToVec
from pretrain.glove_vec import GloveVec
from helper.preprocess import train_pro, test_pro
from gensim.models import Word2Vec
from glove import Glove
from tqdm import tqdm
import torch
import directory
import os


class PretrainVector(object):
    @staticmethod
    def gen_corpus():
        train_df = train_pro(directory.TRAIN_SET_PATH)  #加载训练集
        # test_df = test_pro(directory.TEST_SET_A_PATH)  # 用A榜测试集参与预训练

        train_num = len(train_df)
        # test_num = len(test_df)

        with open(directory.CORPUS_PATH, 'a') as f:
            f.seek(0)
            f.truncate()

            for i in tqdm(range(train_num)):
                des_train = train_df.iloc[i, 0]

                des_train = ' '.join(str(train_item) for train_item in des_train)

                f.write(str(des_train) + '\n')

            # for j in tqdm(range(test_num)):
            #     des_test = test_df.iloc[j, 0]

            #     des_test = ' '.join(str(test_item) for test_item in des_test)

            #     f.write(str(des_test) + '\n')

            f.close()

    def load_pretrained_vec(self, vec_type):
        #创造语料库文件夹
        if not os.path.exists(directory.VECTOR_DIR):
            os.makedirs(directory.VECTOR_DIR)
        # 产生语料库corpus.txt
        if not os.path.exists(directory.CORPUS_PATH):
            self.gen_corpus()

        # 如果没有对应词向量，就先进行预训练
        if vec_type == 'word2vec':
            if not os.path.exists(directory.WORD_TO_VECTOR_PATH):
                WordToVec().word2vec_pretrain()

            word2vec_model = Word2Vec.load(directory.WORD_TO_VECTOR_PATH)

            word2vec_pretrained_vec = torch.from_numpy(word2vec_model.wv.vectors)

            return word2vec_pretrained_vec
        elif vec_type == 'glove':
            if not os.path.exists(directory.GLOVE_VECTOR_PATH):
                GloveVec().glove_pretrain()

            glove_model = Glove.load(directory.GLOVE_VECTOR_PATH)

            glove_pretrained_vec = torch.from_numpy(glove_model.word_vectors).to(torch.float32)

            return glove_pretrained_vec
        elif vec_type == 'concat':
            if not os.path.exists(directory.WORD_TO_VECTOR_PATH):
                WordToVec().word2vec_pretrain()#训练word2vec词向量模型，得到glove_vec.model

            if not os.path.exists(directory.GLOVE_VECTOR_PATH):
                GloveVec().glove_pretrain()#训练glove词向量模型， 得到word_to_vec.model

            word2vec_model = Word2Vec.load(directory.WORD_TO_VECTOR_PATH)

            word2vec_pretrained_vec = torch.from_numpy(word2vec_model.wv.vectors) #[1564, 100]

            glove_model = Glove.load(directory.GLOVE_VECTOR_PATH)

            glove_pretrained_vec = torch.from_numpy(glove_model.word_vectors).to(torch.float32) #[3263, 100]

            return torch.cat((word2vec_pretrained_vec, glove_pretrained_vec), dim=1)#将word2vec和glove拼接
