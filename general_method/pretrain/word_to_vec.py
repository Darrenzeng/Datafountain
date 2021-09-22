from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import warnings
import logging
import directory
import multiprocessing

warnings.filterwarnings('ignore')


class WordToVecConfig(object):
    def __init__(self):
        self.sg = 1
        self.window = 8
        self.size = 100
        self.workers = multiprocessing.cpu_count()
        self.sample = 1e-3
        self.min_count = 20
        self.hs = 1
        self.iter = 2#默认50

#词向量模型，产生词向量
class WordToVec(object):
    def __init__(self):
        word2vec_config = WordToVecConfig()

        self.sg = word2vec_config.sg
        self.window = word2vec_config.window
        self.size = word2vec_config.size
        self.workers = word2vec_config.workers
        self.sample = word2vec_config.sample
        self.min_count = word2vec_config.min_count
        self.hs = word2vec_config.hs
        self.iter = word2vec_config.iter

    def word2vec_pretrain(self):
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

        corpus = open(directory.CORPUS_PATH, 'r')
        #!!!!!!!!
        #将语料库转化为词向量
        word_to_vec_model = Word2Vec(
            LineSentence(corpus), sg=self.sg, window=self.window,
            vector_size=self.size, workers=self.workers, sample=self.sample,
            min_count=self.min_count, hs=self.hs, epochs=self.iter, seed=7
        )
        

        word_to_vec_model.save(directory.WORD_TO_VECTOR_PATH)
