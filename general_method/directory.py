#本文件主要设置各种文件路径
# import sys
# sys.path.append('./')
# 数据集存放目录
DATASET_DIR = './datasets'

# 训练集路径
TRAIN_SET_PATH = DATASET_DIR + '/datagrand_2021_train.csv'

# 测试集路径
TEST_SET_B_PATH = DATASET_DIR + '/datagrand_2021_test.csv'

# 词向量目录
VECTOR_DIR = './vector'

# 语料库文本文件路径
CORPUS_PATH = VECTOR_DIR + '/corpus.txt'

# word2vec词向量模型路径
WORD_TO_VECTOR_PATH = VECTOR_DIR + '/word_to_vec.model'

# glove词向量模型路径
GLOVE_VECTOR_PATH = VECTOR_DIR + '/glove_vec.model'

# 训练模型目录
MODEL_DIR = './model/saved'

# 提交的csv文件目录
SUBMISSION_DIR = './submission'

SUBMISSION_PATH = SUBMISSION_DIR + '/submission.csv'

