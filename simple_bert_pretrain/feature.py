from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
import argparse
import numpy as np
import pandas as pd
import copy
from scipy.sparse import vstack
import re

def process_text(document):
    #删除标点符号
    text = str(document)
    text = text.replace("，", '')
    text = text.replace('！', '')
    # 用单个空格替换多个空格
    text = re.sub(r'\s+', ' ', text, flags=re.I)
    return text

def get_seqs(file):
    text = []
    # with open('data/{}.csv'.format(file), 'r') as f:
    #     for line in f:
    #         items = line.split('\t')
    #         text.append(items[0])
    #         text.append(items[1])
    data = pd.read_csv('data/{}.csv'.format(file))
    for idx in range(len(data)):
        processed_text = process_text(data.iloc[idx].text).strip()
        text.append(processed_text)
    return text

def save_feature(file, n, features): #5个(200000, 128)
    tmp = [] #存5个(100000, 256)
    for feature in features:
        shape = feature.shape
        if shape[0] == n:#10万
            tmp.append(feature.reshape(n, shape[1]))
        else:
            tmp.append(feature.reshape(n, shape[1]*2))#(200000, 128) --> (100000, 256)
    feature = np.concatenate(tmp, 1) #以第一维度来拼接：5个(100000, 256) --> (100000, 1152)
    print(file, 'feature shape:', feature.shape)
    np.save('result/features/{}.npy'.format(file), feature)

class BM25(object):
    def __init__(self, b=0.75, k1=1.6):
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 4), norm=None, smooth_idf=False, token_pattern='\d+')
        self.b = b
        self.k1 = k1

    def fit(self, X):
        self.vectorizer.fit(X)#转换文本
        count = super(TfidfVectorizer, self.vectorizer).transform(X)
        self.avdl = count.sum(1).mean()

    def transform(self, X):
        b, k1, avdl = self.b, self.k1, self.avdl
        count = super(TfidfVectorizer, self.vectorizer).transform(X)
        len_X = count.sum(1).A1
        w = k1*(1-b+b*len_X/avdl)
        idf = self.vectorizer._tfidf.idf_-1
        rows, cols = count.nonzero()
        for i, (row, col) in enumerate(zip(rows, cols)):
            v = count.data[i]
            count.data[i] = (k1+1)*v/(w[row]+v)*idf[col]
        return count #(250000, 1101597)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Text_match')
    parser.add_argument('-train', type=str, default='datagrand_2021_train', choices=['datagrand_2021_train.csv'])
    parser.add_argument('-test', type=str, default='datagrand_2021_test', choices=['datagrand_2021_test'])
    args = parser.parse_args()
    train = get_seqs(args.train) #获得训练集所有文本
    train += train[-1:]
    test = get_seqs(args.test)
    text = train+test
    

    tfidf = TfidfVectorizer(ngram_range=(1, 4), token_pattern='\d+')
    tfidf_feature = tfidf.fit_transform(text)#获得文本的tiidf特征值(250000, 1101597)          新：(20013, 1127607)
    print(tfidf_feature.shape)
    svd1 = TruncatedSVD(n_components=128).fit_transform(tfidf_feature)#获得文本的SVD特征值
    print(svd1.shape) #(250000, 128)
    
    bm25 = BM25()
    bm25.fit(text) #将文本转换机器识别的值
    bm25_feature = bm25.transform(text) #将值转为相应的特征
    print(bm25_feature.shape) #(250000, 1101597)
    svd2 = TruncatedSVD(n_components=128).fit_transform(bm25_feature)
    print(svd2.shape) #(250000, 128)
    
    count = CountVectorizer(ngram_range=(1, 1), token_pattern='\d+')
    count_feature = count.fit_transform(text) #(250000, 20600)
    shape = count_feature.shape
    even_id = np.arange(0, shape[0], 2)#0到250000， 2为跨度  偶位数(125000,)
    count_even = count_feature[even_id]  #(125000, 20600)
    odd_id = np.arange(1, shape[0], 2)                     #奇位数(125000,)
    count_odd = count_feature[odd_id]  #(125000, 20600)
    count_d1 = count_odd-count_even #这里是在做对偶数据增强/闭包数据增强？(125000, 20600)
    count_d2 = count_even-count_odd


    count_d3 = copy.deepcopy(count_d1)
    count_d3[count_d3<0] = 0 #将小于0的值换为0
    count_d4 = copy.deepcopy(count_d2)
    count_d4[count_d4<0] = 0
    count_delta = vstack([count_d1, count_d2]) #闭包数据
    count_nonneg_d = vstack([count_d3, count_d4])#也是闭包数据，只不过将里面低于0的值全都换为0
    count_abs = np.abs(count_odd-count_even)  # count_d1的绝对值 (125000, 20600)
    svd3 = TruncatedSVD(n_components=128).fit_transform(count_delta) #(250000, 128)      (20014, 128)
    svd4 = TruncatedSVD(n_components=128).fit_transform(count_nonneg_d)  #(250000, 128)  (20014, 128)
    svd5 = TruncatedSVD(n_components=128).fit_transform(count_abs)  #(250000, 128)       (20014, 128)
    d = shape[0]//2
    ids = np.concatenate([np.arange(d).reshape(d, 1), np.arange(d, d*2).reshape(d, 1)], 0).reshape(d*2) #(250000, 1)-->(250000,)得到ids来组合做好的数据
    svd3 = svd3[ids]
    svd4 = svd4[ids]
    
    n1, n2, n3, n4 = len(train), len(test), len(train)//2, len(test)//2  #20万， 5万， 10万， 2.5万
    save_feature(args.train, n3, [svd1[:n1], svd2[:n1], svd3[:n1], svd4[:n1], svd5[:n3]]) #做成新的训练集(100000, 1152)
    save_feature(args.test, n4, [svd1[-n2:], svd2[-n2:], svd3[-n2:], svd4[-n2:], svd5[-n4:]]) #(25000, 1152)