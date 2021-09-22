

<center><font size=62>第五届“达观杯”风险标签识别赛</font></center>


+ 第五届“达观杯” 基于大规模预训练模型的风险事件标签识别赛后总结

+ 比赛主页

  https://www.datafountain.cn/competitions/512/ranking?isRedance=0&sch=1804&page=2

# 选择的模型

## <font color=gree>传统方法</font>

+ DP-CNN
+ HAN
+ Capsule
+ RNN+Attention

##  <font color=gree>预训练模型方法</font>

+ Bert
+ Bert_large
+ Nezha
+ Nezha_large

# 数据

已经放在datasets（包含处理过的数据）中

预训练整个项目的文件结构如下：

```python
Datafountain/
  └──datasets    ------------------------------存放的数据
  └──nezha-pytorch/
    └──nezha_model-----------------------------存放原始模型
    └──pretrain/
      └──NEZHA     ----------------------------存放的模型
      └──nezha_out ----------------------------存放预训练后的模型
      └──NLP_Utils.py
      └──train_nezha.py
    └──fintuing/
      └──models_add4_mlm0.5 -------------------存放微调后的模型
      └──NEZHA     ----------------------------存放的模型
      └──submit    ----------------------------生成的提交文件
      └──Config.py
      └──model.py
      └──NEZHA_main.py
      └──predict.py
      └──predict_TTa.py
      └──run.sh
      └──utils.py
      
  └──bert-pytorch/
    └──bert_base_chinese ----------------------存放原始模型
    └──transformers1
      └──若干文件
    └──bert_out -------------------------------存放预训练后的模型
    └──NLP_Utils.py
    └──train_nezha.py
    └──bert_fintuing1/
      └──models_add4_mlm0.5 -------------------存放微调后的模型
      └──NEZHA     ----------------------------存放的模型(我们bert在nezha基础上改的)
      └──submit    ----------------------------生成的提交文件
      └──Config.py
      └──model.py
      └──main_bert.py
      └──predict.py
      └──utils.py
  └──general_method/
      后续补充
```

## <font color=blue>训练策略</font>

+ 保留标点符号，使用特殊的token替代，oov的方法

  因为在预训练中，标点符号也有特殊的意义，比如说“！”，在情感分析任务中就提供了丰富的情感知识，所以保留标点符号进行预训练是有理论依据的。

+ 更改训练的mlm掩码概率，0.15-->0.5

  原因是本任务给的脱敏训练集数量量少，没有办法得到充分的训练。根据以前的比赛视频中选手所说，增大训练任务的难度，有利于使得模型获得更深层的特征信息。

+ 使用双训练任务：mlm任务，nsp任务

  此任务由于自己代码能力有限，没有做成功。但是根据上一届天池的脱敏数据比赛中榜一选手所说，将两个任务的embedding层交替训练，会有很好的收获。[选手视频](https://gaiic.tianchi.aliyun.com/)

+ 数据的截取方式：尾部截断，首部截断，首尾截断

  

# 提分的技巧

## 微调上分

+ 混合精度fp16加速微调训练

+ k折交叉训练

  + 一般随机采样

    ```python
    from sklearn.model_selection import KFold
    ```

  + 分层抽样

    ```python
    from sklearn.model_selection import StratifiedKFold
    ```

+ 加入对抗学习

  + fgm(1倍时间)
  + pgd(1.5倍时间)

+ 优化器选择: [transformers地址](https://huggingface.co/transformers/main_classes/optimizer_schedules.html?highlight=adamw#transformers.AdamW)

  + AdamW (<font color=black>测试出来效果最佳</font>)
  + lookahead
  + Adafactor

+ schedule学习率的预热方法: [transformers地址](https://huggingface.co/transformers/main_classes/optimizer_schedules.html?highlight=adamw#schedules)

  + constant_schedule
  + constant_schedule_with_warmup
  + linear_schedule_with_warmup
  + cosine_schedule_with_warmup (<font color=black>效果最佳</font>)
  + cosine_with_hard_restarts_schedule_with_warmup
  + polynomial_decay_schedule_with_warmup

+ 数据增强方法(本次脱敏数据中效果不佳，最终弃用)

  ​			[回译和EDA----CSDN代码链接1](https://blog.csdn.net/dabingsun/article/details/103232188)

  ​			[回译和EDA----CSDN代码链接2(需付费)](https://blog.csdn.net/herosunly/article/details/113997077?utm_medium=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-1.no_search_link&depth_1-utm_source=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-1.no_search_link)

  + 回译(脱敏数据不适用)

    ps: <font color=red>**源文本--->其他语言（英语等）--->源文本的回译**</font>

    可理解为对源文本进行了同义词替换、语法结构替换、删除无关紧要词汇等**丰富的变换**。

    ```python
    from pygtrans import Translate
    
    def backTran(wordss):
        client = Translate()
        text1 = client.translate(wordss)
        text2 = client.translate(text1.translatedText, target='en')
        return text2.translatedText
    ```

  + EDA

    EDA就是在源文本上进行一些简单的变换，主要包括同义词替换、随机插入、随机删除、随机交换词这四种。

    ```python
    def getEda(df):
        title=df.title.tolist() #我们的测试数据是论文分类，一组数据中有title,abstract
        abstract=df.abstract.tolist()
        categories=df.categories.tolist()#标签
        len1=len(title)
        exTitle=[]
        exabs=[]
        excat=[]
        for i in range(len1):
            lentitle=len(title[i])
            lenabstrct=len(abstract[i])
            txtitle=backTran(title[i])
            txabs=backTran(abstract[i])
            if i==0:
                print(txtitle)
                print(txabs)
            if i%1000==0:
                print(i)
            exTitle.append(txtitle)
            exabs.append(txabs)
            excat.append(categories[i])  
        return exTitle,exabs,excat
    ```

  + TTA测试时数据增强（Test-Time Augmentation）

    ps原理：可以对一幅图像(或文字)做多种变换,创造出多个不同版本,包括**不同区域裁剪**和**更改缩放程度**等,然后对多个版本数据进行计算，最后得到平均输出作为最终结果,提高了结果的稳定性和精准度。

    + 在这次比赛中，我们选择在测试集上，对每一个数据进行以标点符号分割为几句话，然后倒序交换句子的顺序。得到tta测试数据

    ```python
    import pandas as pd
    # from sklearn.feature_extraction.text import TfidfVectorizer
    # from sklearn.model_selection import StratifiedKFold
    # from sklearn.metrics import f1_score
    import numpy as np
    from gensim.models import Word2Vec
    import pandas as pd
    import jieba
    import os
    
    train = pd.read_csv('datagrand_2021_train.csv')
    
    fuhao=['，','！','。','？']
    tmp=train.text.tolist()
    
    #先统计每个符号出现的位置
    totalFuhao=[]
    for text in tmp:
        tF=[]
        t=text.split()
        for j in t:
            if j in fuhao:
                tF.append(j)
    #             print("ok")
        totalFuhao.append(tF)
    
    #然后再将所有符号换成英文逗号
    def getClean(document):
        text = str(document)
        text = text.replace('，', ',')
        text = text.replace('！', ',')
        text = text.replace('？', ',')
        text = text.replace('。', ',')
        return text
      
    #以逗号划分句子，交换句子顺序
    def suffer(document):
        text=str(document)
        t=text.split(',')
        newT=t[::-1]
        return " , ".join(newT)
     
    train['text']=train['text'].apply(lambda x:getClean(x))#将所有的标点符号替换为逗号
    #句子逆序
    train['text']=train['text'].apply(lambda x: suffer(x))#以逗号分句子为几句话，同时交换几句话的位置(逆序)
    
    #符号还原
    def tranform(df):
        ixd=0
        totaldx=0
        ans=[]
        for text in df:
            arr=[]
            dinx=0
            t=text.split()
            if ixd==0:
                print(t)
            for j in t:
                if j==',':
                    arr.append(totalFuhao[totaldx][dinx])
                    dinx+=1
                else :
                    arr.append(j)
            ixd+=1
            totaldx+=1
            if ixd==1 :
                print(" ".join(arr))
            ans.append(" ".join(arr))
        return ans
      
    #将倒序后的句子进行符号还原
    newText=train['text'].tolist()
    neT=tranform(newText)
    train['text']=neT
    train.to_csv("./ttatest.csv",index=False)
    ```

  + 闭包对偶：q1-q2, q2-q1得到两组新特征

    
