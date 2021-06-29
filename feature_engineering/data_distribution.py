# ！/usr/bin/env python
# -*- coding: utf-8 -*-#
# Created by lzy on 2021/5/29 14:54
# @File    :data_distribution.py


import pandas as pd
from collections import Counter
import numpy as np
from sklearn.decomposition import PCA

data = pd.read_csv("E:\WeChartCompetition\code\data\wechat_algo_data1\\feed_embeddings.csv")
temp = [value.split(" ") for value in data.feed_embedding.values.tolist()]
result = list(map(lambda line: [float(value) for value in line if value!='' and value!='-' and value !='.' and value !=' '],temp))
result = np.array(result).astype("float32")
estimator = PCA(n_components=16)
pca_X_train = estimator.fit_transform(result)


df = pd.DataFrame(pca_X_train)
df.columns = ["c_"+str(i) for i in range(16)]
df["feedid"] = data.feedid.values
df.to_csv("E:\WeChartCompetition\code\data\wechat_algo_data1\\feed_embeddings_pca.csv", index=False)
print(df)

