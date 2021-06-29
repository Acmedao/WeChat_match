# ！/usr/bin/env python
# -*- coding: utf-8 -*-#
# Created by lzy on 2021/5/20 23:38
# @File    :data_debug.py


import pandas as pd
from collections import Counter

user_action = "E:/WeChartCompetition/data/user_action.csv"
feed_info = "E:/WeChartCompetition/data/feed_info.csv"
feed_embedding = "E:/WeChartCompetition/data/feed_embeddings.csv"
test_a = "E:/WeChartCompetition/data/test_a.csv"
data = pd.read_csv(user_action)
data1 = data[:1]
# print(data1)
# print("数据组成"+str(data.shape))
# print("数据维度"+str(data.ndim))
# print("数据大小"+str(data.size))
