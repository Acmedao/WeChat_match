# ！/usr/bin/env python
# -*- coding: utf-8 -*-#
# Created by lzy on 2021/5/23 16:10
# @File    :feature_built.py

import os
import time
import logging

from manage_baseline.comm import ACTION_LIST

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__file__)
import numpy as np
import pandas as pd

# 存储数据的根目录
ROOT_PATH = ".\\..\\data"
# 比赛数据集路径
DATASET_PATH = os.path.join(ROOT_PATH, "wechat_algo_data1")
# 训练集
USER_ACTION = os.path.join(DATASET_PATH, "user_action.csv")
FEED_INFO = os.path.join(DATASET_PATH, "feed_info.csv")
FEED_EMBEDDINGS = os.path.join(DATASET_PATH, "feed_embeddings.csv")
# 测试集
TEST_FILE = os.path.join(DATASET_PATH, "test_a.csv")
END_DAY = 15
SEED = 2021

# 用于构造特征的字段列表
FEA_COLUMN_LIST = ["read_comment", "like", "click_avatar",  "forward", "comment", "follow", "favorite"]

def process(item):
    if item["play"] > item["videoplayseconds"]:
        return 1, item["play"] / item["videoplayseconds"]
    else:
        if item["videoplayseconds"]>8*60*1000 and item["play"]>3.5*60*100:
            return 1, item["play"] / item["videoplayseconds"]
        elif item["play"] / item["videoplayseconds"] > 0.8:
            return 1, item["play"] / item["videoplayseconds"]
        else: return 0, item["play"] / item["videoplayseconds"]

def specil_statis_feature(start_day=1, before_day=7):
    history_data = pd.read_csv(USER_ACTION)[["userid", "date_", "feedid", "play", "stay"] + FEA_COLUMN_LIST]
    for action in ACTION_LIST:
        history_data = history_data.drop_duplicates(subset=['userid', 'feedid', action], keep='last')
    feed_info = pd.read_csv(FEED_INFO)[["feedid", "authorid", "videoplayseconds", "bgm_song_id", "bgm_singer_id"]]
    feed_info[["authorid", "bgm_song_id", "bgm_singer_id"]] += 1  # 0 用于填未知
    feed_info[["authorid", "bgm_song_id", "bgm_singer_id", "videoplayseconds"]] = \
        feed_info[["authorid", "bgm_song_id", "bgm_singer_id", "videoplayseconds"]].fillna(0)
    history_data = pd.merge(history_data, feed_info, how="left", on="feedid")
    res_arr = []
    for start in range(start_day, END_DAY - before_day + 1):
        temp = history_data[((history_data["date_"]) >= start) & (history_data["date_"] < (start + before_day))]
        temp["play"] = temp["play"] / 1000
        temp["stay"] = temp["stay"] / 1000
        #temp["is_wanbo"], temp["play_rate"] = zip(*temp.apply(process, axis=1))
        temp["play_rate"] = temp["play"] / temp["videoplayseconds"]
        #todo 播出比率需要调整
        play_rate_index = temp.columns.tolist().index("play_rate")
        temp["play_rate"] = temp.apply(lambda item: 1 if item[play_rate_index] > 1 else item[play_rate_index], axis=1)
        temp["is_wanbo"] = temp.apply(lambda item:0 if item[play_rate_index]<0.8 else 1, axis=1)
        temp = temp.drop(columns=['date_'])
        base_column = temp[["userid", "play", "stay"]]
        base_statis_result = base_column.groupby(["userid"]).agg(["sum","count","mean" ]).reset_index()
        base_statis_result.columns = [''.join(col) for col in base_statis_result.columns]
        base_statis_result = base_statis_result.drop(columns=["staycount"])
        base_id_statis = temp[["userid","feedid","bgm_song_id","bgm_singer_id"]] \
                    .groupby(["userid"])[["feedid","bgm_song_id","bgm_singer_id"]].nunique().reset_index() \
                    .rename(columns={"feedid":"feedid_count","bgm_song_id":"song_count","bgm_singer_id":"singer"})
        replay_num = temp[(temp["videoplayseconds"] < temp["play"])].groupby("userid").agg("count").reset_index() \
            .rename(columns={"feedid":"replay_num"})[["userid","replay_num"]]

        stay_not_play = temp[(temp["play"]==0) & (temp["stay"]>0)].groupby("userid").agg("count").reset_index() \
            .rename(columns={"feedid": "stay_not_play"})[["userid", "stay_not_play"]]

        base_column["play_stay_rate_sum"] = base_column["play"] / base_column["stay"]
        play_stay_rate_sum = base_column.groupby(["userid"])["play_stay_rate_sum"].agg("sum").reset_index()

        wanbo_num = temp.groupby(["userid"])["is_wanbo","play_rate"].agg("sum").reset_index().rename(columns={"is_wanbo":"wanbo_num"})
        wanbo_num_and_base_id_statis = pd.merge(wanbo_num, base_id_statis, how="inner", on="userid")
        wanbo_num_and_base_id_statis["wanbo_rate"] = wanbo_num_and_base_id_statis["wanbo_num"] / wanbo_num_and_base_id_statis["feedid_count"]

        item = pd.merge(base_statis_result, base_id_statis, how="left", on="userid")
        item = pd.merge(item, replay_num, how="left", on="userid")
        item = pd.merge(item, stay_not_play, how="left", on="userid")
        item = pd.merge(item, play_stay_rate_sum, how="left", on="userid")
        all_statis_frature = pd.merge(item, wanbo_num_and_base_id_statis[["userid", "wanbo_num", "wanbo_rate", "play_rate"]], how="left", on="userid")
        all_statis_frature["play_stay_rate_mean"] = all_statis_frature["play_stay_rate_sum"] / all_statis_frature["feedid_count"]
        all_statis_frature = all_statis_frature.fillna(0)
        all_statis_frature.applymap(lambda value: np.log(value + 1))
        all_statis_frature["date_"] = start + before_day
        res_arr.append(all_statis_frature)
        print(start)
    dim_feature = pd.concat(res_arr)
    #dim_feature.iloc[:,1:-1] = (dim_feature.iloc[:,1:-1] - dim_feature.iloc[:,1:-1].min()) / (dim_feature.iloc[:,1:-1].max() - dim_feature.iloc[:,1:-1].min())
    dim_feature.to_csv("E:/WeChartCompetition/code/data/feature/userid_statistic_feature.csv", index=False)
specil_statis_feature()

# data = pd.read_csv("E:/WeChartCompetition/code/data/feature/userid_statistic_feature.csv")
# print(data.shape)













