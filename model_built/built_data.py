# coding: utf-8
import os
import time
import logging

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__file__)
import numpy as np
import pandas as pd

# 存储数据的根目录
ROOT_PATH = ".\\..\\data2"
# 比赛数据集路径
DATASET_PATH = os.path.join(".\\..\\data", "wechat_algo_data1")
# 训练集
USER_ACTION = os.path.join(DATASET_PATH, "user_action.csv")
FEED_INFO = os.path.join(DATASET_PATH, "feed_info.csv")
FEED_EMBEDDINGS = os.path.join(DATASET_PATH, "feed_embeddings.csv")
# 测试集
TEST_FILE = os.path.join(DATASET_PATH, "test_a.csv")
END_DAY = 15
SEED = 1024

# 初赛待预测行为列表
ACTION_LIST = ["read_comment", "like", "click_avatar", "forward"]
# 复赛待预测行为列表
# ACTION_LIST = ["read_comment", "like", "click_avatar",  "forward", "comment", "follow", "favorite"]
# 用于构造特征的字段列表
FEA_COLUMN_LIST = ["read_comment", "like", "click_avatar", "forward", "comment", "follow", "favorite"]
# 负样本下采样比例(负样本:正样本)
ACTION_SAMPLE_RATE = {"read_comment": 0.2, "like": 0.2, "click_avatar": 0.2, "forward": 0.2, "comment": 0.1,
                      "follow": 0.1, "favorite": 0.1}
# 各个阶段数据集的设置的最后一天
STAGE_END_DAY = {"online_train": 14, "offline_train": 13, "evaluate": 14, "submit": 15}

user_action = pd.read_csv("./../data/wechat_algo_data1/user_action.csv")
feed_info = pd.read_csv("./../data/wechat_algo_data1/feed_info.csv")


# 采样
def sample():
    global user_action
    for value in ACTION_LIST:
        user_action = user_action.drop_duplicates(subset=['userid', 'feedid', value], keep='last')

    for action in ACTION_LIST:
        temp = user_action[user_action[action] == 0]
        df_neg = temp.sample(frac=ACTION_SAMPLE_RATE[action], random_state=1024, replace=False)
        df_all = pd.concat([df_neg, user_action[user_action[action] == 1]])
        col = ["userid", "feedid", "date_", "device"] + [action]
        file_name = os.path.join('./../data2/train/' + action + "_generate_sample.csv")
        print('Save to: %s' % file_name)
        df_all[col].to_csv(file_name, index=False)


# 拼接特征
def concat():
    for action in ACTION_LIST:
        file_name = os.path.join('./../data2/train/' + action + "_generate_sample.csv")
        data = pd.read_csv(file_name)
        data_feature = pd.merge(data, feed_info, how="left", on="feedid")
        data_feature = data_feature[['userid', "date_", 'feedid', "device", "authorid", "bgm_singer_id", "bgm_song_id",
                                     "manual_keyword_list", "machine_keyword_list", "manual_tag_list",
                                     "videoplayseconds"] + [action]]
        data_feature[["authorid", "bgm_song_id", "bgm_singer_id"]] += 1  # 0 用于填未知
        data_feature[["authorid", "bgm_song_id", "bgm_singer_id", "videoplayseconds"]] = \
            data_feature[["authorid", "bgm_song_id", "bgm_singer_id", "videoplayseconds"]].fillna(0)
        data_feature["videoplayseconds"] = np.log(data_feature["videoplayseconds"] + 1.0)
        data_feature[["authorid", "bgm_song_id", "bgm_singer_id"]] = \
            data_feature[["authorid", "bgm_song_id", "bgm_singer_id"]].astype(int)
        data_feature[["manual_keyword_list", "machine_keyword_list", "manual_tag_list"]] = \
            data_feature[["manual_keyword_list", "machine_keyword_list", "manual_tag_list"]].fillna(9999)
        file_name = os.path.join('./../data2/sample/' + action + "_concat_sample.csv")
        print('Save to: %s' % file_name)
        data_feature.to_csv(file_name, index=False)


# submit数据
def submit():
    data_feature = pd.read_csv("./../data/wechat_algo_data1/test_a.csv")
    data_feature = pd.merge(data_feature, feed_info, how="left", on="feedid")
    data_feature = data_feature[
        ['userid', 'feedid', "device", "authorid", "bgm_singer_id", "bgm_song_id", "manual_keyword_list",
         "machine_keyword_list", "manual_tag_list", "videoplayseconds"]]
    data_feature["date_"]=15
    data_feature[["authorid", "bgm_song_id", "bgm_singer_id"]] += 1  # 0 用于填未知
    data_feature[["authorid", "bgm_song_id", "bgm_singer_id", "videoplayseconds"]] = \
        data_feature[["authorid", "bgm_song_id", "bgm_singer_id", "videoplayseconds"]].fillna(0)
    data_feature["videoplayseconds"] = np.log(data_feature["videoplayseconds"] + 1.0)
    data_feature[["authorid", "bgm_song_id", "bgm_singer_id"]] = \
        data_feature[["authorid", "bgm_song_id", "bgm_singer_id"]].astype(int)
    data_feature[["manual_keyword_list", "machine_keyword_list", "manual_tag_list"]] = \
        data_feature[["manual_keyword_list", "machine_keyword_list", "manual_tag_list"]].fillna(9999)
    file_name = os.path.join('./../data2/sample/' + "all" + "_concat_sample.csv")
    print('Save to: %s' % file_name)
    data_feature.to_csv(file_name, index=False)



def main():
    sample()
    concat()
    submit()


if __name__ == "__main__":
    main()
