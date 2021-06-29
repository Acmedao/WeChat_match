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
SEED = 1024

# 初赛待预测行为列表
ACTION_LIST = ["read_comment", "like", "click_avatar", "forward"]
# 复赛待预测行为列表
# ACTION_LIST = ["read_comment", "like", "click_avatar",  "forward", "comment", "follow", "favorite"]
# 用于构造特征的字段列表
FEA_COLUMN_LIST = ["read_comment", "like", "click_avatar",  "forward", "comment", "follow", "favorite"]
# 负样本下采样比例(负样本:正样本)
ACTION_SAMPLE_RATE = {"read_comment": 0.25, "like": 0.25, "click_avatar": 0.25, "forward": 0.25, "comment": 0.1, "follow": 0.1, "favorite": 0.1}
# 各个阶段数据集的设置的最后一天
STAGE_END_DAY = {"online_train": 14, "offline_train": 12, "evaluate": 13, "submit": 15}
# 各个行为构造训练数据的天数
ACTION_DAY_NUM = {"read_comment": 7, "like":7, "click_avatar": 7, "forward": 7, "comment": 7, "follow": 7, "favorite": 7}


def create_dir():
    """
    创建所需要的目录
    """
    # 创建data目录
    if not os.path.exists(ROOT_PATH):
        print('Create dir: %s'%ROOT_PATH)
        os.mkdir(ROOT_PATH)
    # data目录下需要创建的子目录
    need_dirs = ["offline_train", "online_train", "evaluate", "submit",
                 "feature", "model", "model/online_train", "model/offline_train"]
    for need_dir in need_dirs:
        need_dir = os.path.join(ROOT_PATH, need_dir)
        if not os.path.exists(need_dir):
            print('Create dir: %s'%need_dir)
            os.mkdir(need_dir)


def check_file():
    '''
    检查数据文件是否存在
    '''
    paths = [USER_ACTION, FEED_INFO, TEST_FILE]
    flag = True
    not_exist_file = []
    for f in paths:
        if not os.path.exists(f):
            not_exist_file.append(f)
            flag = False
    return flag, not_exist_file


def statis_data():
    """
    统计特征最大，最小，均值
    """
    paths = [USER_ACTION, FEED_INFO, TEST_FILE]
    pd.set_option('display.max_columns', None)
    for path in paths:
        df = pd.read_csv(path)
        print(path + " statis: ")
        print(df.describe())
        print('Distinct count:')
        print(df.nunique())


def statis_feature(start_day=1, before_day=7, agg=['sum']):
    """
    统计用户/feed 过去n天各类行为的次
    :param start_day: Int. 起始日期
    :param before_day: Int. 时间范围（天数）
    :param agg: String. 统计方法
    """
    history_data = pd.read_csv(USER_ACTION)[["userid", "date_", "feedid"] + FEA_COLUMN_LIST]
    feature_dir = os.path.join(ROOT_PATH, "feature")
    for dim in ["userid", "feedid"]:
        print(dim)
        user_data = history_data[[dim, "date_"] + FEA_COLUMN_LIST]
        res_arr = []
        for start in range(start_day, END_DAY-before_day+1):
            temp = user_data[((user_data["date_"]) >= start) & (user_data["date_"] < (start + before_day))]
            temp = temp.drop(columns=['date_'])
            temp = temp.groupby([dim]).agg(agg).reset_index()
            #temp.columns = list(map(''.join, temp.columns.values))
            if dim=="userid":
                temp.columns = ["userid"] + ["_".join(i) + "_user" for i in temp.columns.tolist() if i[0] not in ["userid", "feedid"]]
            else:
                temp.columns = ["feedid"] + ["_".join(i) + "_feed" for i in temp.columns.tolist() if i[0] not in ["userid", "feedid"]]
            temp["date_"] = start + before_day
            res_arr.append(temp)
        dim_feature = pd.concat(res_arr)
        #dim_feature.iloc[:, 1:-1] = (dim_feature.iloc[:, 1:-1] - dim_feature.iloc[:, 1:-1].min()) / (dim_feature.iloc[:, 1:-1].max() - dim_feature.iloc[:, 1:-1].min())
        feature_path = os.path.join(feature_dir, dim+"_feature.csv")
        print('Save to: %s'%feature_path)
        dim_feature.to_csv(feature_path, index=False)


def generate_sample(stage="offline_train"):
    """
    对负样本进行下采样，生成各个阶段所需样本
    :param stage: String. Including "online_train"/"offline_train"/"evaluate"/"submit"
    :return: List of sample df
    """
    day = STAGE_END_DAY[stage]
    if stage == "submit":
        sample_path = TEST_FILE
    else:
        sample_path = USER_ACTION
    stage_dir = os.path.join(ROOT_PATH, stage)
    df = pd.read_csv(sample_path)
    df_arr = []
    if stage == "evaluate":
        # 线下评估
        col = ["userid", "feedid", "date_", "device", "play", "stay"] + ACTION_LIST
        df = df[df["date_"] == day][col]
        file_name = os.path.join(stage_dir, stage + "_" + "all" + "_" + str(day) + "_generate_sample.csv")
        print('Save to: %s'%file_name)
        df.to_csv(file_name, index=False)
        df_arr.append(df)
    elif stage == "submit":
        # 线上提交
        file_name = os.path.join(stage_dir, stage + "_" + "all" + "_" + str(day) + "_generate_sample.csv")
        df["date_"] = 15
        print('Save to: %s'%file_name)
        df.to_csv(file_name, index=False)
        df_arr.append(df)
    else:
        # 线下/线上训练
        # 同行为取按时间最近的样本
        for action in ACTION_LIST:
            df = df.drop_duplicates(subset=['userid', 'feedid', action], keep='last')
        # 负样本下采样14-7+1
        for action in ACTION_LIST:
            action_df = df[(df["date_"] <= day) & (df["date_"] >= day - ACTION_DAY_NUM[action] + 1)]
            df_neg = action_df[action_df[action] == 0]
            df_neg = df_neg.sample(frac=ACTION_SAMPLE_RATE[action], random_state=SEED, replace=False)
            df_all = pd.concat([df_neg, action_df[action_df[action] == 1]])
            col = ["userid", "feedid", "date_", "device"] + [action]
            file_name = os.path.join(stage_dir, stage + "_" + action + "_" + str(day) + "_generate_sample.csv")
            print('Save to: %s'%file_name)
            df_all[col].to_csv(file_name, index=False)
            df_arr.append(df_all[col])
    return df_arr


def concat_sample(sample_arr, stage="offline_train"):
    """
    基于样本数据和特征，生成特征数据
    :param sample_arr: List of sample df
    :param stage: String. Including "online_train"/"offline_train"/"evaluate"/"submit"
    """
    day = STAGE_END_DAY[stage]
    # feed信息表
    feed_info = pd.read_csv(FEED_INFO)
    feed_info = feed_info.set_index('feedid')
    # 基于userid统计的历史行为的次数
    user_date_feature_path = os.path.join(ROOT_PATH, "feature", "userid_feature.csv")
    user_date_feature = pd.read_csv(user_date_feature_path)
    user_date_feature = user_date_feature.set_index(["userid", "date_"])
    # 基于feedid统计的历史行为的次数
    feed_date_feature_path = os.path.join(ROOT_PATH, "feature", "feedid_feature.csv")
    feed_date_feature = pd.read_csv(feed_date_feature_path)
    feed_date_feature = feed_date_feature.set_index(["feedid", "date_"])

    for index, sample in enumerate(sample_arr):
        features = ["userid", "feedid", "device", "authorid", "bgm_song_id", "bgm_singer_id", "manual_keyword_list",
                    "machine_keyword_list","manual_tag_list","videoplayseconds"]
        if stage == "evaluate":
            action = "all"
            features += ACTION_LIST
        elif stage == "submit":
            action = "all"
        else:
            action = ACTION_LIST[index]
            features += [action]
        print("action: ", action)
        sample = sample.join(feed_info, on="feedid", how="left", rsuffix="_feed")
        sample = sample.join(feed_date_feature, on=["feedid", "date_"], how="left", rsuffix="_feed")
        sample = sample.join(user_date_feature, on=["userid", "date_"], how="left", rsuffix="_user")
        # feed_feature_col = [b+i+"_feed" for b in FEA_COLUMN_LIST for i in ["_sum","_count","_mean"]]
        # user_feature_col = [b+i+"_user" for b in FEA_COLUMN_LIST for i in ["_sum","_count","_mean"]]
        feed_feature_col = [b+i+"_feed" for b in FEA_COLUMN_LIST for i in ["_sum"]]
        user_feature_col = [b+i+"_user" for b in FEA_COLUMN_LIST for i in ["_sum"]]
        sample[feed_feature_col] = sample[feed_feature_col].fillna(0.0)
        sample[user_feature_col] = sample[user_feature_col].fillna(0.0)
        sample[feed_feature_col] = np.log(sample[feed_feature_col] + 1.0)
        sample[user_feature_col] = np.log(sample[user_feature_col] + 1.0)
        sample["manual_keyword_list"] = sample["manual_keyword_list"].fillna("99999")
        sample["machine_keyword_list"] = sample["machine_keyword_list"].fillna("99999")
        sample["manual_tag_list"] = sample["manual_tag_list"].fillna("99999")
        #sample["manual_keyword_list"] = sample["manual_keyword_list"].apply(lambda item: np.array([np.int64(i) for i in str(item).split(";")]))
        features += feed_feature_col
        features += user_feature_col

        sample[["authorid", "bgm_song_id", "bgm_singer_id"]] += 1  # 0 用于填未知
        sample[["authorid", "bgm_song_id", "bgm_singer_id", "videoplayseconds"]] = \
            sample[["authorid", "bgm_song_id", "bgm_singer_id", "videoplayseconds"]].fillna(0)
        sample["videoplayseconds"] = np.log(sample["videoplayseconds"] + 1.0)

        sample[["authorid", "bgm_song_id", "bgm_singer_id"]] = \
            sample[["authorid", "bgm_song_id", "bgm_singer_id"]].astype(int)
        file_name = os.path.join(ROOT_PATH, stage, stage + "_" + action + "_" + str(day) + "_concate_sample.csv")
        print('Save to: %s'%file_name)
        userid_statistic_feature = pd.read_csv(os.path.join(ROOT_PATH, "feature", "userid_statistic_feature.csv"))
        features += ['playsum', 'playcount', 'playmean', 'staysum', 'staymean', 'feedid_count', 'song_count', 'singer', 'replay_num', 'stay_not_play',
                        'play_stay_rate_sum', 'wanbo_num', 'wanbo_rate', 'play_rate','play_stay_rate_mean']
        result = pd.merge(sample, userid_statistic_feature, how="left", on=["userid", "date_"])
        result[features].to_csv(file_name, index=False)


def main():
    t = time.time()
    #statis_data()
    logger.info('Create dir and check file')
    create_dir()
    flag, not_exists_file = check_file()
    if not flag:
        print("请检查目录中是否存在下列文件: ", ",".join(not_exists_file))
        return
    logger.info('Generate statistic feature')
    statis_feature()
    for stage in STAGE_END_DAY:
        logger.info("Stage: %s"%stage)
        logger.info('Generate sample')
        sample_arr = generate_sample(stage)
        logger.info('Concat sample with feature')
        concat_sample(sample_arr, stage)
    print('Time cost: %.2f s'%(time.time()-t))


if __name__ == "__main__":
    main()
    