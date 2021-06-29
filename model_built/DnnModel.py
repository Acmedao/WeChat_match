# ！/usr/bin/env python
# -*- coding: utf-8 -*-#
# Created by lzy on 2021/5/29 10:47
# @File    :model.py

import argparse
import tensorflow as tf
import os
import pandas as pd
import time
from tensorflow.python.ops import control_flow_ops
import numpy as np
from manage_baseline.comm import ACTION_LIST, STAGE_END_DAY
from manage_baseline.evaluation import uAUC, compute_weighted_score
from model_built.feature_select import base_feature_set
from tensorflow.contrib.layers import sparse_column_with_keys

parser = argparse.ArgumentParser()
parser.add_argument("--model_checkpoint_dir", help="checkpoint path", default=".\\..\\data2\\model\\")
parser.add_argument("--root_path", help="data dir", default=".\\..\\data2\\")
parser.add_argument("--batch_size", help="batch size", type=int, default=128)
parser.add_argument("--hidden_layers", type=str, default="512,256,64")
parser.add_argument("--id_embedding_dimension", type=int, default=8)
parser.add_argument("--embedding_space", help="Space of embedding", type=int, default=5000000)
parser.add_argument("--learning_rate", help="learning rate", type=float, default=0.1)
args = parser.parse_args()
SEED = 2021
ADD_HOOK = True

class InitHook(tf.train.SessionRunHook):
    def __init__(self):
        self.init_op = []

    def after_create_session(self, session, _):
        session.run(self.init_op)


hook = InitHook()

class BaseModel(object):

    def __init__(self, feature_columns, stage, action):
        """
        :param linear_feature_columns: List of tensorflow feature_column
        :param dnn_feature_columns: List of tensorflow feature_column
        :param stage: String. Including "online_train"/"offline_train"/"evaluate"/"submit"
        :param action: String. Including "read_comment"/"like"/"click_avatar"/"favorite"/"forward"/"comment"/"follow"
        """
        super(BaseModel, self).__init__()
        self.num_epochs_dict = {"read_comment": 5, "like": 5, "click_avatar": 5, "favorite": 5, "forward": 5,
                                "comment": 5, "follow": 5}
        self.estimator = None
        self.feature_columns = feature_columns
        self.stage = stage
        self.action = action
        tf.logging.set_verbosity(tf.logging.INFO)

    def get_estimator(self):
        if self.stage in ["evaluate", "offline_train"]:
            stage = "offline_train"
        else:
            stage = "online_train"
        model_checkpoint_stage_dir = os.path.join(args.model_checkpoint_dir, stage, self.action)
        if not os.path.exists(model_checkpoint_stage_dir):
            # 如果模型目录不存在，则创建该目录
            os.makedirs(model_checkpoint_stage_dir)
        elif self.stage in ["online_train", "offline_train"]:
            # 训练时如果模型目录已存在，则清空目录
            del_file(model_checkpoint_stage_dir)
        hidden_layers = [int(x.strip()) for x in args.hidden_layers.strip().split(",")]
        config = tf.estimator.RunConfig(model_dir=model_checkpoint_stage_dir, tf_random_seed=SEED)
        self.estimator = tf.estimator.DNNClassifier(
            hidden_units=hidden_layers,
            feature_columns=self.feature_columns,
            model_dir=model_checkpoint_stage_dir,
            optimizer = tf.train.AdagradOptimizer(learning_rate=0.08),
            dropout=None,
            input_layer_partitioner=None,
            config=config,
            warm_start_from=None,
            batch_norm=False,)





    def parase_data(self, line):
        line["manual_keyword_list"] = tf.string_split([line["manual_keyword_list"]], ";")
        line["machine_keyword_list"] = tf.string_split([line["machine_keyword_list"]], ";")
        line["manual_tag_list"] = tf.string_split([line["manual_tag_list"]], ";")
        if self.stage != "submit":
            label = line[self.action]
            return line, label
        else:
            return line

    def df_to_dataset(self, df, stage, action, shuffle=True, batch_size=128, num_epochs=1):
        print(df.shape)
        print(df.columns)
        print("batch_size: ", batch_size)
        print("num_epochs: ", num_epochs)
        if stage != "submit":
            #label = df[action]
            ds = tf.data.Dataset.from_tensor_slices((dict(df)))
        else:
            ds = tf.data.Dataset.from_tensor_slices((dict(df)))
        if shuffle:
            ds = ds.shuffle(buffer_size=len(df), seed=SEED)
        ds = ds.map(self.parase_data)
        ds = ds.batch(batch_size)
        if stage in ["online_train", "offline_train"]:
            ds = ds.repeat(num_epochs)
        return ds

    def input_fn_train(self, df, stage, action, num_epochs):
        return self.df_to_dataset(df, stage, action, shuffle=True, batch_size=args.batch_size,
                                  num_epochs=num_epochs)

    def input_fn_predict(self, df, stage, action):
        return self.df_to_dataset(df, stage, action, shuffle=False, batch_size=len(df), num_epochs=1)

    def train(self):
        file_name = "{action}_concat_sample.csv".format(action=self.action)
        stage_dir = os.path.join(args.root_path,"sample", file_name)
        dataframe = pd.read_csv(stage_dir)
        dataframe = dataframe.fillna(0)
        self.estimator.train(
            input_fn=lambda: self.input_fn_train(dataframe, self.stage, self.action, self.num_epochs_dict[self.action]),
        )


    def evaluate(self):
        """
        评估单个行为的uAUC值
        """
        if self.stage in ["online_train", "offline_train"]:
            # 训练集，每个action一个文件
            action = self.action
        else:
            # 测试集，所有action在同一个文件
            action = "all"
        file_name = "{action}_concat_sample.csv".format( action=action)
        evaluate_dir = os.path.join(args.root_path, "sample",file_name)
        df = pd.read_csv(evaluate_dir)
        userid_list = df['userid'].astype(str).tolist()
        predicts = self.estimator.predict(
            input_fn=lambda: self.input_fn_predict(df, self.stage, self.action)
        )
        predicts_df = pd.DataFrame.from_dict(predicts)
        #predicts_df.columns = predicts_df.columns.map(lambda x: x[0] + "_" + x[1])
        logits = predicts_df["logistic"].map(lambda x: x[0])
        labels = df[self.action].values
        uauc = uAUC(labels, logits, userid_list)
        return df[["userid", "feedid"]], logits, uauc

    def predict(self):
        '''
        预测单个行为的发生概率
        '''
        file_name = "{action}_concat_sample.csv".format(action="all")
        submit_dir = os.path.join(args.root_path, "sample",file_name)
        df = pd.read_csv(submit_dir)
        df = df.fillna(0)
        t = time.time()
        predicts = self.estimator.predict(
            input_fn=lambda: self.input_fn_predict(df, self.stage, self.action)
        )
        predicts_df = pd.DataFrame.from_dict(predicts)
        #predicts_df.columns = predicts_df.columns.map(lambda x: x[0] + "_" + x[1])
        logits = predicts_df["logistic"].map(lambda x: x[0])
        # 计算2000条样本平均预测耗时（毫秒）
        ts = (time.time()-t)*1000.0/len(df)*2000.0
        return df[["userid", "feedid"]], logits, ts


def del_file(path):
    '''
    删除path目录下的所有内容
    '''
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path, i)
        if os.path.isdir(c_path):
            del_file(c_path)
        else:
            print("del: ", c_path)
            os.remove(c_path)


def get_feature_columns(voca_keyword_list, voca_tag_list):
    """
    特征的基本处理过程
    """
    feature_columns = []
    def get_id_embedding_column(feature, hash_bucket_size, type, dimension=args.id_embedding_dimension):
        """
        idl类特征转embeddding
        """
        categorical_column = tf.feature_column.categorical_column_with_hash_bucket(feature, hash_bucket_size, type)
        feature_columns.append(tf.feature_column.embedding_column(categorical_column, dimension=dimension))

    def get_numeric_column(feature):
        """
        数值列入模
        """
        feature_columns.append(tf.feature_column.numeric_column(feature))

    def get_numerical_embedding_column(feature, boundaries, dimension=args.id_embedding_dimension):
        """
        数值连续性转embedding
        """
        categorical_column = tf.feature_column.bucketized_column(tf.feature_column.numeric_column(feature), boundaries)
        feature_columns.append(tf.feature_column.embedding_column(categorical_column, dimension=dimension))

    def get_category_onehot_column(feature, vocabulary_list, dtype=tf.string, default_value=-1, num_oov_buckets=3):
        """
        类别特征转换onehot
        """
        categorical_column = tf.feature_column.categorical_column_with_vocabulary_list(feature,
                                                                                       vocabulary_list=vocabulary_list,
                                                                                       dtype=dtype,
                                                                                       default_value=default_value,
                                                                                       num_oov_buckets=num_oov_buckets)
        feature_columns.append(tf.feature_column.indicator_column(categorical_column))

    def get_variable_feature_embedding(feature, dimension=args.id_embedding_dimension):
        """
        变长特征embedding化
        """
        categorical_column = sparse_column_with_keys(feature, keys=voca_keyword_list, combiner="mean",dtype=tf.string)
        feature_columns.append(tf.feature_column.embedding_column(categorical_column, combiner="mean", dimension=dimension))


    get_id_embedding_column("userid", 40000, tf.int64)
    get_id_embedding_column("feedid", 240000, tf.int64)
    get_id_embedding_column("authorid", 40000, tf.int64)
    get_id_embedding_column("bgm_singer_id", 40000, tf.int64)
    get_id_embedding_column("bgm_song_id", 60000, tf.int64)
    get_category_onehot_column("device", [1, 2], dtype=tf.int32)
    get_variable_feature_embedding("manual_keyword_list")
    get_variable_feature_embedding("machine_keyword_list")
    get_variable_feature_embedding("manual_tag_list")
    get_numeric_column("videoplayseconds")
    # get_numerical_embedding_column("read_comment_user_sum",
    #                                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 11.0, 15.0, 21.0, 30.0, 46.0, 79.0])
    # get_numerical_embedding_column("like_user_sum",
    #                                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 11.0, 13.0, 17.0, 23.0, 37.0])
    # get_numerical_embedding_column("click_avatar_user_sum", [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 11.0])
    # get_numerical_embedding_column("forward_user_sum", [0.0, 1.0, 2.0, 3.0, 4.0, 7.0, 10.0])
    # get_numerical_embedding_column("comment_user_sum", [0.0, 1.0, 2.0], dimension=4)
    # get_numerical_embedding_column("follow_user_sum", [0.0, 1.0, 2.0, 3.0], dimension=5)
    # get_numerical_embedding_column("favorite_user_sum", [0.0, 1.0, 2.0, 3.0, 6.0], dimension=6)
    #
    # get_numerical_embedding_column("read_comment_feed_sum",
    #                                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 13.0, 18.0, 30.0])
    # get_numerical_embedding_column("like_feed_sum", [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 14.0, 22.0])
    # get_numerical_embedding_column("click_avatar_feed_sum", [0.0, 1.0, 2.0, 3.0, 4.0, 7.0])
    # get_numerical_embedding_column("forward_feed_sum", [0.0, 1.0, 2.0, 4.0], dimension=5)
    # get_numerical_embedding_column("comment_feed_sum", [0.0, 1.0], dimension=3)
    # get_numerical_embedding_column("follow_feed_sum", [0.0, 1.0], dimension=3)
    # get_numerical_embedding_column("favorite_feed_sum", [0.0, 1.0, 2.0], dimension=4)

    # get_numeric_column("read_comment_user_sum")
    # get_numeric_column("like_user_sum")
    # get_numeric_column("click_avatar_user_sum")
    # get_numeric_column("forward_user_sum")
    # get_numeric_column("comment_user_sum")
    # get_numeric_column("follow_user_sum")
    # get_numeric_column("favorite_user_sum")
    #
    # get_numeric_column("read_comment_feed_sum")
    # get_numeric_column("like_feed_sum")
    # get_numeric_column("click_avatar_feed_sum")
    # get_numeric_column("forward_feed_sum")
    # get_numeric_column("comment_feed_sum")
    # get_numeric_column("follow_feed_sum")
    # get_numeric_column("favorite_feed_sum")
    #
    # get_numeric_column("videoplayseconds")
    return feature_columns


def main():
    t = time.time()
    from itertools import chain
    ds = pd.read_csv("E:\WeChartCompetition\code\data\wechat_algo_data1\\feed_info.csv")
    voca_keyword_list = list(set(chain.from_iterable(map(lambda item: str(item).split(";"), ds.manual_keyword_list.values.tolist()))))
    voca_keyword_list += list(set(chain.from_iterable(map(lambda item: str(item).split(";"), ds.machine_keyword_list.values.tolist()))))

    voca_tag_list = list(set(chain.from_iterable(map(lambda item: str(item).split(";"), ds.manual_tag_list.values.tolist()))))
    feature_columns = get_feature_columns(list(set(voca_keyword_list))+["99999"], voca_tag_list+["99999"])
    stage = "submit"
    print('Stage: %s' % stage)
    eval_dict = {}

    predict_dict = {}
    predict_time_cost = {}
    ids = None
    for action in ACTION_LIST:
        print("Action:", action)
        model = BaseModel(feature_columns, stage, action)
        model.get_estimator()

        if stage in ["online_train", "offline_train"]:
            # 训练 并评估
            model.train()
            ids, logits, action_uauc = model.evaluate()
            eval_dict[action] = action_uauc

        if stage == "evaluate":
            # 评估线下测试集结果，计算单个行为的uAUC值，并保存预测结果
            ids, logits, action_uauc = model.evaluate()
            eval_dict[action] = action_uauc
            predict_dict[action] = logits

        if stage == "submit":
            # 预测线上测试集结果，保存预测结果
            ids, logits, ts = model.predict()
            predict_time_cost[action] = ts
            predict_dict[action] = logits
    if stage in ["evaluate", "offline_train", "online_train"]:
        # 计算所有行为的加权uAUC
        print(eval_dict)
        weight_dict = {"read_comment": 4, "like": 3, "click_avatar": 2, "favorite": 1, "forward": 1,
                       "comment": 1, "follow": 1}
        weight_auc = compute_weighted_score(eval_dict, weight_dict)
        print("Weighted uAUC: ", weight_auc)


    if stage in ["evaluate", "submit"]:
        # 保存所有行为的预测结果，生成submit文件
        actions = pd.DataFrame.from_dict(predict_dict)
        print("Actions:", actions)
        ids[["userid", "feedid"]] = ids[["userid", "feedid"]].astype(int)
        res = pd.concat([ids, actions], axis=1)
        # 写文件
        file_name = "submit_" + str(int(time.time())) + ".csv"
        submit_file = os.path.join(args.root_path, stage, file_name)
        print('Save to: %s'%submit_file)
        res.to_csv(submit_file, index=False)

    if stage == "submit":
        print('不同目标行为2000条样本平均预测耗时（毫秒）：')
        print(predict_time_cost)
        print('单个目标行为2000条样本平均预测耗时（毫秒）：')
        print(np.mean([v for v in predict_time_cost.values()]))
    print('Time cost: %.2f s'%(time.time()-t))


if __name__ == '__main__':
    main()
