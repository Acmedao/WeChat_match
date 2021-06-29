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
parser.add_argument("--hidden_layers", type=str, default="256,128")
parser.add_argument("--id_embedding_dimension", type=int, default=10)
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
        self.num_epochs_dict = {"read_comment": 2, "like": 2, "click_avatar": 2, "favorite": 2, "forward": 2,
                                "comment": 2, "follow": 2}
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
        self.estimator = tf.estimator.Estimator(
            model_fn=model_fn_deepfm,
            config=config,
            params={
                'feature_columns': self.feature_columns,
                'hidden_units': hidden_layers,
                'pretrain_step': 4000
            }
        )

    def parase_data(self, line):
        line["manual_keyword_list"] = tf.string_split([line["manual_keyword_list"]], ";")
        line["machine_keyword_list"] = tf.string_split([line["machine_keyword_list"]], ";")
        line["manual_tag_list"] = tf.string_split([line["manual_tag_list"]], ";")
        if self.stage != "submit":
            label = line[self.action]
            return line, {'fm': label>0, 'deep': label>0}
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
        dataframe = pd.read_csv(stage_dir, nrows =500)
        dataframe = dataframe.fillna(0)
        self.estimator.train(
            input_fn=lambda: self.input_fn_train(dataframe, self.stage, self.action, self.num_epochs_dict[self.action]),
            hooks=[] if ADD_HOOK == False else [hook]
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
        df = pd.read_csv(evaluate_dir, nrows =10)
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


def model_fn_deepfm(features, labels, mode, params):
    my_head = tf.contrib.estimator.binary_classification_head(name="deep")
    # my_head = tf.contrib.estimator.multi_head([
    #     tf.contrib.estimator.binary_classification_head(name="deep"),
    #     tf.contrib.estimator.binary_classification_head(name="fm")
    # ])bushu..
    with tf.variable_scope("fm"):
        user_embedding = tf.feature_column.input_layer(features, params["feature_columns"].pop("userid"))
        feed_embedding = tf.feature_column.input_layer(features, params["feature_columns"].pop("feedid"))
        author_embedding = tf.feature_column.input_layer(features, params["feature_columns"].pop("authorid"))
        bgm_singer_embedding = tf.feature_column.input_layer(features, params["feature_columns"].pop("bgm_singer_id"))
        bgm_song_embedding = tf.feature_column.input_layer(features, params["feature_columns"].pop("bgm_song_id"))

        fm_cross = tf.reduce_sum(tf.add_n([tf.multiply(user_embedding, feed_embedding),
                                           tf.multiply(user_embedding, author_embedding),
                                           tf.multiply(user_embedding, bgm_singer_embedding),
                                           tf.multiply(user_embedding, bgm_song_embedding),
                                           tf.multiply(feed_embedding, author_embedding),
                                           tf.multiply(feed_embedding, bgm_singer_embedding),
                                           tf.multiply(feed_embedding, bgm_song_embedding),
                                           tf.multiply(author_embedding, bgm_singer_embedding),
                                           tf.multiply(author_embedding, bgm_song_embedding),
                                           tf.multiply(bgm_singer_embedding, bgm_song_embedding)
                                           ]), axis=1, keepdims=True)
        user_embedding_snapshot = tf.stop_gradient(user_embedding)
        feed_embedding_snapshot = tf.stop_gradient(feed_embedding)
        author_embedding_snapshot = tf.stop_gradient(author_embedding)
        bgm_singer_embedding_snapshot = tf.stop_gradient(bgm_singer_embedding)
        bgm_song_embedding_snapshot = tf.stop_gradient(bgm_song_embedding)
        fm_bias = tf.get_variable(name="fm_bias", shape=[1], initializer=tf.constant_initializer(0.0))

    with tf.variable_scope("deep"):
        deep_embedding = []
        features_key = list(params['feature_columns'].keys())
        features_key.sort()
        for feature in features_key:
            deep_embedding.append(tf.feature_column.input_layer(features, params['feature_columns'][feature]))
        deep_embedding += [user_embedding_snapshot,
                           feed_embedding_snapshot,
                           author_embedding_snapshot,
                           bgm_singer_embedding_snapshot,
                           bgm_song_embedding_snapshot
                           ]
        net = tf.concat(deep_embedding, axis=1, name="deep_input")
        tf.logging.info("deep input shape={}".format(net.shape))
        tf.summary.histogram('input', net)

        layer_name = "dense"
        for units in params['hidden_units']:
            net = tf.layers.dense(net, units=units, activation=tf.nn.relu)
            with tf.variable_scope(layer_name, reuse=True):
                weights = tf.get_variable("kernel")
                bias = tf.get_variable("bias")
            tf.summary.histogram('{}_kernel'.format(units), weights)
            tf.summary.histogram('{}_bias'.format(units), bias)
            tf.summary.histogram('{}'.format(units), net)
            layer_name = "dense_1"
        dense_logits = tf.layers.dense(net, units=my_head.logits_dimension-1, activation=None, name="dense_logit")
        target_bias = tf.get_variable(name="global_bias", shape=[my_head.logits_dimension-1],
                                      initializer=tf.constant_initializer(0.0))
        global_bias = tf.concat([target_bias, fm_bias], axis=0)
        logits = tf.cond(
            tf.train.get_global_step() > params['pretrain_step'],
            lambda: tf.nn.bias_add(dense_logits + fm_cross, global_bias),
            lambda: tf.nn.bias_add(fm_cross, global_bias)
        )
    fm_optimizer = tf.train.FtrlOptimizer(learning_rate=0.1, learning_rate_power=-0.99)
    deep_optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)

    def _train_on_fn(loss):
        train_ops = []
        train_ops.append(deep_optimizer.minimize(loss=loss,
                                                 global_step=tf.train.get_global_step(),
                                                 var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                                            scope="deep")))
        train_ops.append(fm_optimizer.minimize(loss=loss,
                                               var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                                          scope="fm")))
        return control_flow_ops.group(*train_ops)

    return my_head.create_estimator_spec(
        features=features,
        mode=mode,
        labels=labels,
        logits=logits,
        train_op_fn=_train_on_fn
    )


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
    feature_columns = {}
    def get_id_embedding_column(feature, hash_bucket_size, type, dimension=args.id_embedding_dimension):
        """
        idl类特征转embeddding
        """
        categorical_column = tf.feature_column.categorical_column_with_hash_bucket(feature, hash_bucket_size, type)
        feature_columns[feature] = tf.feature_column.embedding_column(categorical_column, dimension=dimension)

    def get_numeric_column(feature):
        """
        数值列入模
        """
        feature_columns[feature]=tf.feature_column.numeric_column(feature)

    def get_numerical_embedding_column(feature, boundaries, dimension=args.id_embedding_dimension):
        """
        数值连续性转embedding
        """
        categorical_column = tf.feature_column.bucketized_column(tf.feature_column.numeric_column(feature), boundaries)
        feature_columns[feature] = tf.feature_column.embedding_column(categorical_column, dimension=dimension)

    def get_category_onehot_column(feature, vocabulary_list, dtype=tf.string, default_value=-1, num_oov_buckets=3):
        """
        类别特征转换onehot
        """
        categorical_column = tf.feature_column.categorical_column_with_vocabulary_list(feature,
                                                                                       vocabulary_list=vocabulary_list,
                                                                                       dtype=dtype,
                                                                                       default_value=default_value,
                                                                                       num_oov_buckets=num_oov_buckets)
        feature_columns[feature] = tf.feature_column.indicator_column(categorical_column)

    def get_variable_feature_embedding(feature, dimension=args.id_embedding_dimension):
        """
        变长特征embedding化
        """
        categorical_column = sparse_column_with_keys(feature, keys=voca_keyword_list, combiner="mean",dtype=tf.string)
        feature_columns[feature] = tf.feature_column.embedding_column(categorical_column, combiner="mean", dimension=dimension)


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





    get_numerical_embedding_column("videoplayseconds", [2.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0,16.0,17.0,18.0,20.0,21.0,23.0,25.0,27.0,29.0,32.0,35.0,38.0,42.0,46.0,50.0,54.0,58.0,59.0,60.0])

    get_numerical_embedding_column("read_comment_sum_feed",[0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0,16.0,17.0,18.0,19.0,20.0,21.0,22.0,24.0,25.0,26.0,28.0,30.0,32.0,35.0,38.0,42.0,46.0,52.0,61.0,74.0,100.48800000001211])
    get_numerical_embedding_column("like_sum_feed",[0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0,16.0,17.0,18.0,19.0,21.0,22.0,23.0,25.0,27.0,29.0,32.0,36.0,41.0,48.0,63.0])
    get_numerical_embedding_column("click_avatar_sum_feed", [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 11.0, 12.0, 15.0, 19.0, 27.0])
    get_numerical_embedding_column("forward_sum_feed", [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 9.0, 11.0, 15.0])
    get_numerical_embedding_column("comment_sum_feed", [0.0, 1.0, 2.0, 3.0, 3.8585599999642, 4.0, 5.0, 6.0, 7.0, 8.0])
    get_numerical_embedding_column("follow_sum_feed", [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.404879999929108])
    get_numerical_embedding_column("favorite_sum_feed", [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 17.0])

    get_numerical_embedding_column("read_comment_sum_user", [0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0,16.0,17.0,19.0,21.0,23.0,25.0,27.0,30.0,33.0,37.0,41.0,46.0,52.0,59.0,68.0,79.0,94.0,117.41000000000349,157.0])
    get_numerical_embedding_column("like_sum_user", [0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0,16.0,17.0,18.0,20.0,21.0,23.0,26.0,29.0,33.0,37.0,45.0,58.0,85.0])
    get_numerical_embedding_column("click_avatar_sum_user", [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 13.0, 15.0, 19.0])
    get_numerical_embedding_column("forward_sum_user", [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 15.0])
    get_numerical_embedding_column("comment_sum_user", [0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 9.0])
    get_numerical_embedding_column("follow_sum_user", [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 10.64100000000326])
    get_numerical_embedding_column("favorite_sum_user", [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 15.0, 19.0, 31.0])

    get_numerical_embedding_column("playsum", [0.0,33.1,116.5,208.83,301.53,398.12,497.0,604.04,710.02,822.34,935.91,1053.01,1178.93,1306.28,1438.48,1585.81,1724.42,1868.48,2020.91,2179.83,2343.03,2509.59,2691.76,2887.16,3087.4,3293.49,3513.41,3732.51,3960.5,4204.96,4465.4,4741.89,5044.29,5344.71,5679.44,6016.45,6359.25,6756.86,7156.73,7593.07,8062.1,8564.95,9120.29,9758.45,10485.29,11335.32,12365.69,13767.06,15684.09,19110.6])
    get_numerical_embedding_column("playcount", [1.0,6.0,13.0,19.0,25.0,30.0,35.0,40.0,46.0,50.0,55.0,60.0,65.0,70.0,76.0,81.0,87.0,93.0,99.0,106.0,112.0,119.0,126.0,133.0,140.0,147.0,155.0,164.0,172.0,181.0,190.0,199.0,209.0,219.0,230.0,242.0,253.0,265.0,278.0,291.0,306.0,322.0,338.0,356.0,376.0,398.0,423.0,453.0,492.0,554.0])
    get_numerical_embedding_column("playmean", [0.0,1.26,2.94,4.42,5.71,6.87,8.01,9.04,10.09,11.08,12.01,12.88,13.74,14.64,15.48,16.3,17.12,17.93,18.73,19.5,20.28,21.07,21.83,22.58,23.32,24.11,24.92,25.73,26.57,27.42,28.31,29.2,30.07,30.99,31.96,32.95,34.04,35.13,36.26,37.5,38.89,40.41,42.08,43.87,45.97,48.54,51.52,55.33,60.6,70.85])
    get_numerical_embedding_column("staysum", [0.0,75.4,209.64,337.6,470.97,606.39,735.0,871.17,1008.3,1145.85,1287.46,1436.05,1585.83,1744.05,1891.88,2047.55,2212.23,2387.88,2560.4,2740.0,2931.72,3139.76,3348.4,3570.82,3802.13,4024.45,4268.56,4525.57,4779.15,5057.24,5363.05,5679.59,5997.14,6325.79,6672.82,7052.34,7439.42,7860.54,8312.18,8808.74,9312.55,9878.99,10512.33,11175.81,11989.86,12908.56,14060.54,15570.06,17664.15,21442.86])
    get_numerical_embedding_column("staymean", [0.0,4.51,6.63,8.17,9.57,10.81,12.03,13.09,14.12,15.09,16.06,16.99,17.93,18.8,19.63,20.48,21.29,22.13,22.95,23.76,24.55,25.29,26.11,26.94,27.75,28.56,29.41,30.26,31.09,31.94,32.84,33.71,34.65,35.66,36.66,37.69,38.77,39.9,41.12,42.46,43.88,45.4,47.1,49.08,51.18,53.76,56.89,61.08,67.28,79.09])
    get_numerical_embedding_column("feedid_count", [1.0,6.0,13.0,19.0,25.0,30.0,35.0,40.0,46.0,50.0,55.0,60.0,65.0,70.0,76.0,81.0,87.0,93.0,99.0,106.0,112.0,119.0,126.0,133.0,140.0,147.0,155.0,164.0,172.0,181.0,190.0,199.0,209.0,219.0,230.0,242.0,253.0,265.0,278.0,291.0,306.0,322.0,338.0,356.0,376.0,398.0,423.0,453.0,492.0,554.0])
    get_numerical_embedding_column("song_count", [1.0,3.0,7.0,9.0,12.0,15.0,17.0,19.0,22.0,24.0,26.0,29.0,31.0,34.0,36.0,39.0,41.0,44.0,47.0,50.0,53.0,56.0,59.0,63.0,66.0,69.0,73.0,77.0,81.0,85.0,89.0,93.0,98.0,103.0,108.0,113.0,118.0,124.0,130.0,136.0,142.0,150.0,157.0,165.0,175.0,185.0,197.0,211.0,230.0,259.0])
    get_numerical_embedding_column("singer", [1.0,3.0,7.0,9.0,12.0,14.0,17.0,19.0,22.0,24.0,26.0,28.0,31.0,33.0,35.0,38.0,40.0,43.0,46.0,49.0,52.0,55.0,58.0,61.0,64.0,67.0,71.0,74.0,78.0,82.0,86.0,90.0,94.0,99.0,104.0,108.0,113.0,118.0,124.0,130.0,136.0,142.0,150.0,157.0,165.0,175.0,186.0,199.0,216.0,242.0])
    get_numerical_embedding_column("replay_num", [0.0,2.0,3.0,5.0,7.0,8.0,10.0,12.0,14.0,16.0,18.0,20.0,23.0,25.0,27.0,30.0,32.0,35.0,38.0,41.0,44.0,47.0,50.0,54.0,57.0,61.0,65.0,69.0,73.0,78.0,83.0,88.0,93.0,99.0,104.0,110.0,117.0,124.0,131.0,140.0,149.0,158.0,169.0,182.0,197.0,213.0,234.0,262.0,306.0])
    get_numerical_embedding_column("stay_not_play", [0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0,17.0,18.0,20.0,22.0,24.0,27.0,29.0,32.0,36.0,41.0,46.0,52.0,60.0,70.0,86.0,111.0,171.0])
    get_numerical_embedding_column("play_stay_rate_sum",[0.0,2.32,5.74,9.46,13.37,17.11,20.82,24.71,28.4,32.22,36.18,40.14,44.1,48.13,52.14,56.49,61.04,65.78,70.79,75.59,80.73,86.34,92.07,97.99,103.85,110.2,116.66,123.4,130.14,137.4,145.13,153.05,161.04,169.69,178.5,187.78,197.24,207.64,218.28,230.29,242.61,256.11,270.35,286.06,302.89,322.66,345.7,373.27,409.18,465.36])
    get_numerical_embedding_column("wanbo_num", [0.0,1.0,2.0,4.0,6.0,8.0,10.0,12.0,14.0,16.0,19.0,21.0,23.0,26.0,29.0,31.0,34.0,37.0,40.0,43.0,46.0,49.0,53.0,56.0,60.0,64.0,68.0,73.0,78.0,82.0,87.0,93.0,98.0,104.0,110.0,116.0,123.0,130.0,137.0,146.0,154.0,164.0,175.0,187.0,200.0,216.0,234.0,256.0,285.0,332.0])
    get_numerical_embedding_column("wanbo_rate", [0.0,0.01,0.05,0.08,0.12,0.15,0.18,0.21,0.24,0.26,0.28,0.3,0.32,0.33,0.35,0.37,0.38,0.4,0.41,0.43,0.44,0.45,0.47,0.48,0.49,0.5,0.52,0.53,0.54,0.55,0.56,0.57,0.58,0.6,0.61,0.62,0.63,0.64,0.65,0.67,0.68,0.69,0.71,0.72,0.74,0.75,0.78,0.8,0.82,0.86])
    get_numerical_embedding_column("play_rate", [0.0,1.13,3.39,5.76,8.31,10.9,13.49,16.12,18.86,21.62,24.33,27.2,30.07,32.97,36.09,39.22,42.5,45.87,49.51,53.18,56.92,60.86,65.17,69.27,73.62,78.31,83.25,88.28,93.7,99.13,104.94,110.91,117.25,123.64,130.43,137.55,144.62,152.53,161.2,170.18,179.95,190.79,202.72,215.67,229.98,247.08,266.11,289.34,320.4,368.25])
    get_numerical_embedding_column("play_stay_rate_mean", [0.0,0.23,0.53,0.62,0.67,0.71,0.74,0.76,0.78,0.8,0.82,0.84,0.85,0.86,0.88,0.89,0.9,0.91,0.93,0.94])

    return feature_columns


def main():
    t = time.time()
    from itertools import chain
    ds = pd.read_csv("E:\WeChartCompetition\code\data\wechat_algo_data1\\feed_info.csv")
    voca_keyword_list = list(set(chain.from_iterable(map(lambda item: str(item).split(";"), ds.manual_keyword_list.values.tolist()))))
    voca_keyword_list += list(set(chain.from_iterable(map(lambda item: str(item).split(";"), ds.machine_keyword_list.values.tolist()))))

    voca_tag_list = list(set(chain.from_iterable(map(lambda item: str(item).split(";"), ds.manual_tag_list.values.tolist()))))
    feature_columns = get_feature_columns(list(set(voca_keyword_list))+["99999"], voca_tag_list+["99999"])
    stage = "online_train"
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
