
import tensorflow as tf
tf.enable_eager_execution()
import pandas as pd


sparse_embeddings = []
y={"xx":[0,1,2,3,4,5,6,7,8,9,55]}
#特征列
# categorical_column1 = tf.feature_column.categorical_column_with_vocabulary_list("feature",vocabulary_list=[1,2],dtype=tf.int64)
# #x = tf.feature_column.indicator_column(categorical_column)
# indicator1 = tf.feature_column.indicator_column(categorical_column1)
# sparse_embeddings.append(tf.feature_column.input_layer(y, [indicator1]))

# categorical_column = tf.feature_column.bucketized_column(tf.feature_column.numeric_column("xx"), [0.0, 1.0, 2.0])
# x = tf.feature_column.embedding_column(categorical_column, dimension=8)
# z = tf.feature_column.input_layer(y, [x])
# print(z)


def df_to_dataset(df, stage, action, shuffle=True, batch_size=128, num_epochs=1):
    print(df.shape)
    print(df.columns)
    print("batch_size: ", batch_size)
    print("num_epochs: ", num_epochs)
    if stage != "submit":
        label = df[action]>0
        ds = tf.data.Dataset.from_tensor_slices((dict(df), label))
    else:
        ds = tf.data.Dataset.from_tensor_slices((dict(df)))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df), seed=2010)
    ds = ds.batch(batch_size)
    if stage in ["online_train", "offline_train"]:
        ds = ds.repeat(num_epochs)
    return ds
# ds = pd.read_csv("E:\WeChartCompetition\code\data\online_train\online_train_click_avatar_14_concate_sample.csv")
# df_to_dataset(ds, stage="online_train", action="click_avatar")

# for i in range(1,10):
#     print(i)
#     v2_nor = tf.get_variable('v2_nor', shape=[1, 4],
#                              initializer=tf.random_normal_initializer(mean=0, stddev=5, seed=0))  # 均值、方差、种子值
#     print(v2_nor)
# tf.enable_eager_execution()
# CATE_SET=['未知分类','鞋子','手机','衣服','袜子','笔记本']
# cate_table = tf.contrib.lookup.index_table_from_tensor(
#     mapping=CATE_SET, num_oov_buckets=2, default_value=-1
# )
# user_A_cate_browse=['鞋子',"鞋子"]
#
# #此处得到的是一个sparse_tensor
# user_A_cates = tf.string_split(user_A_cate_browse, sep=",")
# user_A_cates_sparse = tf.SparseTensor(
#     indices=user_A_cates.indices,
#     values=cate_table.lookup(user_A_cates.values),
#     dense_shape=user_A_cates.dense_shape)
#
# weight_A_cates_sparse = tf.SparseTensor(
#     indices=user_A_cates.indices,
#     values=[0.2,0.5],
#     dense_shape=user_A_cates.dense_shape)
# embedding_params = tf.Variable(tf.truncated_normal([6, 4]))
# user_A_cates_emb = tf.nn.embedding_lookup_sparse(embedding_params, sp_ids=user_A_cates_sparse, sp_weights=weight_A_cates_sparse, combiner="sum")
# #第一个参数，用于查表的embedding矩阵
# #一个sparse tensor，用于记录从param抽取数据的位置
# print(embedding_params)
# print(user_A_cates_emb)
def _parase_data(line):
    # print(line)
    # columns = tf.string_split([line], ',')
    feature_columns = {}
    # feature_columns['feedid'] = columns.values[0]
    feature_columns['manual_keyword_list']  = tf.string_split([line["manual_keyword_list"]],";")
    label = line["feedid"]
    return feature_columns,label



# data = tf.data.TextLineDataset("E:\WeChartCompetition\code\data\wechat_algo_data1\\feed_info.csv").skip(1)
# data = data.map(_parase_data)
# data = data.batch(10)
import numpy as np
from tensorflow.contrib.layers import sparse_column_with_integerized_feature,sparse_column_with_vocabulary_file, sparse_column_with_keys

# dtypes = {
# 'feedid':string,
# 'authorid':np.string,
#          'videoplayseconds':np.string,
#          'description':np.string,
#          'ocr':np.string,
#          'asr':np.string,
#        'bgm_song_id':np.string,
#          'bgm_singer_id':np.string,
#          'manual_keyword_list':np.string,
#        'machine_keyword_list':np.string,
#          'manual_tag_list':np.string,
#          'machine_tag_list':np.string,
#        'description_char':np.string,
#          'ocr_char':np.string,
#          'asr_char':np.string
# }
ds = pd.read_csv("E:\WeChartCompetition\code\data\online_train\online_train_click_avatar_14_concate_sample.csv")
from itertools import chain
voca_list = list(set(chain.from_iterable(map(lambda item: str(item).split(";"),ds.manual_keyword_list.values.tolist()))))
voca_list += list(set(chain.from_iterable(map(lambda item: str(item).split(";"),ds.machine_keyword_list.values.tolist()))))
# cate_table = tf.contrib.lookup.index_table_from_tensor(
#     mapping=list(set(voca_list)), num_oov_buckets=2, default_value=-1
# )
# embedding_params = tf.Variable(tf.truncated_normal([len(voca_list), 8]))
ds = ds.fillna("9999-")
ds = tf.data.Dataset.from_tensor_slices((dict(ds[["feedid","manual_keyword_list"]])))
data = ds.map(_parase_data)
data = data.batch(1)
# user_A_cates_emb = tf.nn.embedding_lookup_sparse(embedding_params, sp_ids=data.manual_keyword_list,  combiner="sum")
# y = {"manual_keyword_list":[13785,16002,211]}
#categorical_column1= tf.feature_column.categorical_column_with_vocabulary_list("manual_keyword_list", list(set(voca_list)))

categorical_column1 = sparse_column_with_keys("manual_keyword_list", keys=list(set(voca_list)), combiner="mean", dtype=tf.string)
x = tf.feature_column.embedding_column(categorical_column1, dimension=8)
z = tf.feature_column.input_layer(data, x)
tf.nn.embedding_lookup
print(z)

















