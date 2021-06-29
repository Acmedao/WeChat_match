# ！/usr/bin/env python
# -*- coding: utf-8 -*-#
# Created by lzy on 2021/5/29 10:05
# @File    :feature_select.py


class Feature:
    """
    对特征的基本处理可在此实现
    """

    def __init__(self, feature):
        self.feature_name = feature


# label
read_comment = Feature("read_comment")
comment = Feature("comment")
like = Feature("like")
click_avatar = Feature("click_avatar")
forward = Feature("forward")
follow = Feature("follow")
favorite = Feature("favorite")

# id
userid = Feature("userid")
feedid = Feature("feedid")
device = Feature("device")
authorid = Feature("authorid")
bgm_singer_id = Feature("bgm_singer_id")
bgm_song_id = Feature("bgm_song_id")

# statistic
# 用户七日内各类行为和
read_comment_user_sum = Feature("read_comment_user_sum")
comment_user_sum = Feature("comment_user_sum")
like_user_sum = Feature("like_user_sum")
click_avatar_user_sum = Feature("click_avatar_user_sum")
forward_user_sum = Feature("forward_user_sum")
follow_user_sum = Feature("follow_user_sum")
favorite_user_sum = Feature("favorite_user_sum")
look_feed_num_7sum = Feature("look_feed_sum")
play_feed_time_7sum = Feature("play_feed_time_7sum")
stay_feed_time_7sum = Feature("stay_feed_time_7sum")
wanbo_num_7sum = Feature("wanbo_num_7sum")

# 视频七日内各类行为和
videoplayseconds = Feature("videoplayseconds")
read_comment_feed_sum = Feature("read_comment_feed_sum")
comment_feed_sum = Feature("comment_feed_sum")
like_feed_sum = Feature("like_feed_sum")
click_avatar_feed_sum = Feature("click_avatar_feed_sum")
forward_feed_sum = Feature("forward_feed_sum")
follow_feed_sum = Feature("follow_feed_sum")
favorite_feed_sum = Feature("favorite_feed_sum")

base_feature_set = [
    read_comment,
    like,
    click_avatar,
    forward,
    userid,
    feedid,
    device,
    authorid,
    bgm_singer_id,
    bgm_song_id,
    read_comment_user_sum,
    comment_user_sum,
    like_user_sum,
    click_avatar_user_sum,
    forward_user_sum,
    follow_user_sum,
    favorite_user_sum,
    # look_feed_num_7sum,
    # play_feed_time_7sum,
    # stay_feed_time_7sum,
    # wanbo_num_7sum,
    videoplayseconds,
    read_comment_feed_sum,
    comment_feed_sum,
    like_feed_sum,
    click_avatar_feed_sum,
    forward_feed_sum,
    follow_feed_sum,
    favorite_feed_sum,
]
