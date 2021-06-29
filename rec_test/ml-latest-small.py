import os

import pandas as pd
import numpy as np

# 数据集下载 https://link.zhihu.com/?target=http%3A//files.grouplens.org/datasets/movielens/ml-latest-small.zip
DATA_PATH = './datasets/ml-latest-small/ratings.csv'
CACHE_DIR = './datasets/cache/'


def load_data(data_path):
    cache_path = os.path.join(CACHE_DIR, "ratings_matrix.pkl")

    if os.path.exists(cache_path):
        ratings_matrix = pd.read_pickle(cache_path)
    else:
        dtype = {"userId": np.int32, "movieId": np.int32, "rating": np.float32}
        ratings = pd.read_csv(data_path, dtype=dtype, usecols=range(3))
        ratings_matrix = ratings.pivot_table(index=["userId"], columns=["movieId"], values="rating")
        ratings_matrix.to_pickle(cache_path)
    return ratings_matrix


# 计算物品相似度
def compute_pearson_similarity(ratings_matrix, based="user"):
    user_similarity_cache_path = os.path.join(CACHE_DIR, "user_similarity.pkl")
    item_similarity_cache_path = os.path.join(CACHE_DIR, "item_similarity.pkl")

    # 基于皮尔逊相关系数
    if based == "user":
        if os.path.exists(user_similarity_cache_path):
            similarity = pd.read_pickle(user_similarity_cache_path)
        else:
            similarity = ratings_matrix.T.corr()
            similarity.to_pickle(user_similarity_cache_path)

    elif based == "item":
        if os.path.exists(item_similarity_cache_path):
            similarity = pd.read_pickle(item_similarity_cache_path)
        else:
            similarity = ratings_matrix.T.corr()
            similarity.to_pickle(item_similarity_cache_path)

    else:
        raise Exception("Unhandled 'based' Value: %s" % based)

    return similarity


# 实现评分预测方法
def predict(uid, iid, ratings_matrix, user_similar):
    # 1. 找出uid用户的相似用户
    similar_users = user_similar[uid].drop([uid]).dropna()
    # 相似用户筛选规则：正相关用户
    similar_users = similar_users.where(similar_users > 0).dropna()
    if similar_users.empty is True:
        raise Exception("用户<%d>没有相似用户" % uid)

    # 2. 从uid用户的邻近相似用户中筛选出对iid物品有评分记录的邻近用户
    ids = set(ratings_matrix[iid].dropna().index) & set(similar_users.index)
    print(list, similar_users)
    finally_similar_users = similar_users.iloc[list(ids)]

    # 3. 结合uid用户与其邻近用户的相似度预测uid用户对iid物品的评分
    sum_up = 0  # 评分预测公式的分子部分的值
    sum_down = 0  # 评分预测公式的分母部分的值
    for sim_uid, similarity in finally_similar_users.iteritems():
        # 邻近用户评分
        sim_user_rated_movies = ratings_matrix.iloc[sim_uid].dropna()
        # 邻近用户对iid物品的评分
        sim_user_rating_for_item = sim_user_rated_movies[iid]
        # 计算分子的值
        sum_up += similarity * sim_user_rating_for_item
        # 计算分母的值
        sum_down += similarity

    # 计算预测的评分值并返回
    predict_rating = sum_up / sum_down
    print("预测用户<%d>对电影<%d>的评分：%0.2f" % (uid, iid, predict_rating))
    return round(predict_rating, 2)


# 预测全部评分
def predict_all(uid, ratings_matrix, user_similar):
    # 准备预测的物品的id列表
    item_ids = ratings_matrix.columns
    # 逐个预测
    for iid in item_ids:
        try:
            rating = predict(uid, iid, ratings_matrix, user_similar)
        except Exception as e:
            print(e)
        else:
            yield uid, iid, rating


if __name__ == '__main__':
    ratings_matrix = load_data(DATA_PATH)
    # print(ratings_matrix)

    user_similar = compute_pearson_similarity(ratings_matrix, based="user")
    # print(user_similar)
    item_similar = compute_pearson_similarity(ratings_matrix, based="item")
    # print(item_similar)

    predict(1, 1, ratings_matrix, user_similar)

    predict(1, 2, ratings_matrix, user_similar)
