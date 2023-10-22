import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.sparse.linalg import svds
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from .setting import category_dict


class Recommender:
    def __init__(self) -> None:
        pass

    def test(self):
        print("hello")

    def calculate_rmse(self, list1, list2):
        if len(list1) != len(list2):
            raise ValueError("The two lists must have the same length.")

        squared_differences = [(x - y) ** 2 for x, y in zip(list1, list2)]
        mean_squared_difference = sum(squared_differences) / len(list1)
        rmse = math.sqrt(mean_squared_difference)

        return rmse

    def create_full_matrix_svd(self):
        metrics = ["views", "likes", "comments"]

        base_path = "files/train-test-matrix"

        categorys = list(category_dict.keys())

        # make full matrix with train data set
        for metric in metrics:
            for category in tqdm(categorys):
                data = pd.read_csv(base_path + f"/{category}_{metric}_train.csv", index_col=0)
                channelid = data["channelid"]
                data.drop("channelid", axis=1, inplace=True)
                data = data.T
                matrix = data.values
                # metric_mean은 keyword의 평균 조회수
                metric_mean = np.mean(matrix, axis=1)

                # R_user_mean : 사용자-영화에 대해 사용자 평균 평점을 뺀 것.
                matrix_user_mean = matrix - metric_mean.reshape(-1, 1)

                U, sigma, Vt = svds(matrix_user_mean, k=12)
                sigma = np.diag(sigma)
                svd_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + metric_mean.reshape(-1, 1)
                df_svd_preds = pd.DataFrame(svd_user_predicted_ratings, columns=data.columns, index=data.index)
                # df_svd_preds.head()
                df_svd_preds = df_svd_preds.T
                df_svd_preds["channedid"] = channelid

                df_svd_preds.to_csv(f"files/full_matrix/svd/{category}_{metric}_fullmatrix.csv")

    def create_full_matrix_coll_inf_based(self):
        metrics = ["views", "likes", "comments"]

        base_path = "files/train-test-matrix"

        categorys = list(category_dict.keys())

    def create_full_matrix_coll_keyword_based(self):
        return None

    def create_full_matrix_deep(self):
        return None

    def create_full_matrix_wide_deep(self):
        return None

    def create_full_matrix_gpt(self):
        return None

    def get_metric_value(self, model_type):
        result = None
        categorys = list(category_dict.keys())

        # read full matrix model_type
        # and calculate rmse using test matrix and full matrix

        train_path = f"files/full_matrix/{model_type}"
        test_path = f"files/train-test-matrix"

        train_value_views = []
        test_value_views = []
        train_value_comments = []
        test_value_comments = []
        train_value_likes = []
        test_value_likes = []
        result = []

        for category in categorys:
            views_predict_df = pd.read_csv(f"{train_path}/{category}_views_fullmatrix.csv")
            comments_predict_df = pd.read_csv(f"{train_path}/{category}_comments_fullmatrix.csv")
            likes_predict_df = pd.read_csv(f"{train_path}/{category}_likes_fullmatrix.csv")

            views_test_df = pd.read_csv(f"{test_path}/{category}_views_test.csv")
            comments_test_df = pd.read_csv(f"{test_path}/{category}_comments_test.csv")
            likes_test_df = pd.read_csv(f"{test_path}/{category}_likes_test.csv")

            keywords = views_predict_df.columns[1:-1]

            # views
            for idx, row in views_test_df.iterrows():
                if row["Unnamed: 0"] != views_predict_df.loc[idx, "Unnamed: 0"]:
                    print("false")
                for keyword in keywords:
                    if row[keyword] != 0:
                        train_value_views.append(views_predict_df.loc[idx, keyword])
                        test_value_views.append(row[keyword])
            # comments
            for idx, row in comments_test_df.iterrows():
                if row["Unnamed: 0"] != comments_predict_df.loc[idx, "Unnamed: 0"]:
                    print("false")
                for keyword in keywords:
                    if row[keyword] != 0:
                        train_value_comments.append(comments_predict_df.loc[idx, keyword])
                        test_value_comments.append(row[keyword])
            # likes
            for idx, row in likes_test_df.iterrows():
                if row["Unnamed: 0"] != likes_predict_df.loc[idx, "Unnamed: 0"]:
                    print("false")
                for keyword in keywords:
                    if row[keyword] != 0:
                        train_value_likes.append(likes_predict_df.loc[idx, keyword])
                        test_value_likes.append(row[keyword])

        views_rmse = np.sqrt(mean_squared_error(train_value_views, test_value_views))
        comments_rmse = np.sqrt(mean_squared_error(train_value_comments, test_value_comments))
        likes_rmse = np.sqrt(mean_squared_error(train_value_likes, test_value_likes))

        views_mae = mean_absolute_error(train_value_views, test_value_views)
        comments_mae = mean_absolute_error(train_value_comments, test_value_comments)
        likes_mae = mean_absolute_error(train_value_likes, test_value_likes)

        views_r2 = r2_score(train_value_views, test_value_views)
        comments_r2 = r2_score(train_value_comments, test_value_comments)
        likes_r2 = r2_score(train_value_likes, test_value_likes)

        # print(views_rmse, comments_rmse, likes_rmse)
        # print(views_mae, comments_mae, likes_mae)
        # print(views_r2, comments_r2, likes_r2)

        result = pd.DataFrame(data=[[views_rmse, comments_rmse, likes_rmse]], columns=["views", "comments", "likes"])
        result.to_csv(f"files/metric/{model_type}_rmse.csv")

        result = pd.DataFrame(data=[[views_mae, comments_mae, likes_mae]], columns=["views", "comments", "likes"])
        result.to_csv(f"files/metric/{model_type}_mae.csv")

        result = pd.DataFrame(data=[[views_r2, comments_r2, likes_r2]], columns=["views", "comments", "likes"])
        result.to_csv(f"files/metric/{model_type}_r2.csv")
