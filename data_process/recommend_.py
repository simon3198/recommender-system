import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.sparse.linalg import svds
from sklearn.decomposition import TruncatedSVD
from tqdm import tqdm

from .setting import category_dict


class Recommender:
    def __init__(self) -> None:
        pass

    def test(self):
        print("hello")

    def create_full_matrix(self):
        metrics = ["views", "likes", "comments"]

        base_path = "files/matrix"

        categorys = list(category_dict.keys())

        for metric in metrics:
            for category in tqdm(categorys):
                # print(metric,category)
                data = pd.read_csv(base_path + f"/{category}_{metric}.csv", index_col=0)
                channelid = data["channelid"]
                data.drop("channelid", axis=1, inplace=True)
                data = data.T
                matrix = data.values
                # print(matrix)
                # metric_mean은 keyword의 평균 조회수
                metric_mean = np.mean(matrix, axis=1)

                # R_user_mean : 사용자-영화에 대해 사용자 평균 평점을 뺀 것.
                matrix_user_mean = matrix - metric_mean.reshape(-1, 1)

                U, sigma, Vt = svds(matrix_user_mean, k=12)
                sigma = np.diag(sigma)
                svd_user_predicted_ratings = np.dot(
                    np.dot(U, sigma), Vt
                ) + metric_mean.reshape(-1, 1)
                df_svd_preds = pd.DataFrame(
                    svd_user_predicted_ratings, columns=data.columns, index=data.index
                )
                # df_svd_preds.head()
                df_svd_preds = df_svd_preds.T
                df_svd_preds["channedid"] = channelid

                df_svd_preds.to_csv(
                    f"files/full_matrix/{category}_{metric}_fullmatrix.csv"
                )
