import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from keras.models import Model
from scipy.sparse.linalg import svds
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dot, Embedding, Flatten, Input
from tensorflow.keras.models import Model
from tqdm import tqdm

from .als import AlternatingLeastSquares
from .mlp_mf import MLP_MF, NCFData
from .ncf import GMF_and_MLP
from .setting import category_dict


class Recommender:
    def __init__(self) -> None:
        self.BATCH_SIZE = 16
        self.EPOCHS = 100

    def test(self):
        print("hello")

    def train(self, model, train_loader, criterion, optimizer, DEVICE):
        model.train()
        train_loss = 0
        for user, item, label in train_loader:
            user = user.to(DEVICE)
            item = item.to(DEVICE)
            label = label.float().to(DEVICE)
            optimizer.zero_grad()
            output = model(user, item)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        return train_loss

    def predict(self, model, loader, DEVICE):
        model.eval()
        result = []
        with torch.no_grad():
            for user, item in loader:
                user = user.to(DEVICE)
                item = item.to(DEVICE)
                output = model(user, item)
                result.append(output.view(-1))

        return result

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

    def create_full_matrix_coll_keyword_based(self):
        metrics = ["views", "likes", "comments"]

        base_path = "files/train-test-matrix"

        categorys = list(category_dict.keys())

        # make full matrix with train data set
        for metric in metrics:
            for category in tqdm(categorys):
                print(category)
                data = pd.read_csv(base_path + f"/{category}_{metric}_train.csv", index_col=0)
                channelid = data["channelid"]
                data.drop("channelid", axis=1, inplace=True)
                matrix = data.values
                keyword_similarity = cosine_similarity(matrix.T)
                predicted_matrix = np.zeros(matrix.shape)
                for inf_index in range(matrix.shape[0]):
                    for keyword_index in range(matrix.shape[1]):
                        if matrix[inf_index, keyword_index] == 0:
                            # Calculate the predicted rating for this user and item
                            numerator = np.sum(matrix[inf_index] * keyword_similarity[keyword_index])
                            denominator = np.sum(np.abs(keyword_similarity[keyword_index]))
                            predicted_rating = numerator / (denominator + 1e-6)  # Avoid division by zero
                            predicted_matrix[inf_index, keyword_index] = predicted_rating

                coll_keyword_df = pd.DataFrame(predicted_matrix, columns=data.columns, index=data.index)

                coll_keyword_df["channedid"] = channelid

                coll_keyword_df.to_csv(f"files/full_matrix/coll_keyword/{category}_{metric}_fullmatrix.csv")

    def create_full_matrix_coll_inf_based(self):
        metrics = ["views", "likes", "comments"]

        base_path = "files/train-test-matrix"

        categorys = list(category_dict.keys())

        # make full matrix with train data set
        for metric in metrics:
            for category in tqdm(categorys):
                print(category)
                data = pd.read_csv(base_path + f"/{category}_{metric}_train.csv", index_col=0)
                channelid = data["channelid"]
                data.drop("channelid", axis=1, inplace=True)
                matrix = data.values
                user_similarity = cosine_similarity(matrix)

                # Fill out the full user-item matrix using user-based collaborative filtering
                predicted_matrix = np.zeros(matrix.shape)

                for inf_index in range(matrix.shape[0]):
                    for keyword_index in range(matrix.shape[1]):
                        if matrix[inf_index, keyword_index] == 0:
                            # Calculate the predicted rating for this user and item
                            weighted_sum = np.sum(user_similarity[inf_index] * matrix[:, keyword_index])
                            denominator = np.sum(np.abs(user_similarity[inf_index]))
                            predicted_rating = weighted_sum / (denominator + 1e-6)  # Avoid division by zero
                            predicted_matrix[inf_index, keyword_index] = predicted_rating

                coll_inf_df = pd.DataFrame(predicted_matrix, columns=data.columns, index=data.index)

                coll_inf_df["channedid"] = channelid

                coll_inf_df.to_csv(f"files/full_matrix/coll_inf/{category}_{metric}_fullmatrix.csv")

    def create_full_matrix_als(self):
        metrics = ["views", "likes", "comments"]

        base_path = "files/train-test-matrix"

        categorys = list(category_dict.keys())

        # make full matrix with train data set
        for metric in metrics:
            for category in tqdm(categorys):
                print(category)
                data = pd.read_csv(base_path + f"/{category}_{metric}_train.csv", index_col=0)
                channelid = data["channelid"]
                data.drop("channelid", axis=1, inplace=True)
                matrix = data.iloc[:, :].values
                als = AlternatingLeastSquares(R=matrix, reg_param=0.1, epochs=100, verbose=True, k=5)
                als.fit()

                deep = als.get_complete_matrix()
                deep_df = pd.DataFrame(deep, columns=data.columns, index=data.index)

                deep_df["channedid"] = channelid

                deep_df.to_csv(f"files/full_matrix/als/{category}_{metric}_fullmatrix.csv")

    def create_full_matrix_mlp_mf(self):
        metrics = ["views", "likes", "comments"]

        base_path = "files/train-test-matrix"

        categorys = list(category_dict.keys())

        # make full matrix with train data set
        for metric in metrics:
            for category in tqdm(categorys):
                print(category)
                DEVICE = None
                if torch.cuda.is_available():
                    DEVICE = torch.device("cuda")
                else:
                    DEVICE = torch.device("cpu")

                data = pd.read_csv(base_path + f"/{category}_{metric}_train.csv", index_col=0)
                channelid = data["channelid"]
                data.drop("channelid", axis=1, inplace=True)
                matrix = data.iloc[:, :].values

                user_num = matrix.shape[0]
                item_num = matrix.shape[1]
                features_idx = []
                for i in range(user_num):
                    for j in range(item_num):
                        features_idx.append([i, j])

                features = np.transpose(matrix.nonzero()).tolist()
                labels = []

                for i, j in features:
                    labels.append(matrix[i, j])

                model = MLP_MF(user_num=user_num, item_num=item_num, factor_num=20).to(DEVICE)
                criterion = nn.MSELoss()
                optimizer = optim.Adam(model.parameters(), lr=0.01)

                train_dataset = NCFData(features, labels)
                train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.BATCH_SIZE)

                for epoch in range(1, self.EPOCHS + 1):
                    train_loss = self.train(model, train_loader, criterion, optimizer, DEVICE)
                    print(f"\n[EPOCH: {epoch}], \tTrain Loss: {train_loss:.4f}")
                dataset = NCFData(features_idx)
                loader = torch.utils.data.DataLoader(dataset, batch_size=self.BATCH_SIZE)
                pred = self.predict(model, loader, DEVICE)
                pred = torch.cat(pred).view(user_num, item_num)

                pred_df = pd.DataFrame(pred, columns=data.columns, index=data.index)

                pred_df["channedid"] = channelid

                pred_df.to_csv(f"files/full_matrix/mlp_mf/{category}_{metric}_fullmatrix.csv")

    def create_full_matrix_ncf(self):
        metrics = ["views", "likes", "comments"]

        base_path = "files/train-test-matrix"

        categorys = list(category_dict.keys())

        # make full matrix with train data set
        for metric in metrics:
            for category in tqdm(categorys):
                print(category)

                DEVICE = None
                if torch.cuda.is_available():
                    DEVICE = torch.device("cuda")
                else:
                    DEVICE = torch.device("cpu")
                data = pd.read_csv(base_path + f"/{category}_{metric}_train.csv", index_col=0)
                channelid = data["channelid"]
                data.drop("channelid", axis=1, inplace=True)
                matrix = data.iloc[:, :].values

                user_num = matrix.shape[0]
                item_num = matrix.shape[1]
                features_idx = []
                for i in range(user_num):
                    for j in range(item_num):
                        features_idx.append([i, j])

                features = np.transpose(matrix.nonzero()).tolist()
                labels = []

                for i, j in features:
                    labels.append(matrix[i, j])

                model = GMF_and_MLP(user_num=user_num, item_num=item_num, factor_num=20).to(DEVICE)
                criterion = nn.MSELoss()
                optimizer = optim.Adam(model.parameters(), lr=0.01)

                train_dataset = NCFData(features, labels)
                train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.BATCH_SIZE)

                for epoch in range(1, self.EPOCHS + 1):
                    train_loss = self.train(model, train_loader, criterion, optimizer, DEVICE)
                    print(f"\n[EPOCH: {epoch}], \tTrain Loss: {train_loss:.4f}")

                dataset = NCFData(features_idx)
                loader = torch.utils.data.DataLoader(dataset, batch_size=self.BATCH_SIZE)
                pred = self.predict(model, loader, DEVICE)
                pred = torch.cat(pred).view(user_num, item_num)

                pred_df = pd.DataFrame(pred, columns=data.columns, index=data.index)

                pred_df["channedid"] = channelid

                pred_df.to_csv(f"files/full_matrix/ncf/{category}_{metric}_fullmatrix.csv")

    def get_metric_value(self, model_type):
        result = None
        categorys = list(category_dict.keys())

        # read full matrix model_type
        # and calculate rmse using test matrix and full matrix

        train_path = f"files/full_matrix/{model_type}"
        test_path = f"files/train-test-matrix"

        # 카테고리별로 metric 계산 후 평균 내기
        rmse_result = []
        mae_result = []
        r2_result = []
        for category in categorys:
            train_value_views = []
            test_value_views = []
            train_value_comments = []
            test_value_comments = []
            train_value_likes = []
            test_value_likes = []
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
                    if row[keyword] != 0 and views_predict_df.loc[idx, keyword] != 0:
                        train_value_views.append(views_predict_df.loc[idx, keyword])
                        test_value_views.append(row[keyword])
            # comments
            for idx, row in comments_test_df.iterrows():
                if row["Unnamed: 0"] != comments_predict_df.loc[idx, "Unnamed: 0"]:
                    print("false")
                for keyword in keywords:
                    if row[keyword] != 0 and comments_predict_df.loc[idx, keyword] != 0:
                        train_value_comments.append(comments_predict_df.loc[idx, keyword])
                        test_value_comments.append(row[keyword])
            # likes
            for idx, row in likes_test_df.iterrows():
                if row["Unnamed: 0"] != likes_predict_df.loc[idx, "Unnamed: 0"]:
                    print("false")
                for keyword in keywords:
                    if row[keyword] != 0 and likes_predict_df.loc[idx, keyword] != 0:
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

            rmse_result.append([views_rmse, comments_rmse, likes_rmse])
            mae_result.append([views_mae, comments_mae, likes_mae])
            r2_result.append([views_r2, comments_r2, likes_r2])

        result_rmse = pd.DataFrame(data=rmse_result, columns=["views", "comments", "likes"], index=categorys)
        result_rmse.loc["mean"] = result_rmse.mean()
        result_rmse.to_csv(f"files/metric/test/{model_type}_rmse.csv")

        result_mae = pd.DataFrame(data=mae_result, columns=["views", "comments", "likes"], index=categorys)
        result_mae.loc["mean"] = result_mae.mean()
        result_mae.to_csv(f"files/metric/test/{model_type}_mae.csv")

        result_r2 = pd.DataFrame(data=r2_result, columns=["views", "comments", "likes"], index=categorys)
        result_r2.loc["mean"] = result_r2.mean()
        result_r2.to_csv(f"files/metric/test/{model_type}_r2.csv")

    def calculate_eval_by_category(self):
        # 먹방_r2.csv
        # 안에 행은 모델 이름 열은 views,likes.comments
        # 총 18 value * 20
        categorys = list(category_dict.keys())

        eval_list = ["rmse", "mae", "r2"]
        model_list = ["svd", "coll_keyword", "coll_inf", "als", "mlp_mf", "ncf"]

        for category in categorys:
            for eval_type in eval_list:
                result = []
                for model_type in model_list:
                    print(eval_type, model_type)
                    data = pd.read_csv(f"files/metric/test/{model_type}_{eval_type}.csv", index_col=0)
                    row = list(data.loc[category])
                    result.append(row)
                pd.DataFrame(data=result, index=model_list, columns=["views", "comments", "likes"]).to_csv(
                    f"files/metric/eval_by_cat/{category}_{eval_type}.csv"
                )

    def calculate_eval_by_metrix(self):
        # views_r2.csv
        # 안에 행은 모델 이름 열은 카테고리
        # 총 120 value * 3
        categorys = list(category_dict.keys())

        stat_list = ["views", "comments", "likes"]
        eval_list = ["rmse", "mae", "r2"]
        model_list = ["svd", "coll_keyword", "coll_inf", "als", "mlp_mf", "ncf"]

        for stat in stat_list:
            for eval_type in eval_list:
                result = []
                for model_type in model_list:
                    print(eval_type, model_type)
                    data = pd.read_csv(f"files/metric/test/{model_type}_{eval_type}.csv", index_col=0)
                    row = list(data.loc[:, stat])
                    result.append(row[:-1])
                data = pd.DataFrame(data=result, index=model_list, columns=categorys)
                data["mean"] = data.mean(axis=1)
                data.to_csv(f"files/metric/eval_by_metrix/{stat}_{eval_type}.csv")

        return None
