import json
import os
import re
from collections import Counter
from datetime import datetime
from math import log

import numpy as np
import pandas as pd
from konlpy.tag import Okt
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm

from .setting import category_dict


class Preprocesser:
    def __init__(self):
        self.JVM_PATH = "/Library/Java/JavaVirtualMachines/zulu-15.jdk/Contents/Home/bin/java"
        self.okt = Okt(jvmpath=self.JVM_PATH)

    def preprocess(self):
        data = pd.read_csv("./files/video_statistics.csv", index_col=0)
        data["keyword"] = ""

        for idx, doc in enumerate(tqdm(data["title"])):
            word_list = []
            doc = self.okt.nouns(doc)
            for word in doc:
                if word.isalnum() == False or word.isdigit() == True or len(word) == 1:
                    continue
                word_list.append(word)

            # print(set(word_list))
            tags = data.iloc[idx]["tags"]
            tag_doc = self.okt.nouns(tags)
            for word in tag_doc:
                if word.isalnum() == False or word.isdigit() == True or len(word) == 1:
                    continue
                word_list.append(word)
            data.loc[idx, "keyword"] = ",".join(list(set(word_list)))

        data.to_csv("files/video_statistics_keyword.csv")

    def save_keywords(self):
        data = pd.read_csv("files/video_statistics_keyword.csv", index_col=0)

        category_names = data["category"].unique()

        for category in tqdm(category_names):
            cat_data = data[data["category"] == category]
            # print(cat_data)
            # break

            channel_names = cat_data["channelname"].unique()

            sentence_list = []
            for channel_name in channel_names:
                try:
                    keywords = ",".join(list(cat_data[cat_data["channelname"] == channel_name]["keyword"]))

                    keywords = set(keywords.split(","))
                    sentence = " ".join(keywords)
                    sentence_list.append(sentence)
                except:
                    continue

            file_path = f"files/keyword/{category}_sentence_list.json"

            with open(file_path, "w", encoding="UTF-8") as json_file:
                json.dump(sentence_list, json_file, ensure_ascii=False, indent=4)

    def tf(self, t, d):
        return d.count(t)

    def idf(self, t, docs):
        N = len(docs)
        df = 0
        for doc in docs:
            df += t in doc
        return log(N / (df + 1))

    def tfidf(self, t, d):
        return self.tf(t, d) * self.idf(t)

    def create_keywords(self):
        for category in list(category_dict.keys()):
            file_path = f"./files/keyword/{category}_sentence_list.json"

            sentence_list = None
            with open(file_path, "r") as file:
                sentence_list = json.load(file)

            # print(sentence_list)
            # break

            vectorizer = CountVectorizer()
            dtm = vectorizer.fit_transform(sentence_list)
            tf = pd.DataFrame(dtm.toarray(), columns=vectorizer.get_feature_names_out())
            df = tf.astype(bool).sum(axis=0)
            # 문서 개수
            D = len(tf)

            # Inverse Document Frequency
            idf = np.log((D + 1) / (df + 1)) + 1
            tfidf = tf * idf
            tfidf = tfidf / np.linalg.norm(tfidf, axis=1, keepdims=True)

            # break
            final_tfidf = tfidf.sum().sort_values(ascending=False)[:100]

            final_tfidf.to_csv(f"files/keyword_columns/{category}keyword_columns.csv")

    def avg_dict(self, input_dict, channelid):
        for key, value in list(input_dict.items()):
            if len(value) == 0:
                input_dict[key] = 0
                continue
            avg = sum(value) / len(value)
            input_dict[key] = avg

        input_dict["channelid"] = channelid
        return input_dict

    def create_matrix_csv(self):
        data = pd.read_csv("files/video_statistics_keyword.csv", index_col=0)

        # Tdata split using temporal global split ( 기준 : 2023 / 6,7,8,9 영상 기준)

        for category in list(category_dict.keys()):
            file_path = f"./files/keyword_columns/{category}keyword_columns.csv"
            columns_df = pd.read_csv(file_path, index_col=0)
            columns = list(columns_df.index)

            # 특정 카테고리 추출
            cat_data = data[data["category"] == category]
            channel_names = cat_data["channelname"].unique()
            channel_id = None
            channel_dict_train = [[], [], []]
            channel_dict_test = [[], [], []]

            for channel_name in tqdm(channel_names):
                # 특정 채널 추출
                chan_data = cat_data[cat_data["channelname"] == channel_name]
                channel_id = chan_data["channelid"].unique()[0]

                # train data set
                chan_dict_views_train = {key: [] for key in columns}
                chan_dict_likes_train = {key: [] for key in columns}
                chan_dict_comments_train = {key: [] for key in columns}

                # test data set
                chan_dict_views_test = {key: [] for key in columns}
                chan_dict_likes_test = {key: [] for key in columns}
                chan_dict_comments_test = {key: [] for key in columns}

                split_date = datetime.strptime("2023-06-01", "%Y-%m-%d")

                # 이제 그 행의 keyword 와 column을 매칭하여 있으면 점수 추가
                for column in columns:
                    for index, row in chan_data.iterrows():
                        if type(row["keyword"]) == float:
                            continue
                        publishat = datetime.strptime(row["publishat"], "%Y-%m-%d")
                        if column in row["keyword"]:
                            if split_date > publishat:
                                chan_dict_views_train[column].append(int(row["views"]))
                                chan_dict_likes_train[column].append(int(row["likes"]))
                                chan_dict_comments_train[column].append(int(row["comments"]))
                            else:
                                chan_dict_views_test[column].append(int(row["views"]))
                                chan_dict_likes_test[column].append(int(row["likes"]))
                                chan_dict_comments_test[column].append(int(row["comments"]))

                # get the average
                chan_dict_views_train = self.avg_dict(chan_dict_views_train, channel_id)
                chan_dict_likes_train = self.avg_dict(chan_dict_likes_train, channel_id)
                chan_dict_comments_train = self.avg_dict(chan_dict_comments_train, channel_id)

                chan_dict_views_test = self.avg_dict(chan_dict_views_test, channel_id)
                chan_dict_likes_test = self.avg_dict(chan_dict_likes_test, channel_id)
                chan_dict_comments_test = self.avg_dict(chan_dict_comments_test, channel_id)

                channel_dict_train[0].append(chan_dict_views_train)
                channel_dict_train[1].append(chan_dict_likes_train)
                channel_dict_train[2].append(chan_dict_comments_train)

                channel_dict_test[0].append(chan_dict_views_test)
                channel_dict_test[1].append(chan_dict_likes_test)
                channel_dict_test[2].append(chan_dict_comments_test)
            # make matrix for category
            result = pd.DataFrame(
                channel_dict_train[0],
                columns=columns.append("channelid"),
                index=list(channel_names),
            )
            result.to_csv(f"./files/train-test-matrix/{category}_views_train.csv")

            result = pd.DataFrame(
                channel_dict_train[1],
                columns=columns.append("channelid"),
                index=list(channel_names),
            )
            result.to_csv(f"./files/train-test-matrix/{category}_likes_train.csv")

            result = pd.DataFrame(
                channel_dict_train[2],
                columns=columns.append("channelid"),
                index=list(channel_names),
            )
            result.to_csv(f"./files/train-test-matrix/{category}_comments_train.csv")

            result = pd.DataFrame(
                channel_dict_test[0],
                columns=columns.append("channelid"),
                index=list(channel_names),
            )
            result.to_csv(f"./files/train-test-matrix/{category}_views_test.csv")

            result = pd.DataFrame(
                channel_dict_test[1],
                columns=columns.append("channelid"),
                index=list(channel_names),
            )
            result.to_csv(f"./files/train-test-matrix/{category}_likes_test.csv")

            result = pd.DataFrame(
                channel_dict_test[2],
                columns=columns.append("channelid"),
                index=list(channel_names),
            )
            result.to_csv(f"./files/train-test-matrix/{category}_comments_test.csv")
            # break

        return None
