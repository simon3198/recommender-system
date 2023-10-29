import os
import time
import warnings

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

warnings.filterwarnings(action="ignore")


class MLP_MF(nn.Module):
    def __init__(self, user_num, item_num, factor_num, num_layers=3):
        super(MLP_MF, self).__init__()

        # 유저 임베딩
        self.user_embedding = nn.Embedding(user_num, factor_num)

        # 아이템 임베딩
        self.item_embedding = nn.Embedding(item_num, factor_num)

        MLP_modules = []
        input_size = factor_num * 2
        for i in range(num_layers):
            MLP_modules.append(nn.Linear(input_size, input_size // 2))
            MLP_modules.append(nn.ReLU())
            input_size //= 2
        self.MLP_layers = nn.Sequential(*MLP_modules)

        # FC
        self.FC_layer = nn.Sequential(nn.Linear(input_size, 1))
        self._init_weight_()

    def _init_weight_(self):
        # weight 초기화
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        for m in self.MLP_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

        for m in self.FC_layer:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, user, item):
        user_embedding = self.user_embedding(user)
        item_embedding = self.item_embedding(item)

        concat_two_latent_vactors = torch.cat((user_embedding, item_embedding), -1)
        output_MLP = self.MLP_layers(concat_two_latent_vactors)

        out = self.FC_layer(output_MLP)
        out = out.view(-1)

        return out


class NCFData(data.Dataset):
    def __init__(self, features, labels=None):
        super(NCFData, self).__init__()
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if self.labels is not None:
            user = self.features[idx][0]
            item = self.features[idx][1]
            label = self.labels[idx]

            return user, item, label

        else:
            user = self.features[idx][0]
            item = self.features[idx][1]
            return user, item
