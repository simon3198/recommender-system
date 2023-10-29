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


class GMF_and_MLP(nn.Module):
    def __init__(self, user_num, item_num, factor_num, num_layers=3):
        super(GMF_and_MLP, self).__init__()

        # GMF 임베딩
        self.GMF_user_embedding = nn.Embedding(user_num, factor_num)
        self.GMF_item_embedding = nn.Embedding(item_num, factor_num)

        # MLP 임베딩
        self.MLP_user_embedding = nn.Embedding(user_num, factor_num)
        self.MLP_item_embedding = nn.Embedding(item_num, factor_num)

        MLP_modules = []
        input_size = factor_num * 2
        for i in range(num_layers):
            MLP_modules.append(nn.Linear(input_size, input_size // 2))
            MLP_modules.append(nn.ReLU())
            input_size //= 2
        self.MLP_layers = nn.Sequential(*MLP_modules)

        # FC
        self.FC_layer = nn.Sequential(nn.Linear(factor_num + input_size, 1))
        self._init_weight_()

    def _init_weight_(self):
        # weight 초기화
        nn.init.normal_(self.GMF_user_embedding.weight, std=0.01)
        nn.init.normal_(self.GMF_item_embedding.weight, std=0.01)
        nn.init.normal_(self.MLP_user_embedding.weight, std=0.01)
        nn.init.normal_(self.MLP_item_embedding.weight, std=0.01)

        for m in self.MLP_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

        for m in self.FC_layer:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, user, item):
        GMF_user_embedding = self.GMF_user_embedding(user)
        GMF_item_embedding = self.GMF_item_embedding(item)

        output_GMF = GMF_user_embedding * GMF_item_embedding

        MLP_user_embedding = self.MLP_user_embedding(user)
        MLP_item_embedding = self.MLP_item_embedding(item)

        concat_two_latent_vactors = torch.cat((MLP_user_embedding, MLP_item_embedding), -1)
        output_MLP = self.MLP_layers(concat_two_latent_vactors)

        concat_GMF_MLP = torch.cat((output_GMF, output_MLP), -1)

        out = self.FC_layer(concat_GMF_MLP)
        out = out.view(-1)

        return out
