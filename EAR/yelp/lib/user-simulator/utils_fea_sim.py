# BB-8 and R2-D2 are best friends.

import sys
sys.path.insert(0, '../FM')
sys.path.insert(0, '../yelp')

import random
import torch
import torch.nn as nn
import json
import pickle
import numpy as np
from sklearn.metrics import roc_auc_score
import time
from torch.nn.utils import clip_grad_value_, clip_grad_norm_
from torch.nn.utils.rnn import pad_sequence
from config import global_config as cfg
from sklearn.metrics.pairwise import cosine_similarity
import argparse
import sys

from FM_old import FactorizationMachine


def feature_similarity(given_preference, userID, TopKTaxo):
    preference_matrix_all = cfg.user_emb[[userID]]
    if len(given_preference) > 0:
        preference_matrix = cfg.emb_matrix[given_preference]
        preference_matrix_all = np.concatenate((preference_matrix, preference_matrix_all), axis=0)

    total_result = list()
    result_dict = dict()

    for index, big_feature in enumerate(cfg.FACET_POOL[: 3]):
        left = cfg.spans[index][0]
        right = cfg.spans[index][1]
        big_feature_matrix = cfg.emb_matrix[left: right, :]
        cosine_result = cosine_similarity(big_feature_matrix, preference_matrix_all)
        cosine_result = np.sum(cosine_result, axis=1)
        cosine_result = -np.sort(-cosine_result)  # Sort it descending
        total_result.append((cosine_result))
        result_dict[big_feature] = np.sum(cosine_result[: TopKTaxo]) / (len(given_preference) + 1)  # choose top K, normalization

    for big_feature in cfg.FACET_POOL[3: ]:  # for those features, not in {city stars price}
        feature_index = [item + cfg.star_count + cfg.city_count + cfg.price_count for item in cfg.taxo_dict[big_feature]]
        big_feature_matrix = cfg.emb_matrix[feature_index]
        cosine_result = cosine_similarity(big_feature_matrix, preference_matrix_all)
        cosine_result = np.sum(cosine_result, axis=1)
        cosine_result = -np.sort(-cosine_result)  # Sort it descending
        total_result.append((cosine_result))
        result_dict[big_feature] = np.sum(cosine_result[: TopKTaxo])  / (len(given_preference) + 1)  # choose top K, normalization

    return result_dict
