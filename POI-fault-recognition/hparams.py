# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 11:43:15 2020

@author: 陈宇
"""

params = {
    'max_depth': -1,
    'min_data_in_leaf': 20,
    'min_sum_hessian_in_leaf': 1e-3,
    'feature_fraction': 1,
    'feature_fraction_seed': 2,
    'bagging_fraction': 1,
    'bagging_freq': 0,
    'bagging_seed': 3,
    'lambda_l1': 0,
    'lambda_l2': 0,
    'min_split_gain': 0,
    'drop_rate': 0,
    'skip_rate': 0.5,
    'max_drop': 50,
    'uniform_drop': False,
    'xgboost_dart_mode': False,
    'drop_seed': 4,
    'top_rate': 1,
    'other_rate': 1,
    'min_data_per_group': 100,
    'max_cat_threshold': 32,
    'cat_smooth': 10,
    'cat_l2': 10,
    'top_k ': 20,
    'num_leaves': 63,
    'num_trees': 100
}

num_boost_round = 1000
early_stopping_round = 0
pred_thread = 0.5