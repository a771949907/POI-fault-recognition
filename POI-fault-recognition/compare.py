# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 19:32:41 2020

@author: 77194
"""

import numpy as np
import pandas as pd
import lightgbm as lgb 
from sklearn import metrics
import sklearn
from data_processor import process
import hparams as hp
import CRF2

train, test = process([r'dataset\AroundDotsNonNoise.csv',
                   r'dataset\AroundDotsNonNoise2.csv',
                   r'dataset\TrainDotsNoise.csv',
                   r'dataset\TrainDotsNonNoise.csv',
                   r'dataset\TestDotsNoise.csv',
                   r'dataset\TrainDotsNoise2.csv',
                   #r'dataset\TestDotsNonNoise.csv'
                   ])

drop_columns = ['dot_id', 'label', 'x', 'y', 'has_filter_desc', 'filter_cfd']

print("训练集")
y_train = train.label                                                  # 训练集标签
X_train = train.drop(drop_columns, axis=1)                  # 训练集特征矩阵
#X_train = (X_train-X_train.min())/(X_train.max()-X_train.min())

print("测试集")
offline_test_X=test.drop(drop_columns, axis=1) # 线下测试特征矩阵
online_test_X=test.drop(['dot_id'], axis=1)              # 线上测试特征矩阵
y_test = test['label']

print(type(y_train))
traindata = pd.concat([y_train, X_train], axis = 1).values
testdata = pd.concat([y_test, offline_test_X], axis = 1).values
CRF2.crfnfl_all(traindata, testdata, testdata)