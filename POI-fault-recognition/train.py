# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 11:34:06 2020

@author: 陈宇
"""


import numpy as np
import pandas as pd
import lightgbm as lgb 
from sklearn import metrics
import sklearn
from data_processor import process
import hparams as hp

def print_result(preds_offline, test):
    offline = test[['dot_id','label']].copy()
    preds_label = [1 if a > hp.pred_thread else 0 for a in preds_offline]
    #print(offline)
    #print(preds_offline)
    #print(preds_label)
    offline['preds'] = preds_offline
    offline.label = offline['label'].astype(np.float64)
    #print('log_loss', metrics.log_loss(offline.label, preds_offline))
    print('auc', metrics.roc_auc_score(offline.label, preds_label))
    print("accuracy_score", metrics.accuracy_score(offline.label, preds_label))
    classify_report = sklearn.metrics.classification_report(offline.label, preds_label)
    print(classify_report)






# train, test = process([r'dataset\AroundDotsNonNoise.csv',
#                    r'dataset\AroundDotsNonNoise2.csv',
#                    r'dataset\TrainDotsNoise.csv',
#                    r'dataset\TrainDotsNonNoise.csv',
#                    r'dataset\TestDotsNoise.csv',
#                    r'dataset\TrainDotsNoise2.csv',
#                    #r'dataset\TestDotsNonNoise.csv'
#                    ])

train = pd.read_csv(r'dataset\train_data.csv')
test = pd.read_csv(r'dataset\test_data.csv')

drop_columns = ['dot_id', 'label', 'x', 'y', 'has_filter_desc', 'filter_cfd']

print("训练集")
y_train = train.label                                                  # 训练集标签
X_train = train.drop(drop_columns, axis=1)                  # 训练集特征矩阵
#X_train = (X_train-X_train.min())/(X_train.max()-X_train.min())

print("测试集")
offline_test_X=test.drop(drop_columns, axis=1) # 线下测试特征矩阵
online_test_X=test.drop(['dot_id'], axis=1)              # 线上测试特征矩阵
y_test = test['label']
#online_test_X = (online_test_X-online_test_X.min())/(online_test_X.max()-online_test_X.min())

#CRFNFL_Denoise
import CRF2
traindata = pd.concat([y_train, X_train], axis = 1).values
testdata = pd.concat([y_test, offline_test_X], axis = 1).values
#denoiseTraindata, noiseForest = CRF2.CRFNFL_Denoise(traindata, testdata)
denoiseTraindata = np.loadtxt(open(r'dataset\denoised_train_data.csv',"rb"), delimiter=",", skiprows=0)
y_train = denoiseTraindata[:, 0]
X_train = denoiseTraindata[:, 1:]
offline_test_X = offline_test_X.values
y_test = y_test.values
print(traindata.shape ,traindata)
print(denoiseTraindata.shape, denoiseTraindata)
np.savetxt("denoised_train_data.csv", denoiseTraindata, delimiter=',')


### 数据转换
lgb_train = lgb.Dataset(X_train, y_train, free_raw_data=False)

### 开始训练
print('设置参数')
#params = hp.params
params = {}

print("开始训练")
lgb_model = lgb.train(params,                     
                lgb_train,                  
                num_boost_round=hp.num_boost_round,
                #early_stopping_rounds=hp.early_stopping_round
        )

# ### 线下预测
print("线下预测")
preds_offline1 = lgb_model.predict(offline_test_X, num_iteration=lgb_model.best_iteration)

lgb_model.save_model("model.txt")

res = pd.DataFrame({'dot_id':test['dot_id'], 'pred':preds_offline1, 'label':test['label']})
res.to_csv(r'ans.csv')


#xgboost
from xgboost import XGBRegressor
from xgboost import XGBClassifier
#xgboost 有封装好的分类器和回归器，可以直接用XGBRegressor建立模型
xgb_model = XGBClassifier()
# Add silent=True to avoid printing out updates with each cycle
xgb_model.fit(X_train, y_train)
preds_offline2 = xgb_model.predict(offline_test_X)


#GBDT
from sklearn.ensemble import GradientBoostingClassifier
gbdt_model = GradientBoostingClassifier(n_estimators=50, random_state=10, subsample=0.6, max_depth=7,
                                  min_samples_split=900)
gbdt_model.fit(X_train, y_train)
preds_offline3 = gbdt_model.predict(offline_test_X)

#Adaboost
from sklearn.ensemble import AdaBoostClassifier
ada_model = AdaBoostClassifier(n_estimators=10)
ada_model.fit(X_train, y_train)
preds_offline4 = ada_model.predict(offline_test_X)


#Vote
preds_label = [1 if a > hp.pred_thread else 0 for a in preds_offline1]
preds_offline5 = preds_offline1 + preds_offline2 + preds_offline3 + preds_offline4
for i in range(len(preds_offline5)):
    if preds_offline5[i] > 2:
        preds_offline5[i] = 1
    else:
        if preds_offline5[i] == 2:
            preds_offline5[i] = 0
        else:
            preds_offline5[i] = 0
        
#stacking
from sklearn.model_selection import KFold
SEED = 0 # for reproducibility
NFOLDS = 5 # set folds for out-of-fold prediction
kf = KFold(n_splits=NFOLDS, random_state=SEED)
ntrain = len(y_train)
ntest = len(y_test)
def get_oof(clf, x_train, y_train, x_test):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    # np.empty()返回一个随机元素的矩阵
    oof_test_skf = np.empty((NFOLDS, ntest))  
 
    for i, (train_index, test_index) in enumerate(kf.split(oof_train)):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]
 
        clf.fit(x_tr, y_tr)
 
        oof_train[test_index] = clf.predict(x_te)
        # k则验证之后会产生[k x ntest]个验证集
        oof_test_skf[i, :] = clf.predict(x_test)
    # 按列取平均值
    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)

oof_train1, oof_test1 = get_oof(lgb.LGBMClassifier(), X_train, y_train, offline_test_X)
oof_train2, oof_test2 = get_oof(xgb_model, X_train, y_train, offline_test_X)
oof_train3, oof_test3 = get_oof(gbdt_model, X_train, y_train, offline_test_X)
oof_train4, oof_test4 = get_oof(ada_model, X_train, y_train, offline_test_X)

base_predictions_train = pd.DataFrame( {'LightGBM': oof_train1.ravel(),
     'XGBoost': oof_train2.ravel(),
     'GBDT': oof_train3.ravel(),
      'AdaBoost': oof_train4.ravel()
    })

base_predictions_test = pd.DataFrame( {'LightGBM': oof_test1.ravel(),
     'XGBoost': oof_test2.ravel(),
     'GBDT': oof_test3.ravel(),
      'AdaBoost': oof_test4.ravel()
    })
#LR作为最后判定
from sklearn import linear_model
lr = linear_model.LogisticRegression(C=1e5)
lr.fit(base_predictions_train, y_train)
preds_offline6 = lr.predict(base_predictions_test)

    

#CRFNFL_GBDT

#gbdt_pre1 = CRF2.CRFNFL_GBDT(traindata, testdata, testdata)
# gbdt_pre1 = CRF2.gbdtFunc(denoiseTraindata, testdata)
# gbdt_pre0 = CRF2.gbdtFunc(traindata, testdata)
#打印结果
print('LightGBM')
print_result(preds_offline1, test)
print('XGBoost')
print_result(preds_offline2, test)
print('GBDT')
print_result(preds_offline3, test)
print('AdaBoost')
print_result(preds_offline4, test)
print('voting')
print_result(preds_offline5, test)
print('stacking')
print_result(preds_offline6, test)
#print('CRFNFL_GBDT', gbdt_pre0, gbdt_pre1)


