# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 10:42:18 2020

@author: 陈宇
"""


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn import preprocessing
enc = preprocessing.OneHotEncoder()  #相关onehot的包

#归一化
def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

#独热编码
def set_OneHotEncoder(data,colname,start_index,end_index):
    '''
    Parameters
    ----------
    data : [[1,2,3,4,7],[0,5,6,8,9]]
    start_index : 起始列位置索引
    end_index : 结束列位置索引. 如start_index为1，end_index为3，则取出来的为[[2,3,4],[5,6,8]]
    
    Returns
    -------
    x_ : 对应列经过独热编码后的数据框
    '''
    if type(data) == pd.core.frame.DataFrame:
        data = np.array(data).tolist()
    if type(data) != list:
        return  'Error dataType, expect list but ' + str(type(data))
    _data,_colname =[line[:start_index] for line in data],colname[:start_index]
    data_,colname_ = [line[end_index+1:] for line in data],colname[end_index+1:]
    
    data = [line[start_index:end_index+1] for line in data]
    data = pd.DataFrame(data)
    data.columns = colname[start_index:end_index+1]
    enc.fit(data)
    x_ = enc.transform(data).toarray() #已生成
    x_ = [list(line) for line in x_]
    #加栏目名
    new_columns = []
    for col in data.columns:
        dd = sorted(list(set(list(data[col])))) #去重并根据升序排列
        for line in dd:
            new_columns.append(str(col)+'#'+str(line))
 
    end_x = list(map(lambda x,y,z:x+y+z,_data,x_,data_))
    end_columns = list(_colname)+new_columns+list(colname_)
    x__ = pd.DataFrame(end_x,columns = end_columns)
    return x__ #返回数据框形式


#数据预处理
def process(file_list, test_size=0.1):
    '''
    数据的读取和预处理.

    Parameters
    ----------
    file_list : ['csv_file_path1', 'csv_file_path2', ...]
    test_size : 测试集占数据集的比
    
    Returns
    -------
    train : 训练集数据框
    test : 测试集数据框
    '''
    data = pd.concat([pd.read_csv(file) for file in file_list], ignore_index = True)
    data.drop_duplicates(inplace=True)
    data.fillna(0, inplace=True)
    data['has_inf'] = data.has_inf.apply(lambda inf: 1 if inf else 0)
    data['has_filter_desc'] = data.has_filter_desc.apply(lambda inf: 1 if inf else 0)
    data['flg_type'] = data.flg_type.apply(lambda t: t // 100000)
    data = shuffle(data)
    data = set_OneHotEncoder(data, data.columns, 4, 4)
    print(data.columns)
    if test_size >= 0:
        return train_test_split(data, test_size=test_size, random_state=21)
    else:
        return data