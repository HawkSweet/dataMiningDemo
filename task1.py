# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import random
import math


def splitData(datasize, train_ratio):
    train_set = set()
    random.seed(2018)
    for i in range(math.floor(train_ratio * datasize)):
        train_set.add(random.randrange(0, datasize))
    test_set = set(range(datasize)) - train_set
    return train_set, test_set


if __name__ == '__main__':
    # data details
    data_file = pd.read_csv('./data/data-utf8.csv')
    datasize = data_file.shape[0]
    print(data_file.head())
    # print(data_file.info())
    print(data_file.describe())
    print(data_file.shape)
    print('positive sample, negative samples\n', data_file.status.value_counts())

    # know the data type
    print("data types: ", data_file.dtypes)

    # delete irrelevant features
    df1 = data_file
    drop_cols = [df1.columns[0], 'custid', 'trade_no', 'bank_card_no', 'id_name', 'latest_query_time',
                 'loans_latest_time', 'source', 'first_transaction_time']
    df1.drop(drop_cols, axis=1, inplace=True)
    print("after delete features: ", df1.shape)

    # transform to a number value type
    transform_cols = ['reg_preference_for_trad']
    city_value = df1[transform_cols[0]].unique()
    print(city_value)
    num_value = [1, 3, 5, 2, 4]
    for i in range(len(num_value)):
        df1[transform_cols[0]][df1[transform_cols[0]] == city_value[i]] = num_value[i]
    print("after transform columns: ", df1[transform_cols[0]].head())

    data = df1.infer_objects()#infer feature type
    tmp = data.dtypes
    print(data.dtypes.value_counts())
    # print(tmp[tmp.values == 'object'].index)

    # solve the missing values
    for column in list(df1.columns[:]):
        fill_value = df1[column][df1[column]!= np.nan].mean()
        fill_value = round(fill_value, 3)
        df1[column].fillna(value=fill_value, inplace=True)
    tmp = df1.isnull().T.any()
    for i in tmp:#判断是否还有空值
        if i is True:
            print('exist a Na value')

    # split to train set and test set
    train_set, test_set = splitData(datasize, train_ratio=0.7)
    df1.loc[train_set].to_csv('./data/train.csv', index=False)
    df1.loc[test_set].to_csv('./data/test.csv', index=False)