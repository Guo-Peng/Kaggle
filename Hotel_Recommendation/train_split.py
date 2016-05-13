# -*- coding: UTF-8 -*-
import pandas as pd
import numpy as np


def split_data(path, n=10000):
    user_ids = pd.read_csv(path + 'train.csv', usecols=['user_id']).user_id.unique()
    np.random.shuffle(user_ids)
    return user_ids[:n]


def get_split(path, user):
    chunker = pd.read_csv(path + 'train.csv', chunksize=1000000)
    data_split = []
    for chunk in chunker:
        data_split.append(chunk[chunk.user_id.isin(user)])
    return pd.concat(data_split)


def train_test_split(data):
    data["date_time"] = pd.to_datetime(data["date_time"])
    data["year"] = data["date_time"].dt.year
    data["month"] = data["date_time"].dt.month
    train = data[((data.year == 2013) | ((data.year == 2014) & (data.month < 8)))]
    test = data[((data.year == 2014) & (data.month >= 8))]
    return train, test


def main(data_path):
    user_id = split_data(data_path)
    train_split = get_split(data_path, user_id)
    train, test = train_test_split(train_split)
    train.to_csv(data_path + 'train_new.csv', index_label='id')
    test.to_csv(data_path + 'test_new.csv', index_label='id')


if __name__ == '__main__':
    d_path = '../../kaggle_data/hotel_recommendation/data/'
    main(d_path)
