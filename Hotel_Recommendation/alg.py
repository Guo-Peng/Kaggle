# -*- coding: UTF-8 -*-
import pandas as pd
import numpy as np
from average_precision import mapk


def most_common(train, test):
    most_common_clusters = list(train.hotel_cluster.value_counts().head().index)
    predictions = [most_common_clusters for _ in range(test.shape[0])]
    target = [[l] for l in test["hotel_cluster"]]
    return mapk(target, predictions, k=5)


def main(data_path):
    train = pd.read_csv(data_path + 'train_new.csv')
    test = pd.read_csv(data_path + 'test_new.csv')
    score = most_common(train, test)
    print score


if __name__ == '__main__':
    d_path = '../../kaggle_data/hotel_recommendation/data/'
    main(d_path)
