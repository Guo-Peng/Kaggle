# -*- coding: UTF-8 -*-
import pandas as pd
from time import time


def correlation_cal(path):
    columns = list(pd.read_csv(path + 'train.csv', nrows=10).columns)
    columns.remove('hotel_cluster')
    cor = []
    for index, col in enumerate(columns):
        start = time()
        data = pd.read_csv(path + 'train.csv', usecols=['hotel_cluster', col])
        cor.append(data.corr()['hotel_cluster'])
        print 'index: %d  column: %s  time: %f' % (index + 1, col, (time() - start) / 60)
    return pd.concat(cor).drop_duplicates()


def main(data_path):
    # 无明显的线性相关，linear regression 跟 logistic regression 效果不会好
    correlations = correlation_cal(data_path)
    correlations.to_csv('../../kaggle_data/hotel_recommendation/property/correlations')


if __name__ == '__main__':
    d_path = '../../kaggle_data/hotel_recommendation/data/'
    main(d_path)
