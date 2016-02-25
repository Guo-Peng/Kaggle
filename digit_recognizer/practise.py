# -*- coding: UTF-8 -*-
import pandas as pd


def get_result(path):
    data = pd.read_csv(path)
    return data['Label'].values


d1 = get_result('/Users/Peterkwok/Kaggle/result_linearSVM.csv')
d2 = get_result('/Users/Peterkwok/Kaggle/result_RandomForest.csv')
num = (d1 == d2).sum()
print num, d1.shape[0]
