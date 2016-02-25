# -*- coding: UTF-8 -*-
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
import time


def linear_svm(train_path, test_path, result_path, store_path):
    x_train, y_train = get_data(train_path)
    x_test = get_data(test_path, False)
    y_test = train_predict(x_train, y_train, x_test)
    result = get_result(result_path)
    result_store(y_test, store_path)
    num = (result == y_test).sum()
    print num, result.shape[0]
    print float(num) / result.shape[0]


def train_predict(features, label, test_features):
    print 'get model...'
    # clf = svm.LinearSVC(C=5)
    # clf = svm.SVC()
    clf = RandomForestClassifier(n_estimators=100)
    # clf = KNeighborsClassifier(algorithm='kd_tree')
    print clf
    print 'fiting...'
    clf.fit(features, label)
    print 'predicting...'
    result = clf.predict(test_features)
    print 'done!'
    return result


def get_data(path, train=True):
    data = pd.read_csv(path).values
    if not train:
        return data
    features = data[:, 1:]
    label = data[:, :1]
    return features, label.ravel()


def get_result(path):
    data = pd.read_csv(path)
    return data['Label'].values


def result_store(result, path):
    data = pd.DataFrame({'ImageId': np.arange(1, result.shape[0] + 1), 'Label': result})
    data.to_csv(path, index=False)


if __name__ == '__main__':
    tr_path = '/Users/Peterkwok/Kaggle/train.csv'
    te_path = '/Users/Peterkwok/Kaggle/test.csv'
    true_path = '/Users/Peterkwok/Kaggle/rf_benchmark.csv'
    s_path = '/Users/Peterkwok/Kaggle/result_RandomForest_100.csv'
    start = time.time()
    linear_svm(tr_path, te_path, true_path, s_path)
    print (time.time() - start) / 60
