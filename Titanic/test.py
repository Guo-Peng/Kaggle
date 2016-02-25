# encoding:utf-8

# http://www.cnblogs.com/north-north/p/4353365.html

import numpy as np
import pandas as pd
import re
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor

# replace NA
df = pd.read_csv('D:\\workspace\\Kaggle\\Titanic\\train.csv')
df.Embarked = df.Embarked.fillna(df.Embarked.dropna().mode()[0])
df.Cabin = df.Cabin.replace(np.nan, 'U0')

age_null = df[df.Age.isnull()]
age_not_null = df[df.Age.notnull()]
train = age_not_null[['Survived', 'Fare', 'Parch', 'SibSp', 'Pclass', 'Age']]
test = age_null[['Survived', 'Fare', 'Parch', 'SibSp', 'Pclass', 'Age']]
X = train.values[:, :-1]
Y = train.values[:, -1]
model = RandomForestRegressor(n_estimators=1000, n_jobs=-1)
model.fit(X, Y)
y = model.predict(test.values[:, :-1])
df.loc[(df.Age.isnull()), 'Age'] = y

# data transform

# 离散到连续
dummies_Embarked = pd.get_dummies(df.Embarked, prefix='Embarked_')
df = pd.concat([df, dummies_Embarked], axis=1)
df = df.drop('Embarked', axis=1)

df['CabinLetter'] = df.Cabin.map(lambda x: ''.join(re.findall('[a-zA-Z]', x)))
df.CabinLetter = pd.factorize(df.CabinLetter)[0]
df['CabinNumber'] = df.Cabin.map(lambda x: ''.join(re.findall('[0-9]', x))).map(lambda x: int(x) + 1 if x != ''else 1)
df = df.drop('Cabin', axis=1)

# 归一化
scale = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(df.Age)
df['Age_Scaled'] = scale.transform(df.Age)

# 分段
df['Fare_bin_id'] = pd.qcut(df.Fare, 4)
df.Fare_bin_id = pd.factorize(df.Fare_bin_id)[0] + 1

print df.info()


