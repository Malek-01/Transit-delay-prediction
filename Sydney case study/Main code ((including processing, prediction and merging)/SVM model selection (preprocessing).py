# -*- coding: utf-8 -*-
"""

@author: Malek
"""
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.preprocessing 
from sklearn import preprocessing
from sklearn.svm import SVR
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn import metrics
from sklearn.metrics import r2_score
import time
from datetime import datetime
label_encoder = preprocessing.LabelEncoder()
from sklearn.ensemble import RandomForestRegressor

def smape(A, F):
    return 100/len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))

data = pd.read_excel('Sydney2.xlsx',sheet_name=0)
Weather = pd.read_excel('Weather-infor.xlsx',sheet_name=0)


                        


data = pd.merge(data, Weather, left_on=['Vehicle Trip Start Date'],  right_on=['Date'])
data.to_excel("data3.xlsx")    
feature_cols1 = [0, 2,3,  4, 5, 6, 7, 8, 9, 10,  11, 12, 13, 14, 15, 16, 17, 18]
print("aaaa")
#print(Weather)
for col in data.columns:
    print(col)
class_cols = [2]

data.iloc[:,7] = pd.to_datetime(data.iloc[:,7], format="%Y%m%d ")
data.iloc[:,6] = pd.to_datetime(data.iloc[:,6], format="%H:%M:%S")
data['Hour'] = pd.to_datetime(data.iloc[:,6]).dt.hour
#data['Minute'] = pd.to_datetime(data.iloc[:,7]).dt.minute
data['weekday'] = data.iloc[:,7].dt.dayofweek
#data.iloc[:,9] = data.iloc[:,9].astype(str).str.replace(',', '')
data.iloc[:,0] = label_encoder.fit_transform(data.iloc[:,0]).astype('float64')
data.iloc[:,4] = label_encoder.fit_transform(data.iloc[:,4]).astype('float64')
data.iloc[:,5] = label_encoder.fit_transform(data.iloc[:,5]).astype('float64')
data.iloc[:,6] = label_encoder.fit_transform(data.iloc[:,6]).astype('float64')
data.iloc[:,7] = label_encoder.fit_transform(data.iloc[:,7]).astype('float64')
data.iloc[:,8] = label_encoder.fit_transform(data.iloc[:,8]).astype('float64')
data.iloc[:,9] = label_encoder.fit_transform(data.iloc[:,9]).astype('float64')
data.iloc[:,10] = label_encoder.fit_transform(data.iloc[:,10]).astype('float64')
data.iloc[:,11] = label_encoder.fit_transform(data.iloc[:,11]).astype('float64')
data.iloc[:,12] = label_encoder.fit_transform(data.iloc[:,12]).astype('float64')


print("feature_cols1")
X = data.iloc[:, feature_cols1]

for col in X.columns:
    print(col)




y = data.iloc[:,1]

train_pct_index = int(0.8 * len(data.iloc[:, 2]))
X_train, X_test = X[:train_pct_index], X[train_pct_index:]
y_train, y_test = y[:train_pct_index], y[train_pct_index:]

print('feature_cols1')
print(X.head())

print(y.head())
feature_cols1 = label_encoder.fit_transform(feature_cols1).astype('float64')
y_test = y_test.apply(pd.to_numeric, errors='coerce')

print(y.head())
parameters = [{'C': [1, 10, 100, 1000], 
                'kernel': ['sigmoid','rbf','poly'], 
                'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]
feature_cols1 = label_encoder.fit_transform(feature_cols1).astype('float64')
model = SVR(kernel='rbf', C=100, gamma='auto', epsilon=0.1) # Linear Kernel
cv = RepeatedKFold(n_splits=5, n_repeats=1, random_state=1)
search = GridSearchCV(model, parameters, scoring='neg_mean_absolute_error', n_jobs=-1, cv=cv)
result = search.fit(X, y)
print('Best Score: %s' % result.best_score_)
print('Best Hyperparameters: %s' % result.best_params_)

