# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 20:10:45 2021
@author: Malek
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.preprocessing 
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import linear_model
from sklearn import metrics
from sklearn.metrics import r2_score
import time
from datetime import datetime
label_encoder = preprocessing.LabelEncoder()
from sklearn.ensemble import RandomForestRegressor

def smape(A, F):
    return 100/len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))

data = pd.read_excel('data1.xlsx',sheet_name=0)
Weather = pd.read_excel('Weather-Dublin2.xlsx',sheet_name=0)

print("aaaa")
for col in data.columns:
    print(col)
data.drop(data.index[data.iloc[:,3] == "1970-01-01"], inplace = True)
data.drop(data.index[data.iloc[:,2] == "47:00:00"], inplace = True)
data.drop(data.index[data.iloc[:,2] == "47:02:00"], inplace = True)
data.drop(data.index[data.iloc[:,2] == "47:05:00"], inplace = True)
data.drop(data.index[data.iloc[:,2] == "24:00:00"], inplace = True)
data.drop(data.index[data.iloc[:,2] == "24:05:00"], inplace = True)

data.iloc[:,7] = pd.to_datetime(data.iloc[:,7], format="%Y%m%d ")
data.iloc[:,2] = pd.to_datetime(data.iloc[:,2], format="%H:%M:%S")

data['Hour'] = pd.to_datetime(data.iloc[:,6]).dt.hour
data['weekday'] = data.iloc[:,7].dt.dayofweek


data = pd.merge(data, Weather, left_on=['trip.start_date'],  right_on=['Date'])
data.to_excel("data4.xlsx")    
feature_cols1 = [1, 2, 3,  4, 5, 6, 7, 8, 10,  11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]

data.iloc[:,0] = label_encoder.fit_transform(data.iloc[:,0]).astype('float64')
data.iloc[:,1] = label_encoder.fit_transform(data.iloc[:,1]).astype('float64')
data.iloc[:,2] = label_encoder.fit_transform(data.iloc[:,2]).astype('float64')
data.iloc[:,3] = label_encoder.fit_transform(data.iloc[:,2]).astype('float64')

data.iloc[:,4] = label_encoder.fit_transform(data.iloc[:,4]).astype('float64')

data.iloc[:,5] = label_encoder.fit_transform(data.iloc[:,5]).astype('float64')
data.iloc[:,6] = label_encoder.fit_transform(data.iloc[:,6]).astype('float64')
#data.iloc[:,7] = label_encoder.fit_transform(data.iloc[:,7]).astype('float64')
data.iloc[:,8] = label_encoder.fit_transform(data.iloc[:,8]).astype('float64')
data.iloc[:,9] = label_encoder.fit_transform(data.iloc[:,9]).astype('float64')
data.iloc[:,10] = label_encoder.fit_transform(data.iloc[:,10]).astype('float64')
data.iloc[:,11] = label_encoder.fit_transform(data.iloc[:,11]).astype('float64')
data.iloc[:,12] = label_encoder.fit_transform(data.iloc[:,12]).astype('float64')
data.iloc[:,13] = label_encoder.fit_transform(data.iloc[:,13]).astype('float64')

data.iloc[:,7] = data.iloc[:,7].astype('datetime64').values.astype(float)
data.iloc[:,12] = label_encoder.fit_transform(data.iloc[:,12]).astype('float64')
data.iloc[:,13] = label_encoder.fit_transform(data.iloc[:,13]).astype('float64')
data.iloc[:,14] = label_encoder.fit_transform(data.iloc[:,14]).astype('float64')
data.iloc[:,15] = label_encoder.fit_transform(data.iloc[:,15]).astype('float64')

print("feature_cols1")
X = data.iloc[:, feature_cols1]

for col in X.columns:
    print(col)




y = data.iloc[:,9]

train_pct_index = int(0.8 * len(data.iloc[:, 11]))
X_train, X_test = X[:train_pct_index], X[train_pct_index:]
y_train, y_test = y[:train_pct_index], y[train_pct_index:]

feature_cols1 = label_encoder.fit_transform(feature_cols1).astype('float64')
y_test = y_test.apply(pd.to_numeric, errors='coerce')
reg = GradientBoostingRegressor(random_state=1)
reg.fit(X_train, y_train)
y_pred1 = reg.predict(X_test)

print('R:', r2_score(y_test, y_pred1)) 
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred1))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred1))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred1)))
print('sMAPE:', smape(np.transpose(y_pred1),  y_test)) 
