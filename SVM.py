# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 20:10:45 2021

@author: Malek
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.preprocessing 
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn import metrics
from sklearn.metrics import r2_score
import time
from datetime import datetime
label_encoder = preprocessing.LabelEncoder()
from sklearn.svm import SVR

def smape(A, F):
    return 100/len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))

df3 = pd.read_csv('/Data/stop_times.txt', error_bad_lines=False, low_memory=False)
df4 = pd.read_csv('/Data/Stops.txt', error_bad_lines=False, low_memory=False)
df1 = pd.read_csv('/Data/Vehicle_Update.csv', error_bad_lines=False, low_memory=False)
df2 = pd.read_csv('/Data/Trip_Update.csv', error_bad_lines=False, low_memory=False)

df5 = pd.read_excel('/Data/Weather.xlsx',sheet_name=0)
df6 = pd.read_excel('/Data/Patronage_Proceeded.xlsx',sheet_name=0)

df3['trip_id'] = df3['trip_id'].astype(str)
data = pd.merge(df2, df1, on=['timestamp', 'tripID', 'stopSequence']).merge(df3, left_on=['tripID', 'stopSequence'], right_on=['trip_id', 'stop_sequence'])
data.iloc[:,7] = pd.to_datetime(data.iloc[:,7], format="%m/%d/%Y %I:%M:%S %p")
data['Dates'] = pd.to_datetime(data.iloc[:,7]).dt.date
data['Dates'] = pd.to_datetime(data['Dates'])
data['Time'] = pd.to_datetime(data.iloc[:,7]).dt.time
data['Hour'] = pd.to_datetime(data.iloc[:,7]).dt.hour
data['Minute'] = pd.to_datetime(data.iloc[:,7]).dt.minute
data['weekday'] = data.iloc[:,7].dt.dayofweek
data["HR"] =  data["weekday"].astype(str)  + data["Hour"].astype(str)  + data["Minute"].astype(str) 
data["HR"]  = data["HR"].astype(int)

print("Drop")
print(data["HR"])

print(data['Dates'] == datetime(1970, 1, 1))
data = pd.merge(data, df5, left_on=['Dates'], right_on=['Date'])


df6.index = pd.IntervalIndex.from_arrays(df6['Init (weekday+time)'],df6['End (weekday+time)'],closed='both')
data['weekday-time'] = data['HR'].apply(lambda x : data.iloc[data.index.get_loc(x)]['HR'])
data = data.sort_values(by='Dates',ascending=True)
#data.to_excel("data3.xlsx") 

#data['Dates2'] = pd.to_datetime(data.iloc[:,4]).dt.date
#data = data.drop(data.index[data['Dates2'] == pd.Timestamp(1970,1,1)])

print(data['Dates'])

for col in data.columns:
    print(col)

print(data)

#data.to_excel("data.xlsx") 
#data= pd.read_excel("data.xlsx")
def get_dataframe(data):
    df = data

    
feature_cols1 = [1, 7,  11, 16, 17, 18, 22, 23, 24, 25, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
class_cols = [2]

data.iloc[:,9] = data.iloc[:,9].astype(str).str.replace(',', '')
data.iloc[:,1] = label_encoder.fit_transform(data.iloc[:,1]).astype('float64')
for i in range(3, 8):
    data.iloc[:,i] = label_encoder.fit_transform(data.iloc[:,i]).astype('float64')
for i in range(10, 48):
    data.iloc[:,i] = label_encoder.fit_transform(data.iloc[:,i]).astype('float64')

X = data.iloc[:, feature_cols1]

print("feature_cols1")
for col in X.columns:
    print(col)
y = data.iloc[:,2]

train_pct_index = int(0.8 * len(data.iloc[:, 2]))
X_train, X_test = X[:train_pct_index], X[train_pct_index:]
y_train, y_test = y[:train_pct_index], y[train_pct_index:]


print('feature_cols1')
print(X.head())

print(y.head())
feature_cols1 = label_encoder.fit_transform(feature_cols1).astype('float64')
y_test = y_test.apply(pd.to_numeric, errors='coerce')
clf = SVR(kernel='rbf', C=100, gamma='auto', epsilon=0.1) # Linear Kernel
clf.fit(X, y)
y_pred1 = reg.predict(X_test)
print("y_pred1")
print(y_pred1)
print("y_test")
print(y_test)
print('R:', r2_score(y_test, y_pred1)) 
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred1))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred1))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred1)))
print('sMAPE:', smape(np.transpose(y_pred1),  y_test)) 
l = list(range(len(y_pred1)))
plt.figure(figsize=(16, 8))
plt.plot(l, y_pred1, 'g-', label = 'LR')
plt.plot(l, y_test, 'b-', label = 'Real')
y_pred2 = pd.DataFrame (y_pred1)
with pd.ExcelWriter('Predicted_values.xlsx', engine='openpyxl', mode='a') as writer:
    y_pred2.to_excel(writer,sheet_name='RF')
plt.xlabel('Sample'); plt.ylabel('Delay (in seconds)'); plt.title('Prediction of early/late arrivals of transit in the test set')
plt.legend();
#plt.savefig('prediction.png')
#, low_memory=False
