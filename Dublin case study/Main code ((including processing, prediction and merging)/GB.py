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
#print(Weather)
for col in data.columns:
    print(col)
#data = pd.merge(df2, df1, on=['timestamp', 'tripID', 'stopSequence']).merge(df3, left_on=['tripID', 'stopSequence'], right_on=['trip_id', 'stop_sequence'])
data.drop(data.index[data.iloc[:,3] == "1970-01-01"], inplace = True)
data.drop(data.index[data.iloc[:,2] == "47:00:00"], inplace = True)
data.drop(data.index[data.iloc[:,2] == "47:02:00"], inplace = True)
data.drop(data.index[data.iloc[:,2] == "47:05:00"], inplace = True)
data.drop(data.index[data.iloc[:,2] == "24:00:00"], inplace = True)
data.drop(data.index[data.iloc[:,2] == "24:05:00"], inplace = True)

#data.iloc[:,3] = pd.to_datetime(data.iloc[:,3], format="%y%M%d")
data.iloc[:,7] = pd.to_datetime(data.iloc[:,7], format="%Y%m%d ")
data.iloc[:,2] = pd.to_datetime(data.iloc[:,2], format="%H:%M:%S")

#data['Dates'] = pd.to_datetime(data.iloc[:,7]).dt.date
#data['Dates'] = pd.to_datetime(data['Dates'])
#data['Time'] = pd.to_datetime(data.iloc[:,7]).dt.time
data['Hour'] = pd.to_datetime(data.iloc[:,6]).dt.hour
#data['Minute'] = pd.to_datetime(data.iloc[:,7]).dt.minute
data['weekday'] = data.iloc[:,7].dt.dayofweek
#data["HR"] =  data["weekday"].astype(str)  + data["Hour"].astype(str)  + data["Minute"].astype(str) 
#data["HR"]  = data["HR"].astype(int)

#print("Drop")
#print(data["HR"])

#print(data['Dates'] == datetime(1970, 1, 1))
#data = pd.merge(data, df5, left_on=['Dates'], right_on=['Date'])


#df6.index = pd.IntervalIndex.from_arrays(df6['Init (weekday+time)'],df6['End (weekday+time)'],closed='both')
#data['weekday-time'] = data['HR'].apply(lambda x : data.iloc[data.index.get_loc(x)]['HR'])
#data = data.sort_values(by='Dates',ascending=True)
#data.to_excel("data3.xlsx") 

#data['Dates2'] = pd.to_datetime(data.iloc[:,4]).dt.date
#data = data.drop(data.index[data['Dates2'] == pd.Timestamp(1970,1,1)])


#data= pd.read_excel("data.xlsx")
#def get_dataframe(data):
#    df = data

data = pd.merge(data, Weather, left_on=['trip.start_date'],  right_on=['Date'])
data.to_excel("data4.xlsx")    
feature_cols1 = [1, 2, 3,  4, 5, 6, 7, 8, 10,  11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
#class_cols = [2]
#data.iloc[:,9] = data.iloc[:,9].astype(str).str.replace(',', '')
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

#date_str = "6:29:2006"
data.iloc[:,7] = data.iloc[:,7].astype('datetime64').values.astype(float)
#data.iloc[:,8] = pd.to_numeric(pd.to_datetime(data.iloc[:,8]))
data.iloc[:,12] = label_encoder.fit_transform(data.iloc[:,12]).astype('float64')
data.iloc[:,13] = label_encoder.fit_transform(data.iloc[:,13]).astype('float64')
data.iloc[:,14] = label_encoder.fit_transform(data.iloc[:,14]).astype('float64')
data.iloc[:,15] = label_encoder.fit_transform(data.iloc[:,15]).astype('float64')
#data.iloc[:,18] = label_encoder.fit_transform(data.iloc[:,18]).astype('float64')
#data.iloc[:,19] = label_encoder.fit_transform(data.iloc[:,19]).astype('float64')
#data.iloc[:,20] = label_encoder.fit_transform(data.iloc[:,20]).astype('float64')
#data.iloc[:,21] = label_encoder.fit_transform(data.iloc[:,21]).astype('float64')
#data.iloc[:,22] = label_encoder.fit_transform(data.iloc[:,22]).astype('float64')

#data.iloc[:,7]  = time.strptime(data.iloc[1,7], "%m:%d:%Y")
#data.iloc[:,7] = time.mktime(data.iloc[:,7])
#data.iloc[:,5] = data.iloc[:,5].str.replace('-', '').astype(float)

#data.iloc[:,5].fillna('', inplace=True)
#for i  in range(0, 22):
#    data.iloc[:,i] = data.iloc[:,i].fillna(10, inplace=True)
print("feature_cols1")
X = data.iloc[:, feature_cols1]
#pd.DataFrame(X).fillna()

for col in X.columns:
    print(col)




y = data.iloc[:,9]

train_pct_index = int(0.8 * len(data.iloc[:, 11]))
X_train, X_test = X[:train_pct_index], X[train_pct_index:]
y_train, y_test = y[:train_pct_index], y[train_pct_index:]


#print('feature_cols1')
#print(X.head())

#print(y.head())
feature_cols1 = label_encoder.fit_transform(feature_cols1).astype('float64')
y_test = y_test.apply(pd.to_numeric, errors='coerce')
reg = GradientBoostingRegressor(random_state=1)
reg.fit(X_train, y_train)
y_pred1 = reg.predict(X_test)
#print("y_pred1")
#print(y_pred1)
#print("y_test")
#print(y_test)
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