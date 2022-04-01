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
from sklearn.ensemble import RandomForestRegressor

def smape(A, F):
    return 100/len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))

df3 = pd.read_csv('C:/Users/Malek/Desktop/Policy Transport/Github/Data/stop_times.txt', error_bad_lines=False, low_memory=False)
df4 = pd.read_csv('C:/Users/Malek/Desktop/Policy Transport/Github//Data/Stops.txt', error_bad_lines=False, low_memory=False)
df1 = pd.read_csv('C:/Users/Malek/Desktop/Policy Transport/Github//Data/Vehicle_Update.csv', error_bad_lines=False, low_memory=False)
df2 = pd.read_csv('C:/Users/Malek/Desktop/Policy Transport/Github//Data/Trip_Update.csv', error_bad_lines=False, low_memory=False)

df5 = pd.read_excel('C:/Users/Malek/Desktop/Policy Transport/Github/Data/Weather.xlsx',sheet_name=0)
df6 = pd.read_excel('C:/Users/Malek/Desktop/Policy Transport/Github/Data/Patronage_Proceeded.xlsx',sheet_name=0)

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

data.to_excel("data.xlsx") 
#data= pd.read_excel("data.xlsx")
#def get_dataframe(data):
#    df = data
