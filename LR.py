# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 20:10:45 2021

@author: Malek
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing, metrics, linear_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from datetime import datetime

# SMAPE Function
def smape(A, F):
    return 100 / len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))

# Load datasets
df3 = pd.read_csv('/Data/stop_times.txt', low_memory=False)
df4 = pd.read_csv('/Data/Stops.txt', low_memory=False)
df1 = pd.read_csv('/Data/Vehicle_Update.csv', low_memory=False)
df2 = pd.read_csv('/Data/Trip_Update.csv', low_memory=False)
df5 = pd.read_excel('/Data/Weather.xlsx', sheet_name=0)
df6 = pd.read_excel('/Data/Patronage_Proceeded.xlsx', sheet_name=0)

# Data preparation and merging
df3['trip_id'] = df3['trip_id'].astype(str)
data = (
    pd.merge(df2, df1, on=['timestamp', 'tripID', 'stopSequence'])
    .merge(df3, left_on=['tripID', 'stopSequence'], right_on=['trip_id', 'stop_sequence'])
)

data.iloc[:, 7] = pd.to_datetime(data.iloc[:, 7], format="%m/%d/%Y %I:%M:%S %p")
data['Dates'] = pd.to_datetime(data.iloc[:, 7]).dt.date
data['Dates'] = pd.to_datetime(data['Dates'])
data['Hour'] = data.iloc[:, 7].dt.hour
data['Minute'] = data.iloc[:, 7].dt.minute
data['weekday'] = data.iloc[:, 7].dt.dayofweek
data["HR"] = data["weekday"].astype(str) + data["Hour"].astype(str) + data["Minute"].astype(str)
data["HR"] = data["HR"].astype(int)

# Merge weather data
data = pd.merge(data, df5, left_on=['Dates'], right_on=['Date'])

# Handling patronage intervals
df6.index = pd.IntervalIndex.from_arrays(df6['Init (weekday+time)'], df6['End (weekday+time)'], closed='both')
data['weekday-time'] = data['HR'].apply(lambda x: x if x in df6.index else None)

# Sort data
data = data.sort_values(by='Dates', ascending=True)

# Feature encoding
label_encoder = preprocessing.LabelEncoder()
data.iloc[:, 9] = data.iloc[:, 9].astype(str).str.replace(',', '')
data.iloc[:, 1] = label_encoder.fit_transform(data.iloc[:, 1]).astype('float64')
for i in range(3, 8):
    data.iloc[:, i] = label_encoder.fit_transform(data.iloc[:, i]).astype('float64')
for i in range(10, data.shape[1]):
    data.iloc[:, i] = label_encoder.fit_transform(data.iloc[:, i]).astype('float64')

# Feature selection and target variable
feature_cols1 = [1, 7, 11, 16, 17, 18, 22, 23, 24, 25, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
X = data.iloc[:, feature_cols1]
y = data.iloc[:, 2]

# Train-test split
train_pct_index = int(0.8 * len(y))
X_train, X_test = X[:train_pct_index], X[train_pct_index:]
y_train, y_test = y[:train_pct_index], y[train_pct_index:]

# Linear Regression
reg = LinearRegression()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)

# Evaluation metrics
print("R2 Score:", r2_score(y_test, y_pred))
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("SMAPE:", smape(np.array(y_test), np.array(y_pred)))

# Visualization
plt.figure(figsize=(16, 8))
plt.plot(range(len(y_pred)), y_pred, 'g-', label='Predicted')
plt.plot(range(len(y_test)), y_test, 'b-', label='Actual')
plt.xlabel('Sample')
plt.ylabel('Delay (in seconds)')
plt.title('Prediction of Transit Delays')
plt.legend()
plt.show()

# Save predictions to Excel
y_pred_df = pd.DataFrame(y_pred, columns=['Predicted'])
y_pred_df.to_excel('Predicted_values.xlsx', index=False, sheet_name='LR')
