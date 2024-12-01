# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 20:10:45 2021

@author: Malek
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing, metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from datetime import datetime

# SMAPE Function
def smape(A, F):
    return 100 / len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))

# Load datasets
data = pd.read_excel('Sydney2.xlsx', sheet_name=0)
weather = pd.read_excel('Weather-infor.xlsx', sheet_name=0)

# Merge data with weather information
data = pd.merge(data, weather, left_on=['Vehicle Trip Start Date'], right_on=['Date'])
data.to_excel("data3.xlsx")  

# Define feature columns
feature_cols1 = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]

# Convert relevant columns to datetime and extract features
data.iloc[:, 7] = pd.to_datetime(data.iloc[:, 7], format="%Y%m%d")
data.iloc[:, 6] = pd.to_datetime(data.iloc[:, 6], format="%H:%M:%S")
data['Hour'] = pd.to_datetime(data.iloc[:, 6]).dt.hour
data['weekday'] = data.iloc[:, 7].dt.dayofweek

# Label encoding for categorical features
label_encoder = preprocessing.LabelEncoder()
categorical_columns = [0, 4, 5, 6, 7, 8, 9, 10, 11, 12]
for col in categorical_columns:
    data.iloc[:, col] = label_encoder.fit_transform(data.iloc[:, col]).astype('float64')

# Prepare features (X) and target (y)
X = data.iloc[:, feature_cols1]
y = data.iloc[:, 1]

# Train-test split
train_pct_index = int(0.8 * len(data))
X_train, X_test = X[:train_pct_index], X[train_pct_index:]
y_train, y_test = y[:train_pct_index], y[train_pct_index:]

# Random Forest Regressor
reg = RandomForestRegressor(max_depth=100, random_state=0)
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)

# Evaluation metrics
print('R2 Score:', r2_score(y_test, y_pred))
print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, y_pred)))
print('SMAPE:', smape(np.array(y_test), np.array(y_pred)))

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
y_pred_df.to_excel('Predicted_values.xlsx', index=False, sheet_name='RF')
