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
df3 = pd.read_csv('/Data/stop_times.txt', low_memory=False)
df4 = pd.read_csv('/Data/Stops.txt', low_memory=False)
df1 = pd.read_csv('/Data/Vehicle_Update.csv', low_memory=False)
df2 = pd.read_csv('/Data/Trip_Update.csv', low_memory=False)
df5 = pd.read_excel('/Data/Weather.xlsx', sheet_name=0)
df6 = pd.read_excel('/Data/Patronage_Proceeded.xlsx', sheet_name=0)

# Merge datasets
try:
    df3['trip_id'] = df3['trip_id'].astype(str)
    data = (
        pd.merge(df2, df1, on=['timestamp', 'tripID', 'stopSequence'])
        .merge(df3, left_on=['tripID', 'stopSequence'], right_on=['trip_id', 'stop_sequence'])
    )
    data = pd.merge(data, df5, left_on=['Dates'], right_on=['Date'], how='inner')
except Exception as e:
    print(f"Error merging datasets: {e}")

# Feature engineering
try:
    data['timestamp'] = pd.to_datetime(data['timestamp'], errors='coerce')
    data['Hour'] = data['timestamp'].dt.hour
    data['weekday'] = data['timestamp'].dt.dayofweek
except Exception as e:
    print(f"Error during feature engineering: {e}")

# Handle patronage intervals
try:
    df6.index = pd.IntervalIndex.from_arrays(df6['Init (weekday+time)'], df6['End (weekday+time)'], closed='both')
    data['weekday-time'] = data['HR'].apply(lambda x: x if x in df6.index else None)
except Exception as e:
    print(f"Error handling patronage intervals: {e}")

# Encode categorical variables
label_encoder = preprocessing.LabelEncoder()
categorical_cols = ['trip_id', 'stop_id']  # Add other relevant categorical columns
for col in categorical_cols:
    try:
        data[col] = label_encoder.fit_transform(data[col].astype(str))
    except Exception as e:
        print(f"Error encoding column {col}: {e}")

# Define features and target variable
feature_cols1 = [1, 7, 11, 16, 17, 18, 22, 23, 24, 25, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]
X = data.iloc[:, feature_cols1]
y = data.iloc[:, 2]

# Train-test split
train_pct_index = int(0.8 * len(data))
X_train, X_test = X[:train_pct_index], X[train_pct_index:]
y_train, y_test = y[:train_pct_index], y[train_pct_index:]

# Model training and evaluation
reg = RandomForestRegressor(max_depth=100, random_state=0)
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)

# Metrics
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

plt.ylabel('Delay (in seconds)')
plt.title('Prediction of Transit Delays')
plt.legend()
plt.show()

# Save predictions to Excel
y_pred_df = pd.DataFrame(y_pred, columns=['Predicted'])
y_pred_df.to_excel('Predicted_values.xlsx', index=False, sheet_name='RF')
