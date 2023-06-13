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
import shap 
import eli5
from eli5.sklearn import PermutationImportance
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR


def smape(A, F):
    return 100/len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))

X = pd.read_excel('C:/Users/Malek/Desktop/Policy Transport/Github/Data/Features.xlsx',sheet_name=0)
y = pd.read_excel('C:/Users/Malek/Desktop/Policy Transport/Github/Data/Label.xlsx',sheet_name=0)
#X = X.iloc[1:100,:]
#y = y.iloc[1:100]

print('feature_cols1')
print(X.head())
train_pct_index = int(0.8 * len(X.iloc[:, 2]))
X_train, X_test = X[:train_pct_index], X[train_pct_index:]
y_train, y_test = y[:train_pct_index], y[train_pct_index:]
print(y.head())
reg = GradientBoostingRegressor(random_state=1)
#reg = SVR(kernel='rbf', C=100, gamma='auto', epsilon=0.1) # Linear Kernel
reg.fit(X, y)
#explainer = shap.KernelExplainer(reg.predict, X_train,  nsamples=100)

explainer = shap.Explainer(reg)
#shap_test = explainer.shap_values(X_test)
shap_test = explainer(X_test)
print(f"Shap values length: {len(shap_test)}\n")
print(f"Sample shap value:\n{shap_test[0]}")
shap.plots.bar(shap_test)
#shap.plots.bar(shap_test, max_display=22)

#shap.summary_plot(shap_test)
#shap.summary_plot(shap_test, plot_type='violin')


#plt.savefig('prediction.png')
#, low_memory=False