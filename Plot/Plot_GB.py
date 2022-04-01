# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 17:03:55 2022

@author: Malek
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

import seaborn as sns
import sklearn.preprocessing 

y_test = pd.read_excel('Predicted_values.xlsx',sheet_name='Real_values')
y_pred1 = pd.read_excel('Predicted_values.xlsx',sheet_name='GB')


#l = list(range(len(y_pred1)))
l = list(range(len(y_pred1[1:1000])))
figure(figsize=(8, 16), dpi=80)
#figure(figsize=(80, 6), dpi=80)


plt.plot(l, y_test[1:1000], 'g-', label = 'GB')
plt.plot(l, y_pred1[1:1000], 'b-', label = 'Real')
plt.legend(loc='best')
plt.savefig('Plot_GB.png')


#plt.axis([0, 1000, -10, 10])

