import matplotlib.pyplot as plt
import numpy as np
import re

# Average stock prize in last 5 years
SVMFS = [0.025, 0.258, 0.0034]
SVM = [0.025, 0.258, 0.0034]

GB = [0.848, 0.8239, 0.1542]
DNN = [1, 1, 1]
RF = [0.902, 0.7645, 0.02617]
LR = [0.9306, 0.8861, 0.2333]


year = [" " + " " + " " + " " + " " +" " + " " + " " + " " + " " + " " + " " + " " + " " + " " + " " + " " + " " + " " +  '     MAE', " " + " " + " " + " " + " " +" " + " " + " " + " " + " " + " " + " " + " " + " " + " " + " " + " " + " " + " " + 'RMSE', " " + " " + " " + " " + " " +" " + " " + " " + " " + " " + " " + " " + " " + " " + " " + " " + " " + " " + " " + 'sMAPE' ]

# We cannot add width to year so we create another list
indices = np.arange(len(year))

width = 0.15

# Plotting
barplt = plt.bar(indices, SVMFS, width=width, label='SVM-FS')

# Offsetting by width to shift the bars to the right
normplt = plt.bar(indices + width, SVM, width=width, label='SVM')
normplt2 = plt.bar(indices + width+ width , GB, width=width, label='GB')
normplt3 = plt.bar(indices + width+ width + width, DNN, width=width, label='DNN')
normplt4 = plt.bar(indices + width+ width + width+ width , RF, width=width, label='RF')
normplt5 = plt.bar(indices + width+ width + width+ width+ width , LR, width=width, label='LR')

plt.legend(handles=[barplt,normplt,normplt2,normplt3,normplt4,normplt5])

# Displaying year on top of indices
plt.xticks(ticks=indices, labels=year)


plt.xlabel("Measure")
plt.ylabel("Error")
plt.title("MAE, RMSE and sMAPE results of all algorithms")

plt.savefig("plot.png")
plt.show()
