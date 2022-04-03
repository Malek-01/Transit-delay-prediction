# Transit-delay-prediction


This work shows a practical example of how to leverage and merge open data sources to provide accurate prediction of transit delays. 
The corresponding paper is titled "Predicting Transit Delays with Machine Learning: How to Exploit Open Data Sources" (in progress).

The main code demonstrates how to use different machine learning methods, namely support vector machines (with and without feature selection (FS), random forest (RF), gradient boosting (GB), deep neural networks (DNN ) and Linear Regression (LR) to predict transit delays. 

In this work, we have takee the Canberra Lig release (ht Rail System  as a case study. The "Data" folder contains the data used. We note that the large "Trip_Update" and "Vehicle_Update" files are available in the release ("Large data").

The "plot" folder shows a partial graphical representation of the difference between real  and predicted values for the various algorithms, in addition to a side-by-side bar graph illustrating the difference between three metrics, namely MAE, RMSE and sMAPE. 

The "merging" folder contains the merging part and the resulting file "data.xlsx" is available in the release.
