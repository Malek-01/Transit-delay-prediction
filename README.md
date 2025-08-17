# Transit-delay-prediction


This work demonstrates a practical example of leveraging and merging open data sources to accurately predict transit delays. The corresponding paper, titled Predicting Rail Transit Delays with Machine Learning: Leveraging Open Data Sources' is currently in progress.

The main code showcases the utilization of various machine learning methods, including support vector machines (with feature selection and parameter tuning), random forest, gradient boosting, deep neural networks, and linear regression, for transit delay prediction.

The case study focuses on the Canberra Rail System, and the necessary data can be found in the 'Data' folder. Additionally, the release includes the 'Large data' folder, which contains the substantial 'Trip_Update' and 'Vehicle_Update' files (https://github.com/Malek-01/Transit-delay-prediction/releases/tag/Transit).

The 'plot' folder provides graphical representations comparing real and predicted values for different algorithms. It also includes a side-by-side bar graph illustrating three evaluation metrics: mean absolute error (MAE), root mean square error (RMSE), and symmetric mean absolute percentage error (sMAPE).

For the merging process, the 'merging' folder contains relevant files, and the resulting file, 'data.xlsx,' is available in the release.

The codes to be executed are located in the main code, namely the following files: SVM-FS.py, GB.py, RF.py, DNN.py, and LR.py.

To enhance the assessment of features influencing delays, two additional case studies have been included, notably for Dublin and Sydney cities. The information about them are available in Dublin case study and Sydney case study, respectively.

## Contact

For any questions or inquiries, please contact the main contributor:

- Malek Sarhani: malek.sarhani.aui@gmail.com or m.sarhani@aui.ma

## Disclaimer

### Computational Results
The computational results in the paper were obtained based on initial experiments. During further review, it was identified that:
1. **Support Vector Regression (SVR):** 
   The code in the paper mistakenly fit the regression model on raw data instead of the training data. Correcting this issue may result in different values compared to those reported in the paper.
2. **Data Completeness:**
   The Canberra dataset originally contained several instances of missing data. We aimed to handle these missing values using appropriate preprocessing techniques.

**Note:** Due to that data completeness issue, the public sharing of the dataset has been temporarily suspended until this is fully addressed.

### Reproducibility
While we have shared the code and data, differences in results may arise due to factors such as:
- Variations in data preprocessing or imputation methods.
- Changes in software libraries or versions.
- Random state initialization.

If you have further questions, please reach out to us for clarification.

