# CMPE255_TEAM10 []

**Team Members**

Yoonjung Choi
Abhiteja Mandava
Ishaan Bhalla
Sakruthi Avirineni


## Introduction Dataset

**Source Dataset**
https://www.openml.org/search?type=data&status=active&id=43687&sort=runs

* number of instance: 39998
* number of features: 12

**Feature**
Age_At_The_Time_Of_Mammography INTEGER

Radiologists_Assessment STRING

Is_Binary_Indicator_Of_Cancer_Diagnosis {True, False}

Comparison_Mammogram_From_Mammography STRING

Patients_BI_RADS_Breast_Density STRING

Family_History_Of_Breast_Cancer STRING

Current_Use_Of_Hormone_Therapy STRING

Binary_Indicator STRING

History_Of_Breast_Biopsy STRING

Is_Film_Or_Digital_Mammogram {True, False}

Cancer_Type STRING

Body_Mass_Index STRING

Patients_Study_ID INTEGER

## Problem
Binary or multiple classification Problem (Supervised Learning)
This dataset have two indicators

* Is_Binary_Indicator_Of_Cancer_Diagnosis {True, False}
* Cancer_Type STRING

We will use **Logistic Regression, SVM(linear & Kernal), Decision Tree** to solve this classification problem.

**Preprocessing & Understanding Dataset**
* Change all to numeric type.
* Visualize correlations among features
* Visualize any other relationship between features or labels
* Identify mssing Values and replace Mean, Zero or others.    
* Identify label.

**Modeling**:
   
* Split train and test set
* Build training model
* Train the data set in model
* Find best tuning values of model

**Prediction**:
* Predict test date 
* Obtain accuracy - F1 Score, Precision, Recall...
* Visualize results
* Compare each models to find best results.

