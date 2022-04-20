# CMPE255_TEAM10 []

**Team Members**

* Yoonjung Choi
* Abhiteja Mandava
* Ishaan Bhalla
* Sakruthi Avirineni


## Introduction Dataset

**Source Dataset**

[Source Link](https://www.openml.org/search?type=data&status=active&id=43687&sort=runs)

* number of instance: 39998
* number of features: 12
* number of missing values: 

**Feature**

| Feature Name                                              |  TYPE. | MissigValue|
|-----------------------------------------------------------|--------|------------|
| Age_At_The_Time_Of_Mammography INTEGER                    | INTEGER|            | 
| Radiologists_Assessment STRING                            | STRING |            | 
| Is_Binary_Indicator_Of_Cancer_Diagnosis {True, False}     | BINARY |            | 
| Comparison_Mammogram_From_Mammography STRING              | STRING |   4680     |
| Patients_BI_RADS_Breast_Density STRING                    | STRING |            | 
| Family_History_Of_Breast_Cancer STRING                    | STRING |    288     | 
| Current_Use_Of_Hormone_Therapy STRING                     | STRING |   1772     | 
| Binary_Indicator STRING                                   | STRING |    578     | 
| History_Of_Breast_Biopsy STRING                           | STRING |    815     |
| Is_Film_Or_Digital_Mammogram {True, False}                | BINARY |            | 
| Cancer_Type STRING                                        | STRING |            | 
| Body_Mass_Index STRING                                    | STRING |   23208    |
| Patients_Study_ID INTEGER                                 | INTEGER|            |


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

