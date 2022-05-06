# CMPE255_TEAM10 [Natural Language Processing With Disaster Tweets]

**Team Members**

* Yoonjung Choi
* Abhiteja Mandava
* Ishaan Bhalla
* Sakruthi Avirineni

[Paper](https://www.overleaf.com/read/gnxxgdkfggzs)

## Introduction
Social Network Services have become not only an important sources of emergency information during disaster but also a medium for expressing immediate responses of warning, evacuation or rescue providing immediate assistance, assessing damage, continuing assistance and the immediate restoration or construction of infrastructure. As a result, predicting context of SNS is a crucial concern in our society. Also, more agencies want to monitor or track Twitter intelligently by using technologies. This paper can be utilized to track, monitor and predict disasters from content of SNS and would help making prediction model. 

Twitter is one of the popular communication medium and people would start to annouce information via tweeter in disaster situation. This paper on Tweetter analysis is about prediction problem on whether a person's words are actually telling a disaster.

![image](https://storage.googleapis.com/kaggle-media/competitions/tweet_screenshot.png)

Here is example.
If someone says that 'On plus side LOOK AT THE SKY LIST NIGHT IT WAS ABLAZE', We can understand it does not mean 'disaster' but it means metaphorically something. However, It is not clear to Machine. 

Thus, our TEAM10 will investigate what techniques are for NLP and explore them. 

Other submission of Kaggle used similar preprocessing but used single model or even in case of ensemble, it trained with the same data set.
**However, we tried to find best combination of feature vectors and model and made custom voting classifier combined each combination, e.g. SVM with CountVectorizer, Decision Tree with Tf-Idf, Logistic Regression with CountVectorizer.**

## Data set
**Source**

[Source Link](https://www.kaggle.com/competitions/nlp-getting-started/data)

* number of instance: 7613
* number of features: 5

**Feature**

| Feature   |  Dtype | Descrition                  |
|-----------|--------|-----------------------------|
| id        | int64  |                             |
| keyword   | object | 39 null-values              |
| location. | object | 2533 null-valus             |
| text      | object | tweetter content            |

**label**

| Feature   |  Dtype | Descrition                  |
|-----------|--------|-----------------------------|
| target    | int64  | 0:non-disaster, 1:disaster  |

**This dataset have 'target' label having 0 or 1**

## Problem
Prediction problem on whether a person's words are actually telling a disaster.

This is categorized by Supervised Learning, Binary classification Problem and Natural Language Processing.

![image](https://user-images.githubusercontent.com/20979517/164575693-d0ee93c4-d68e-4697-a108-d616754b6eed.png)


We will use **Logistic Regression, SVM, Decision Tree, LSTM** to solve this classification problem.

* Logistic Regression

Logistic Regression is easier to implement, interpret, and very efficient to train.	

Logistic Regression is good accuracy for many simple data sets and it performs well when the dataset is linearly separable.

* SVM
 
SVM works relatively well when there is a clear margin of separation between classes.

SVM is more effective in high dimensional spaces.

SVM is effective in cases where the number of dimensions is greater than the number of samples.

SVM is relatively memory efficient.

* Decision Tree

Decision Tree algorithm can be used to solve both regression and classification problems.

Decision Tree can be used to handle both numerical and categorical data.

* LSTM 

Long Short Term Memory (LSTM) network were used as deep learning models for automatic feature extraction from tweet texts.

LSTM network is also proposed using this hybrid feature to classify tweets into rumor and non-rumor classes.


## WorkFlow

**1) Data Exploration**
* Loading dataset and understanding data

**2) Data Cleaning**
* Change all characters to lowercase
* Makes sure to remove URL, HTML, Emojis, ASCII and Punctuation. 

**3) Data Preprocessing Part1 Using [NLTK](https://www.nltk.org/index.html)**:
* Tokenize
* Remove Stopwords(Common words, example: are)
* PorterStemmer (Stemming is the important in NLP, example: asked -> ask)
* WordNetLemmatizer (example: peopl => people) -> We decided to use this, not stemming, due to performance.

**4) Data Preprocessing Part2 to transform text to numerical value**
* apply CountVector, visualize count info
* apply TF-IDF, visualize count info
* apply Word2Vec, visualize vectors based on similiar and not opposite words
* apply Word2Vec with PCA, visualize principal components.

**5) Data Split**
* split each type of feature

**6) Modeling**:
* Build training model
* Train each type of feature vectors in model
* Find best tuning values of model
* Make sure to save all information(F1 Score, Precision, Recall, Accuracy)

**7) Custom Ensemble**
* create best funing values of model
* create ensemble model with selected feature vector
* Make sure to save all information(F1 Score, Precision, Recall, Accuracy)

**8) Visualization**
* Visualize results
* Compare each models to find best results.
