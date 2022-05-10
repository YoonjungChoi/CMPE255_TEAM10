# CMPE255_TEAM10 [Natural Language Processing With Disaster Tweets]

**Team Members**

* Yoonjung Choi
* Abhiteja Mandava
* Ishaan Bhalla
* Sakruthi Avirineni

[Paper](https://www.overleaf.com/read/gnxxgdkfggzs)

## Introduction
Social Network Services have become not only an important source of emergency information during disaster but also a medium for expressing immediate responses of warning, evacuation or rescue, providing immediate assistance, assessing damage, continuing assistance and the immediate restoration or construction of infrastructure. As a result, predicting the context of SNS is a crucial concern in our society. Also, more agencies want to monitor or track Twitter intelligently by using technologies. This paper can be utilized to track, monitor and predict disasters from content of SNS and would help making prediction models.

Twitter is a popular communication medium and people would start to announce information via twitter in disaster situations. This paper on Twitter analysis is about prediction problems on whether a person's words are actually telling of a disaster.


![image](https://storage.googleapis.com/kaggle-media/competitions/tweet_screenshot.png)

Here is an example.
If someone says that 'On plus side LOOK AT THE SKY LAST NIGHT IT WAS ABLAZE', We can understand it does not mean 'disaster' but it means metaphorically something. However, It is not clear to the Machine. 

Thus, our TEAM10 will investigate what techniques are for NLP and explore them.
First step is about data cleaning, which means that removal of meaningless data, e.g. html tags, url, emojis.
Second step is about applying practical algorithms e.g. stop words, stemming to find the root of words.
Third step is about word embedding to transform text into numerical feature vectors, e.g. Tf-Idf, word2vec.

Other submissions of Kaggle used similar preprocessing and a single model or even in case of ensemble, it trained with the same data set.
**However, we tried to find the best combination of feature vectors and static models and made custom voting classifier combined each combination, e.g. SVM with CountVectorizer, Decision Tree with Tf-Idf, Logistic Regression with CountVectorizer. Also, we investigated dynamic model e.g. LSTM**


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


## Word Embedding

**CountVectorizer**

**TF-IDF(Term Frequency Inverse Document Frequency)**

**Word2Vec**

**Glove**


## Classifiers

We will use **Static Models(Logistic Regression, SVM, Decision Tree, Random Tree, XGboost) and Dynamic Model(LSTM)** to solve this binary classification problem.

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
* Loading data set
* Visualization data set
* **Expectations: understanding balanced or imbalanced label, feature selection**

**2) Data Cleaning**
* Change all characters to lowercase
* Makes sure to remove URL, HTML, Emojis, ASCII and Punctuation. 
* **Expectations: learning about what, why we should do before processing. **

**3) Data Preprocessing Part1 Using [NLTK](https://www.nltk.org/index.html)**:
* Tokenize
* Remove Stopwords(Common words, example: are)
* PorterStemmer (Stemming is the important in NLP, example: asked -> ask)
* WordNetLemmatizer (example: peopl => people) -> We decided to use this, not stemming, due to performance.
* **Expectations: learning about what, why algorithms can be applied for preprocessing**

**4) Data Preprocessing Part2 to transform text to numerical value**
* apply CountVector, visualize the number of count of words.
* apply TF-IDF, visualize the number of count of words.
* apply Word2Vec, visualize vectors based on similiar and not opposite words.
  -> we build a sentence embedding by averaging the values across all token embeddings output by Word2Vec.
* apply Word2Vec with PCA.
* **Expectations: Invesigating Word Embedding algorithms**

**5) Data Split**
* split each type of feature
  -> data set by using count vectorizer
  -> data set by using tf-idf
  -> data set by using word2vec(sentence embedding using average of vector)
  -> data set by using word2vec with PCA
* **Expectations: we can create 4 different data set having feature vectors**

**6) Modeling with Static Models**:
* build training model
  -> Logistic Regression, SVM, Decision Tree, RandomTree, XGboost(Static models)
* train each type of feature vectors in model
* find best parameters of model
* make sure to save all information(F1 Score, Precision, Recall, Accuracy)
* **Expectations: improvement of performance of model **


**7) Custom Ensemble**
* create model with parameters.
* create ensemble model with selected feature vector.
* make sure to save all information(F1 Score, Precision, Recall, Accuracy)
* **Expectations: performance of ensemble model **

**8) Modeling with Dynamic Model(LSTM) with Glove, Word2Vec**
* create Word2Vec, Glove word embedding
* train LSTM
* make sure to save all information(F1 Score, Precision, Recall, Accuracy)
* **Expectations: performance of dynamic model **

**9) Visualization**
* Visualize results of ROC courve.
* Compare each models to find best results.
