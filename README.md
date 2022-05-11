# CMPE255_TEAM10 [Natural Language Processing With Disaster Tweets]

**Team Members**

* Yoonjung Choi
* Abhiteja Mandava
* Ishaan Bhalla
* Sakruthi Avirineni

## Introduction
Social Network Services have become not only an important source of emergency information during disaster but also a medium for expressing immediate responses of warning, evacuation or rescue, providing immediate assistance, assessing damage, continuing assistance and the immediate restoration or construction of infrastructure. As a result, predicting the context of SNS is a crucial concern in our society. More agencies want to monitor or track SNS intelligently by using technologies. By analyzing context, we can utilize this study to track, monitor and predict disasters from the real time data and this study would help making prediction model.

Natural Language Processing(NLP) has been widely used to analyze SNS and extract potential patterns. Twitter is one of the popular SNS platforms and many tweets has been delivered in emergency situation. This project on Twitter analysis is about prediction problems whether a person's words are actually telling of a disaster.

![image](https://storage.googleapis.com/kaggle-media/competitions/tweet_screenshot.png)

Here is an example. If someone says that 'On plus side LOOK AT THE SKY LAST NIGHT IT WAS ABLAZE', We can understand it does not mean 'disaster' but it means metaphorically something. However, It is not clear to the Machine. Thus, we investigated techniques for natural language processing and explored diverse machine learning models with several useful word embedding algotithms.

First step is about data cleaning, which means that we have to remove meaningless data, e.g. html tags, url, emojis, ascii codes and punctuation.

Second step is about applying practical algorithms e.g. stop words, stemming to find the root of words. Again, we need to make sure that data has meaningful data as much as possible, so we remove stop words(are, the, a) and apply stemming to lowers inflection in words to their root forms. According to Wikipedia, although a word may exist in several inflected forms, having multiple inflected forms inside the same text adds redundancy to the NLP process.

Third step is about applying word embedding to transform text into numerical feature vectors, e.g. CountVectorizer, Tf-Idf, word2vec, glove. We have to transform text into numerical feature vectors so machine learning classifiers can understand input data. The below we will talk in detail about word embedding algorithms. For the non sequential models such as SVM, Logistic Regression, Decision Tree, RandomForest, XGboost are trained with CountVectorizer, Tf-Idf, Word2vec(We build a sentence embedding by averaging the values across all token embeddings output). For deep learning or the sequential model such as LSTM, we trained word2vec and glove. The below, we will talk about model why we choose. 

For comparison, other submissions of Kaggle have done similar steps for preprocessing and applying models. They tried to train a single model or even in case of ensemble, they trained exisiting ensemble classifier with the same data set. However, we have a little bit different direction on that. We expect that there would be suitable combination of betweet feature vectors and models. So we will try to applying a few featuer vectors and models to find combination to get better performance and we will try to make a comparison. 


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

**1) 'target' label**

This plot shows distribution of the **target** label.

![image](https://user-images.githubusercontent.com/20979517/164575693-d0ee93c4-d68e-4697-a108-d616754b6eed.png)

**Observation:** we cannot say that it has perfectly balanced dataset, but slightly it is balanced data set.

**2) 'keyword' feature**

This plot shows distribution of the **keyword** feature based on target.

![image](https://github.com/YoonjungChoi/CMPE255_TEAM10/blob/main/paper/images/keywordSet.png)

**Observation:** keyword feature does not have much null values(39), but there are many common words. The 'keyword' feature of disaster has 221 words, and The 'keyword' feature of non-disaster has 219 words. There are 218 intersections words. The difference that only 'disaster' tweets have are {'debris', 'wreckage', 'derailment'}. The difference that only 'non-disaster' tweets have are {'aftershock'}. Thus, we does not use keyword feature.

**3) 'location' feature**

This Figure shows a few samples of the **location** feature.

![image](https://github.com/YoonjungChoi/CMPE255_TEAM10/blob/main/paper/images/locationEx.png)

**Observation:** 'location' feature have many null values(2533) and does not have format and it is not generated automatically. This feature has invalid data such as 'Happily Married with 2 kids', 'have car; will travel', 'peekskill. new york, 10566', or 'milky way'. We do not use 'location' as a feature.

**4) 'text' feature**

This plot shows distribution of the **text** feature based on target.

@abhiteja


**Observation:** The 'text' feature has unnecessary data that we should handle. 


**Observation:** Finally, we dropped 'id', 'keyword', 'location' features and use only 'text' feature and 'target' label.

## Word Embedding
Word embedding is one of the most popular representation of document vocabulary. It is capable of capturing context of a word in a document, semantic and syntactic similarity, relation with other words, etc. 

**Why?** 
One of the biggest problems with text is that it is messy and unstructured, and machine learning algorithms prefer structured, well defined fixed-length inputs. To make machine learning understand input data, we need to transform 'text' or words or sentences into fixed-length numerical feature vectors.

### How?

**1) CountVectorizer**

The most simple and known method is the Bag-Of-Words representation of text that describes the occurrence of words within a document. It’s an algorithm that transforms the text into fixed-length vectors. Count vectorizer can be used for Bag-Of-Words representation.

**2) TF-IDF(Term Frequency Inverse Document Frequency)**

Some articles says that TF-IDF is better than Count Vectorizers because it not only focuses on the frequency of words present in the corpus but also provides the importance of the words. It gives weight more to a term that occurs in only a few documents.

**3) Word2Vec**

According to Wikipedia, Word2vec is a group of related models that are used to produce word embeddings. The word2vec algorithm uses a neural network model to learn word associations from a large corpus of text. Once trained, such a model can detect synonymous words or suggest additional words for a partial sentence. If we put a word in trained word2vec, it returns a feature vector. In case of non sequential model, since one instance of 'text' feature have several words(sentence), we can build a sentence embedding by averaging the values across all token embeddings output by Word2Vec.

**4) Glove**

GloVe stands for global vectors for word representation. It is an unsupervised learning algorithm developed by Stanford for generating word embeddings by aggregating global word-word co-occurrence matrix from a corpus. The resulting embeddings show interesting linear substructures of the word in vector space. Files with the **pre-trained vectors** Glove can be found in many sites like [Kaggle](https://www.kaggle.com/datasets/danielwillgeorge/glove6b100dtxt). We will use the glove.6B.100d.txt file containing the glove vectors trained on the Wikipedia and GigaWord dataset.

**Expectations: we expect to understand how each word embedding algorithm works on it and its performances with models**

## Classifiers

We will use **Non-Sequential Model(Logistic Regression, SVM, Decision Tree, Random Tree, XGboost) and Deep Learning or Sequential Model(LSTM)** to solve this binary classification problem.

(Sequence models are the machine learning models that input or output sequences of data. Sequential data includes text streams, audio clips, video clips, time-series data and etc [Article](https://towardsdatascience.com/sequence-models-and-recurrent-neural-networks-rnns-62cadeb4f1e1).)

Before experiments, we have searched some machine learning classifiers we can use for a binary classification problem. There are explanations **how** the classifier works and **why** the classfier would work for our problem.

**1) Logistic Regression**

Logistic Regression is a supervised machine learning algorithm that can be used to model the probability of a certain class or event. It is used when the data is linearly separable and the outcome is binary or dichotomous in nature. That means Logistic regression is usually used for Binary classification problems. Also, Logistic Regression is easier to implement, interpret, and very efficient to train.	Logistic Regression is good accuracy for many simple data sets and it performs well when the dataset is linearly separable. So first we tried to train data sets on logistic regression. 

**2) Support Vector Machine(SVM)**

We can use a support vector machine (SVM) when data has exactly two classes. This is the reason we choose this classifer. An SVM classifies data by finding the best hyperplane that separates all data points of one class from those of the other class. The best hyperplane for an SVM means the one with the largest margin between the two classes. Margin means the maximal width of the slab parallel to the hyperplane that has no interior data points.

**3) Decision Tree**

A decision tree can be used for either regression or classification. Advantages of classification with Decision Trees are inexpensive to construct, extremely fast at classifying unknown records, easy to interpret for small-sized trees, accuracy comparable to other classification techniques for many simple data sets, excludes unimportant features. Thus, we tried to train data set on decision tree as well.

**4) Random Tree**

**5) XGboost**
 

**6) LSTM**

For Natural language processing, Long Short Term Memory (LSTM) network were used as deep learning models for automatic feature extraction from tweet texts. Unlike standard feedforward neural networks, LSTM has feedback connections. Such a recurrent neural network can process not only single data points (such as images), but also entire sequences of data [Wiki](https://en.wikipedia.org/wiki/Long_short-term_memory).


**7) Ensemble**
Ensemble methods are techniques that create multiple models and then combine them to produce improved results. Ensemble methods usually produces more accurate solutions than a single model would. There are different criteria to make output, but we will create the voting classifier that predicts on the basis of aggregation the findings of each base models.

**Expectations: we expect to understand how models work on it and its performances**

## WorkFlow

**1) Data Exploration**
* Loading data set
* Visualization data set
* Understanding data set
* **Expectations: From data exploration, we would understand label is balanced or imbalanced, which feature should drop or keep to use, know about preprocessing phase we needed**

**2) Data Cleaning**
* Change all characters to lowercase
* Makes sure to remove URL, HTML, Emojis, ASCII and Punctuation. 
* **Expectations: From data cleaning, we would understand what or why we should do before next phase**

**3) Data Preprocessing Part1 Using [NLTK](https://www.nltk.org/index.html)**:
* Tokenize
* Remove Stopwords(Common words, example: are)
* PorterStemmer (Stemming is the important in NLP, example: asked -> ask)
* WordNetLemmatizer (example: peopl => people) -> We decided to use this, not stemming, due to performance.
* **Expectations: From applying algorithms, we would understand what or why algorithms can be applied for preprocessing, how they work, and words' changes after applying algorithms**

**4) Data Preprocessing Part2 Word Embedding to transform text to numerical feature vectors**
* apply CountVector, visualize the number of count of words.
* apply TF-IDF, visualize the number of count of words.
* apply Word2Vec, visualize vectors based on similiar and not opposite words.
  -> we build a sentence embedding by averaging the values across all token embeddings output by Word2Vec.
* apply Word2Vec with PCA.
* **Expectations: From exploring word embedding algorithms, we would learn how embedding processes are doing.

**5) Data Split**
* split each type of feature
  -> data set by using count vectorizer
  -> data set by using tf-idf
  -> data set by using word2vec(sentence embedding using average of vector)
  -> data set by using word2vec with PCA
* **Expectations: we would create 4 different feature vectors sets from one 'text' data set.**

**6) Modeling with Machine Learning Algorithms:** 
* build model
  -> Logistic Regression, SVM, Decision Tree, RandomTree, XGboost
* train each type of feature vectors on models
* **Expectations: From train models, we would find one suitable feature vector having better performance rather than other feature vectors sets** 

* re-train model with selected feature set
* find best parameters of model
* make sure to save all information(F1 Score, Precision, Recall, Accuracy, [Confusion Matrix](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html))
* **Expectations: From parameter optimazation, we would improve the performance of model**

**7) Modeling with sequence model(LSTM) with Glove, Word2Vec**
* create Word2Vec, Glove word embedding
* train LSTM 
* make sure to save all information(F1 Score, Precision, Recall, Accuracy, Confusion Matrix)
* **Expectations: we would expect better performance of sequence model because it consider sequence of text.**

**8) Custom Ensemble**
* create model with parameters obatined by step 6.
* create ensemble model with combination of models and feature sets.
* make sure to save all information(F1 Score, Precision, Recall, Accuracy)
* **Expectations: we would expect better performance of ensemble model because apparently, it would cause better performance.**

**9) Visualization**
* Visualize results of ROC courve.
* Compare each models to find best results.
* **Expectations: we would expect to have ROC courve on diverse models**

