---
title: "Natural Language Processing with Disaster Tweets"
date: "May 2022"
author: Yoonjung Choi, Abhiteja Mandava, Ishaan Bhalla, Sakruthi Avirineni, San José State University

header-includes: |
  \usepackage{booktabs}
  \usepackage{caption}
---

# Abstract

Social Network Service have become not only an important sources of emergency information during disaster but also a medium for expressing immediate responses of warning, evacuation or rescue. As a result, predicting context of SNS is a crucial concern in our society. By analyzing context, we can utilize this study to track, monitor and predict disasters from the real time data and this study would help making prediction model. In this paper, data was retrieved from the company Figure-Eight, and key problem is dealing with the natural language processing by using ensemble model handling different feature vectors. We proposed ensemble model using combinations with different feature vectors and classifiers. The results are compared with each classifier with different feature vectors and existing ensemble classifiers that apply the same data set to several classifiers. After analyzing data, factors normalizing data set and transforming to feature vectors were identified, and measures to improve accuracy were proposed.

# Introduction

Social Network Service has been playing a crucial role in communicating in our society and Natural Language Processing has been widely used to analyze SNS and extract potential patterns. Twitter is one of the popular SNS platforms and many tweets has been delivered in emergency situation. Since there are demands for companies to utilize this tweets, we investigated natural language processes and developed prediction model having the better performance.

In this paper, for pre-processing, we cleaned data set from unnecessary information such as URL, Emojis or HTML tags and normalized data set by using useful algorithms; tokenizer, stopwords and lemmatization. We transformed given text into feature vectors by using Count Vectorizer, Inverse Document Frequency, Word2Vec and Word2Vec with PCA applied, and trained each numerical feature vector on different models; Decision Tree, Support Vector Machine, Logistic Regression, and Ensemble Model. We found each combination how each model has the higher performance on which feature vectors. Then, after fine-tuning, we made ensemble model training each classifier with feature vectors that resulted in higher accuracy, unlike an existing ensemble model using the same data set.

As a result, the ensemble model of SVM with Count Vectorizer, Logistic Regression with Count Vectorizer and Decision Tree with Tf-Idf gave the better results. The Results were compared based on different performance matrics such as Accuracy, Recall, Precision, F1 Score.

# Data Exploration

## Data Set

The data set has been collected from the company figure-eight and originally shared on their ‘Data For Everyone’ [website](https://appen.com/datasets-resource-center). We found the data set from [Kaggle Competition](https://www.kaggle.com/competitions/nlp-getting-started/data). It contains 7613 tweets data with the following features:

![image](https://github.com/YoonjungChoi/CMPE255_TEAM10/blob/main/paper/images/fig.dataset.png)

The figure shows the missing values of 'location' and 'keywords' features. The 'location' feature does not have format and it is not generated automatically. That's why it has dirty values, such as 'have car; will travel', 'peekskill. new york, 10566', or 'milky way'. We do not use 'location' as a feature.

![image](https://github.com/YoonjungChoi/CMPE255_TEAM10/blob/main/paper/images/fig.missing.png)

The figure 4 shows the percentage of feature 'target's distribution.

![image](https://github.com/YoonjungChoi/CMPE255_TEAM10/blob/main/paper/images/fig.labelpie.png)

## Data Cleaning

We should modify data to filter meaningless data. For cleaning text, we changed all words to lowercase, removed URL, HTML tags, Emojis, punctuation and ASCII codes.

This table shows parital of original 'text' feature.

![image](https://github.com/YoonjungChoi/CMPE255_TEAM10/blob/main/paper/images/fig.oridata.png)

This tableshows changes after cleaned meaningless data.

![image](https://github.com/YoonjungChoi/CMPE255_TEAM10/blob/main/paper/images/fig.cleantext.png)

## Data Preprocessing

Now, we have a cleaned text set and we should apply some methods to normalize words. [NLTK](https://www.nltk.org/index.html) provides easy-to use interfaces for natural language processing.

### Takenization

Tokenization divides strings into lists of substrings. We can use library to find the words and punctuation in a sentences.
The table shows changes after applying tokenization.

![image](https://github.com/YoonjungChoi/CMPE255_TEAM10/blob/main/paper/images/fig.tok.png)

### Stopwords

We should remove commonly used words, such as "the", "a", "is", "in".
The table shows changes after applying stopwords.

![image](https://github.com/YoonjungChoi/CMPE255_TEAM10/blob/main/paper/images/fig.stopw.png)

### Stemming

Stemming is the process of producing morphological variants of a root/base word. For example, words such as “Likes”, ”liked”, ”likely” and ”liking” will be reduced to “like” after stemming. There are different algorithms for stemming. Porter Stemmer, one of them, is a basic stemmer and it is straightforward and fast to run.
The table shows changes after applying stemming.

![image](https://github.com/YoonjungChoi/CMPE255_TEAM10/blob/main/paper/images/fig.stem.png)

### Lemmatization

Lemmatization is the process of grouping together the inflected forms of a word so they can be analysed as a single item, identified by the word's lemma, or dictionary form. Both stemming and lemmatization are word normalization techniques, but we can find the word in dictionary in case of lemmatization. For example, original words 'populated' changed 'popul' in Stemming, but it is not changed in lemmatization. Lemmatization is more better performed than Stemming[Naturalstemming-vs-lemmatization](https://www.baeldung.com/cs/stemming-vs-lemmatization). We decided to apply lemmatization.

![image](https://github.com/YoonjungChoi/CMPE255_TEAM10/blob/main/paper/images/fig.lem.png)

## Data Visualization

After normalized text, we made data visualization by using word cloud. In disaster tweet's words, we can discover disaster related words; suicide, police, news, kill, attack, death, california, storm, flood. In the other hand, the non disaster tweets shows that time, want, great, feel, read, but also injury or emergency are found.

This wordcloud is about disaster.

![image](https://github.com/YoonjungChoi/CMPE255_TEAM10/blob/main/paper/images/fig.wordcloud_1.png)

This wordcloud is about non-disaster.

![image](https://github.com/YoonjungChoi/CMPE255_TEAM10/blob/main/paper/images/fig.wordcloud_0.png)

The Figure represent histogram of the number of words at each sample.

![image](https://github.com/YoonjungChoi/CMPE255_TEAM10/blob/main/paper/images/fig.numWords_label.png)

## Transforming numerical feature vectors

Word embedding is a term used for the representation of words for text analysis, typically in the form of a real-valued vector that encodes the meaning of the word such that the words that are closer in the vector space are expected to be similar in meaning.

### CountVectorizer

Bag of Words model is a simplified representation used in natural language processing. A text is represented as the bag of its words, disregarding grammar and describes the occurrences of words with in a document. CountVectorizer can be used for bag of words model. This convert a collection of text documents to a matrics of token counts.

![image](https://github.com/YoonjungChoi/CMPE255_TEAM10/blob/main/paper/images/fig.countvec.png)

### Term Frequency Inverse Document Frequency(Tf-Idf)

The Tf-Idf is a measure of whether a term is common or rare. It gives weight more to a term that occurs in only a few documents.

![image](https://github.com/YoonjungChoi/CMPE255_TEAM10/blob/main/paper/images/fig.tfidf.png)

### Word2Vec

Word2Vec, a word embedding toolkit, uses a neural network model to learn word associations from a large corpus of text[wiki](https://en.wikipedia.org/wiki/Word2vec). Word2Vec represents words in vector space in a way of similar meaning words are positioned in close locations but dissimilar words are placed far away.

![image](https://github.com/YoonjungChoi/CMPE255_TEAM10/blob/main/paper/images/fig.w2v_wildfire.png)

### Word2Vec with PCA applied

As principal component analysis is a strategy to reduce dimension, we applied PCA with 100 components on feature vectors from Word2Vec. The below figure is shown when applying PCA with 2 components.

![image](https://github.com/YoonjungChoi/CMPE255_TEAM10/blob/main/paper/images/fig.pca_comp2_word2vec.png)

# Methods

We trained each numerical feature vectors on the basic models to find which feature vectors can yield better performance. For the fine-tuning, we adjusted parameters on the model with the selected feature vector. We repeated the same steps on other models. We also trained each feature vectors with ensemble method.

## Support Vector Machine(SVM)

Support Vector Machine is a supervised learning model used for classification and regression problems. We trained each numerical feature vectors on basic SVM, which means no changes of parameters. In case of SVM, CountVectorizer feature vector has higher accuracy and f1 score than other feature vectors.

![image](https://github.com/YoonjungChoi/CMPE255_TEAM10/blob/main/paper/images/fig.svm_basic.png)

We adjusted parameters to yield better accuracy. In the final SVM model, it has default C value as 1, gamma value as 'auto', kernel value as 'sigmoid'. We obtained the result and confusion matrics of the model.

![image](https://github.com/YoonjungChoi/CMPE255_TEAM10/blob/main/paper/images/fig.svm_final_score.png)

![image](https://github.com/YoonjungChoi/CMPE255_TEAM10/blob/main/paper/images/fig.svm_final_cm.png)

## Logistic Regression(LR)

Logistic regression is a process of modeling the probability of a discrete outcome given an input variable. It is also a supervised learning model used for classification problems. We trained each feature vectors on basic Logistic Regression without fine-tuning, and Count Vector has higher accuracy as well.

![image](https://github.com/YoonjungChoi/CMPE255_TEAM10/blob/main/paper/images/fig.LogisticR.png)

From the fine-tuning, we finalized parameters as C=0.15, penalty='l2', tol=0.001, solver='saga', random state=42, max iter=1000. We obtained the following result and confusion matrics:

![image](https://github.com/YoonjungChoi/CMPE255_TEAM10/blob/main/paper/images/fig.lr_final_score.png)

![image](https://github.com/YoonjungChoi/CMPE255_TEAM10/blob/main/paper/images/fig.logisticR_final_cm.png)

## Decision Tree

Decision Tree is a non-parametric supervised learning method used for classification and regression problems. We trained each numerical feature vectors on basic decision tree, and Tf-Idf feature vector has a little bit higher accuracy rather than others.

![image](https://github.com/YoonjungChoi/CMPE255_TEAM10/blob/main/paper/images/fig.dt.png)

From the fine-tuning, we finalized parameters as min samples split=8. We obtained the result and confusion matrics.

![image](https://github.com/YoonjungChoi/CMPE255_TEAM10/blob/main/paper/images/fig.dt_final_score.png)

![image](https://github.com/YoonjungChoi/CMPE255_TEAM10/blob/main/paper/images/flg.dt_final_cm.png)

## Ensemble

Ensemble methods are techniques that create multiple models and then combine them to produce improved results. Ensemble methods usually produces more accurate solutions than a single model would. We used hard voting classifier and trained each feature vectors on ensemble model consisted of SVM, Logistic Regression and Decision Tree. The figure 27 shows each ensemble's accuracy and ensemble model with CountVectorizer feature vector yields better accuracy.

![image](https://github.com/YoonjungChoi/CMPE255_TEAM10/blob/main/paper/images/fig.ensemble.png)

Based on the hard voting, we made custom ensemble model combined of SVM with CountVectorizer, Logistic Regression with CountVectorizer, and Decision Tree with Tf-Idf. As a result, we got 0.806 accuracy. The following figure is about confusion matrics of custom ensemble model.

![image](https://github.com/YoonjungChoi/CMPE255_TEAM10/blob/main/paper/images/fig.custom_mx.png)

## Random Forest

Random Forest is a supervised learning algorithm. It can be used for both classification and regression. However, it's mainly used for classification problems. A forest comprises trees and its said that more trees it has, more robust the forest is. Random Forest is a set of multiple decision trees. Random Forest creates decision trees on randomly selected data samples, gets predictions from each tree and selects the best solution by means of voting.

Decision trees may suffer from overfitting but random forest prevents overfitting by creating trees on random subsets. Decision trees are computationally faster.

Bag of words:
It is a way of extracting features from the text for use in machine learning algorithms. It has seen great success in problems like:
1)NLP
2)Document Classification
3)Information retrieval from documents

Vectorization:
The process of converting NLP text into numbers is called vectorization.
Different ways to convert text to vectors are:
1)Counting the number of times each word appears in the document.
2)Calculating the frequency that each word appears in the document out of all words in the document.

Bag of words implementation:
1)Count vectorizer
2)TF-IDF vectorizer
3)N-Grams

Feature Extraction:
\_Input text -> Clean text -> Tokenize -> Build vocabulary -> Generator vectors -> ML algorithm

# Comparisons

## Performance Matrics

**1. Accuracy**
Accuracy is a metric that generally describes how the model performs across all classes. It is calculated as the ratio between the number of correct predictions to the total number of predictions.

**2. Precision**
The precision is calculated as the ratio between the number of Positive samples correctly classified to the total number of samples classified as Positive (either correctly or incorrectly).

**3. Recall**
The recall is calculated as the ratio between the number of Positive samples correctly classified as Positive to the total number of Positive samples.

**4. F1 Score**
The F1-score combines the precision and recall of a classifier into a single metric.

**5. ROC Curve**
ROC curve is a graphical plot that illustrates recall(x-axis) and precision(y-axis) for different thresholds.

## Comparison

(insert ROC Curve graph)

Also, other submissions of Kaggle competition used similar steps using algorithms to transform to numerical feature vectors and classifiers including ensemble models as well. However, there is no comparison to find each combination of feature vectors and classifiers, to make custom ensemble models. Our model considered finding suitable combination of a feature vector and a classifier and then, applying ensemble model.

# Conclusions

In this analysis we experienced three prominent word embedding and classification techniques of using an Fiure-Eight Company data set.
Ensemble with combined different feature vectors and classifiers in our experiment outperfomanced the other classifiers on each data set.
//[Not Yet..]

# Limitations And Future research

We obtained the qualified data set from company, so we assumed that content of data are true. However, the thing that content could be fake is the main limitation of this study. Overcoming these limitations can be done in future research. By dealing with distinguishing the content is fake or not first, we can predict emergency situations and properly respond them.

# References
