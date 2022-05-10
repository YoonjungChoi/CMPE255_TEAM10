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

The figure shows the missing values of 'location' and 'keywords' features.

![image](https://github.com/YoonjungChoi/CMPE255_TEAM10/blob/main/paper/images/fig.missing.png)

(put images by ishaan and abhitteja)

**Observation:** The 'location' feature does not have format and it is not generated automatically. The feature has invalid data such as 'have car; will travel', 'peekskill. new york, 10566', or 'milky way'. We do not use 'location' as a feature.

The figure 4 shows the percentage of feature 'target's distribution.

![image](https://github.com/YoonjungChoi/CMPE255_TEAM10/blob/main/paper/images/fig.labelpie.png)

**Observation:** we cannot say that we have perfectly balanced dataset, but slightly it is balanced data set.

## Data Cleaning

We should modify data to filter meaningless data. For cleaning text, we changed all words to lowercase, removed URL, HTML tags, Emojis, punctuation and ASCII codes.

This table shows parital of original 'text' feature.

![image](https://github.com/YoonjungChoi/CMPE255_TEAM10/blob/main/paper/images/fig.oridata.png)

**Observation:** We observed each instance has mixed data such as upper/lower cases, url, emojis.

This tableshows changes after cleaned meaningless data.

![image](https://github.com/YoonjungChoi/CMPE255_TEAM10/blob/main/paper/images/fig.cleantext.png)

**Observation:** We observed there is removal of unnecessary data.

## Data Preprocessing

Now, we have a cleaned text set and we should apply some methods to normalize words. [NLTK](https://www.nltk.org/index.html) provides easy-to use interfaces for natural language processing.

### Takenization

Tokenization divides strings into lists of substrings. We can use library to find the words and punctuation in a sentences.
The table shows changes after applying tokenization.

![image](https://github.com/YoonjungChoi/CMPE255_TEAM10/blob/main/paper/images/fig.tok.png)

**Observation:** we observed the seperated words in each instance.

### Stopwords

We should remove commonly used words, such as "the", "a", "is", "in".
The table shows changes after applying stopwords.

![image](https://github.com/YoonjungChoi/CMPE255_TEAM10/blob/main/paper/images/fig.stopw.png)

**Observation:** we observed the removal of stop words. In first instance, 'out', 'for', 'more', 'set', 'me' is removed.

### Stemming

Stemming is the process of producing morphological variants of a root/base word. For example, words such as “Likes”, ”liked”, ”likely” and ”liking” will be reduced to “like” after stemming. There are different algorithms for stemming. Porter Stemmer, one of them, is a basic stemmer and it is straightforward and fast to run.
The table shows changes after applying stemming.

![image](https://github.com/YoonjungChoi/CMPE255_TEAM10/blob/main/paper/images/fig.stem.png)

**Observation:** we observed some changes of words. The 'crying' changed to 'cri' or 'acquisitions' changed to 'acquisit'.

### Lemmatization

Lemmatization is the process of grouping together the inflected forms of a word so they can be analysed as a single item, identified by the word's lemma, or dictionary form. Both stemming and lemmatization are word normalization techniques, but we can find the word in dictionary in case of lemmatization. For example, original words 'populated' changed 'popul' in Stemming, but it is not changed in lemmatization. Lemmatization is more better performed than Stemming[Naturalstemming-vs-lemmatization](https://www.baeldung.com/cs/stemming-vs-lemmatization). We decided to apply lemmatization.

![image](https://github.com/YoonjungChoi/CMPE255_TEAM10/blob/main/paper/images/fig.lem.png)

**Observation:** we observed some changes of words. Unlike stemming, lemmatization made that 'crying' changed to 'cry' or 'acquisitions' changed to 'acquisition'.

## Data Visualization

After normalized text, we made data visualization by using word cloud. 

This wordcloud is about disaster.

![image](https://github.com/YoonjungChoi/CMPE255_TEAM10/blob/main/paper/images/fig.wordcloud_1.png)

**Observation:** we can discover disaster related words; suicide, police, news, kill, attack, death, california, storm, flood. 

This wordcloud is about non-disaster.

![image](https://github.com/YoonjungChoi/CMPE255_TEAM10/blob/main/paper/images/fig.wordcloud_0.png)

**Observation:** the non disaster tweets shows that time, want, great, feel, read, but also injury or emergency are found.

The Figure represent histogram of the number of words at each sample.

![image](https://github.com/YoonjungChoi/CMPE255_TEAM10/blob/main/paper/images/fig.numWords_label.png)

**Observation:** 

## Word Embedding to transform data into numerical feature vectors

Word embedding is one of the most popular representation of document vocabulary. It is capable of capturing context of a word in a document, semantic and syntactic similarity, relation with other words, etc.

### CountVectorizer

Bag of Words model is a simplified representation used in natural language processing. A text is represented as the bag of its words, disregarding grammar and describes the occurrences of words with in a document. CountVectorizer can be used for bag of words model. This convert a collection of text documents to a matrics of token counts.

![image](https://github.com/YoonjungChoi/CMPE255_TEAM10/blob/main/paper/images/fig.countvec.png)

**Observation:** we observed the number of occurrences of words. The 'deed' occurs around 4000.

### Term Frequency Inverse Document Frequency(Tf-Idf)

Some articles say that TF-IDF is better than Count Vectorizers because it not only focuses on the frequency of words present in the corpus but also provides the importance of the words. We can then remove the words that are less important for analysis, hence making the model building less complex by reducing the input dimensions.

![image](https://github.com/YoonjungChoi/CMPE255_TEAM10/blob/main/paper/images/fig.tfidf.png)

**Observation:** we observed the number of occurrences of words. The 'deed' occurs around 15000.

### Word2Vec

Word2Vec, a word embedding toolkit, uses a neural network model to learn word associations from a large corpus of text. Word2Vec represents words in vector space in a way of similar meaning words are positioned in close locations but dissimilar words are placed far away. For non sequential models, we build a sentence embedding by averaging the values across all token embeddings output by Word2Vec. We followed the gensim-tutorial to visualize the relationship with words.

![image](https://github.com/YoonjungChoi/CMPE255_TEAM10/blob/main/paper/images/fig.w2v_wildfire.png)

**Observation:** The blue are most similar words with the 'Wildfire' and green is most unrelated words with the 'Wildfire'

### Word2Vec with PCA applied

As principal component analysis is a strategy to reduce dimension, we applied PCA to reduce 100 components from Word2Vec with dimensionality of 300. 
Word2Vec of Gensim's default dimensionalty is 100. We tried to train and get better accuracy in several experiments. As a result, for instance, with Xgboost model, the case of creating 300 dimensionalty and appling PCA to create 100 pricipal components has better performance rather than word2vec with 100 dimensionality without PCA.

### Glove




# Methods

We trained each numerical feature vectors on the basic models to find which feature vectors can yield better performance. For the fine-tuning, we adjusted parameters on the model with the selected feature vector. We repeated the same steps on other models. We also trained each feature vectors with ensemble method.

## Logistic Regression(LR)

Logistic regression is a process of modeling the probability of a discrete outcome given an input variable. This classifer is easier to implement, interpret, very efficient to train and performs well. Thus, at first, we tried to train four feature vectors sets on logistic regression. 

![image](https://github.com/YoonjungChoi/CMPE255_TEAM10/blob/main/paper/images/fig.LogisticR.png)

**Observation:** We observed the count vectorizer feature set resulted in better accuracy(0.797) and precision(0.813).

From parameter optimization, we finalized parameters as C=0.15, penalty='l2', tol=0.001, solver='saga', random state=42, max iter=1000. We obtained the following result and confusion matrics:

![image](https://github.com/YoonjungChoi/CMPE255_TEAM10/blob/main/paper/images/fig.lr_final_score.png)

**Observation:** We observed that optimization does not improve significantly, but it improved to accuracy(0.800) and precision(0.837).

![image](https://github.com/YoonjungChoi/CMPE255_TEAM10/blob/main/paper/images/fig.logisticR_final_cm.png)

**Observation:** This confusion matrix shows the number of instance between prediction and actuals. This Logistic Regression model predicts 668 true positive (disaster) and 1160 true negative(non-disaster) instances.

## Support Vector Machine(SVM)
Support Vector Machine is a supervised learning model used for classification and regression problems. SVM can be used when data has exactly two classes. SVM classifies data by finding the best hyperplane that separates all data points of one class from those of the other class. The best hyperplane for an SVM means the one with the largest margin between the two classes. We trained four feature vectors on basic SVM, which means no changes of parameters.

![image](https://github.com/YoonjungChoi/CMPE255_TEAM10/blob/main/paper/images/fig.svm_basic.png)

**Observation:** We observed the count vectorizer feature set resulted in better accuracy(0.799).

We adjusted parameters to yield better accuracy. In the final SVM model, it has default C value as 1, gamma value as 'auto', kernel value as 'sigmoid'. We obtained the result and confusion matrics of the model.

![image](https://github.com/YoonjungChoi/CMPE255_TEAM10/blob/main/paper/images/fig.svm_final_score.png)

**Observation:** We observed that optimization does not improve significantly, improvements, but it improved to accuracy(0.800). 

![image](https://github.com/YoonjungChoi/CMPE255_TEAM10/blob/main/paper/images/fig.svm_final_cm.png)

**Observation:** This confusion matrix shows that SVM predicts 664 true positive(disaster)and 1163 true negative(non-disaster) instances.


## Decision Tree

Decision Tree is also used for classification problem. Advantages of classification with Decision Trees are inexpensive to construct, extremely fast at classifying unknown records, easy to interpret for small-sized trees, accuracy comparable to other classification techniques for many simple data sets, excludes unimportant features. Thus, we tried to train data set on decision tree.

![image](https://github.com/YoonjungChoi/CMPE255_TEAM10/blob/main/paper/images/fig.dt.png)

**Observation:** We observed the tf-idf feature set resulted in better accuracy(0.752) than other feature sets and count vectorizer feature resulted in better precision(0.731). Decision Tree decided to different node at each time, so result can be differ.


From parameter optimization, we finalized parameters as min samples split=8. We obtained the result and confusion matrics.

![image](https://github.com/YoonjungChoi/CMPE255_TEAM10/blob/main/paper/images/fig.dt_final_score.png)

**Observation:** We observed accuracy(0.0.756). We don't have significant improvement from parameter optimization, but the default parameter could be work well because it designed general purpose. 

![image](https://github.com/YoonjungChoi/CMPE255_TEAM10/blob/main/paper/images/flg.dt_final_cm.png)

**Observation:** This confusion matrix shows that SVM predicts 671 true positive(disaster) and 1055 true negative (non-disaster) instances.

## Random Forest

Random Forest is a supervised learning algorithm. It can be used for both classification and regression. However, it's mainly used for classification problems. A forest comprises trees and its said that more trees it has, more robust the forest is. Random Forest is a set of multiple decision trees. Random Forest creates decision trees on randomly selected data samples, gets predictions from each tree and selects the best solution by means of voting.

Decision trees may suffer from overfitting but random forest prevents overfitting by creating trees on random subsets. Decision trees are computationally faster.

[image for score, confusion matrix]

**Observation:** 

## Xgboost


[image for score, confusion matrix]

**Observation:** 

## LSTM


## Ensemble
Ensemble methods are techniques that create multiple models and then combine them to produce improved results. Ensemble methods usually produces more accurate solutions than a single model would.  We have four different feature sets and random_state parameter enable to split feature set in the same way, which means we can use ensemble model by our own. Based on the voting way, First ensemble model consisted of non sequential models; Logistic Regression with Count vetorizer, SVM with Counter vectiroizer, Decision Tree with Tf-Idf, RandomForeset with counter vectorizer, Xgboost with word2vec applied PCA.


![image]()

**Observation:** 


**Observation:** 

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

[1] Text Preprocessing for NLP (Natural Language Processing),Beginners to Master, https://medium.com/analytics-vidhya/text-preprocessing-for-nlp-natural-language-processing-beginners-to-master-fd82dfecf95
[2] Text Preprocessing in NLP, https://towardsdatascience.com/text-preprocessing-in-natural-language-processing-using-python-6113ff5decd8
[3] WordCloud, https://kavita-ganesan.com/python-word-cloud/#.YnnPcPPMIeU
[4] Word2Vec, gensim-word2vec-tutorial, https://www.kaggle.com/code/pierremegret/gensim-word2vec-tutorial/notebook
[5] Word2Vec, Wikipedia, https://en.wikipedia.org/wiki/Word2vec
