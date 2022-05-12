---
title: "Natural Language Processing with Disaster Tweets"
date: "May 2022"
author: Yoonjung Choi, Abhiteja Mandava, Ishaan Bhalla, Sakruthi Avirineni, San José State University

header-includes: |
  \usepackage{booktabs}
  \usepackage{caption}
---

# Abstract

Social Network Services(SNS) have become not only an important source of emergency information during disasters but also a medium for expressing immediate responses of warning, evacuation or rescue. As a result, predicting the context of SNS is a crucial concern in our society. By analyzing context, we can utilize this study to track, monitor and predict disasters from the real time data and this study would help make prediction models. In this paper, data was retrieved from the company Figure-Eight, and the key problem is dealing with diverse word embedding and machine learning models for this problem. We tried to find combinations of word embedding algorithms and classifiers, and furthermore, we tried to apply a deep learning model such as Long short-term memory(LSTM) with two different word embedding algorithms. Lastly, we created an ensemble model, and expect a comparison of models we trained. After analyzing data, factors normalizing data set and transforming to feature vector sets were identified, measures to improve performance were proposed, and comparison was conducted.

# Introduction

NLP is a branch of artificial intelligence that helps computers understand and interpret human language and give computers the ability to understand written and spoken words in much the same way human beings can.  Worldwide revenue from the NLP market forecasts to be almost 14 times larger in 2025 than it was in 2017, increasing from around three billion U.S. dollars in 2017 to over 43 billion in 2025 [1].  NLP is  one of the most promising avenues for social media data processing.

NLP has been widely used to analyze SNS and extract potential patterns. SNS has been playing a crucial role in communicating in our society. SNS has become an important vehicle of emergency information during disasters to deliver immediate responses of warning, evacuation or rescue, providing immediate assistance, assessing damage, continuing assistance and the immediate restoration or construction of infrastructure.

Twitter is one of the popular SNS platforms and many tweets have been delivered in emergency situations. Since there are demands for companies to utilize these tweets, we investigate natural language processes and develop prediction models having better performance in this paper. For preprocessing, we clean the data from unnecessary data such as URL, Emojis or HTML tags and normalize the data by using useful algorithms; tokenizer, stopwords and lemmatization. We transform cleaned data into four feature vectors sets by using Countvectorizer, Term Frequency Inverse Document Frequency, Word2Vec and Word2Vec with PCA applied, and trained them on non sequential models; Logistic Regression, Support Vector Machine, Decision Tree, RandomForeset, XGboost. We also train a sequential model such as LSTM with Word2Vec and Glove. We found each combination between word embeddings and classifiers having better performance. Lastly, we build an ensemble model consisting of the combinations we found unlike an existing ensemble model using the same data set. The results were compared based on different performance metrics such as Accuracy, Recall, Precision, F1 Score, Confusion Matrix, ROC Curve.

# Data Exploration

## Data Set

The data set has been collected from the company figure-eight and originally shared on their ‘Data For Everyone’ website [2]. We found the data set from Kaggle Competition [3]. It contains 7613 tweets data with the following features:

![image](https://github.com/YoonjungChoi/CMPE255_TEAM10/blob/main/paper/images/fig.dataset.png)

**Observation:** we observed data set has four features and one labels.

Figure shows the missing values.

![image](https://github.com/YoonjungChoi/CMPE255_TEAM10/blob/main/paper/images/fig.missing.png)

**Observation:** we observed 'location' feature has many null values(2533) and 'keyword' feature has null values(39).

Figure shows the percentage of 'target' label.

![image](https://github.com/YoonjungChoi/CMPE255_TEAM10/blob/main/paper/images/fig.labelpie.png)

**Observation:** we cannot say that it has perfectly balanced dataset, but slightly it is balanced data set.

Figure shows the number of unique words of the 'keyword' feature based on each target.

![image](https://github.com/YoonjungChoi/CMPE255_TEAM10/blob/main/paper/images/keywordSet.png)

**Observation:** keyword feature does not have much null values(39), but there are many common words. The 'keyword' feature of disaster has 221 words, and The 'keyword' feature of non-disaster has 219 words. There are 218 intersections words. The difference that only 'disaster' tweets have are {'debris', 'wreckage', 'derailment'}. The difference that only 'non-disaster' tweets have are {'aftershock'}. Thus, we does not use keyword feature.

Figure shows a few samples of the 'location' feature.

![image](https://github.com/YoonjungChoi/CMPE255_TEAM10/blob/main/paper/images/locationEx.png)

**Observation:** 'location' feature have many null values(2533) and does not have format and it is not generated automatically. This feature has invalid data such as 'Happily Married with 2 kids', 'have car; will travel', 'peekskill. new york, 10566', or 'milky way'. We do not use 'location' as a feature.

The 'id' feature is nominal data, which means that there is no meaningful information. Finally, we dropped 'id', 'keyword', 'location' features and use only 'text' feature and 'target' label.


## Data Cleaning
We should clean data to remove meaningless data because properly cleaned data enable to do good text analysis and help in making accurate decisions for our problem. For cleaning text, we changed all words to lowercase, removed URL, HTML tags, Emojis, punctuation and ASCII codes.

Figure shows parital of original 'text' feature.

![image](https://github.com/YoonjungChoi/CMPE255_TEAM10/blob/main/paper/images/fig.oridata.png)

**Observation:** We observed each instance has mixed data such as upper/lower cases, url, emojis.

Figure shows changes after cleaned meaningless data.

![image](https://github.com/YoonjungChoi/CMPE255_TEAM10/blob/main/paper/images/fig.cleantext.png)

**Observation:** We observed there is removal of unnecessary data.

## Data Preprocessing

We have a cleaned data through the previous step, but here are still a few ways that we can do to extract meaningful information from the cleaned data. We apply stemming or lemmatization to get normalized words. Natural Language Toolkit (NLTK)[ ] provides easy-to use interfaces for natural language processing.

### Takenization

Tokenization divides strings into lists of substrings. We can use library to find the words and punctuation in a sentences.
The table shows changes after applying tokenization.

![image](https://github.com/YoonjungChoi/CMPE255_TEAM10/blob/main/paper/images/fig.tok.png)

**Observation:** we observed the seperated words in each instance.

### Stopwords

We remove commonly used words, such as "the", "a", "is", "in". Figure shows changes after applying stopwords.

![image](https://github.com/YoonjungChoi/CMPE255_TEAM10/blob/main/paper/images/fig.stopw.png)

**Observation:** we observed the removal of stop words. In first instance, 'out', 'for', 'more', 'set', 'me' is removed.

### Stemming

Stemming is the process of producing morphological variants of a root/base word. For example, words such as “Likes”, ”liked”, ”likely” and ”liking” will be reduced to “like” after stemming. There are different algorithms for stemming. Porter Stemmer, one of them, is a basic stemmer and it is straightforward and fast to run. Figure shows changes after applying stemming.

![image](https://github.com/YoonjungChoi/CMPE255_TEAM10/blob/main/paper/images/fig.stem.png)

**Observation:** we observed some changes of words. The 'crying' changed to 'cri' or 'acquisitions' changed to 'acquisit'.

### Lemmatization

Lemmatization is the process of grouping together the inflected forms of a word so they can be analysed as a single item, identified by the word's lemma, or dictionary form. Both stemming and lemmatization are word normalization techniques, but we can find the word in dictionary in case of lemmatization. For example, original words 'populated' changed 'popul' in Stemming, but it is not changed in lemmatization. Lemmatization is more better performed than Stemming []. We decided to apply lemmatization.

![image](https://github.com/YoonjungChoi/CMPE255_TEAM10/blob/main/paper/images/fig.lem.png)

**Observation:** we observed some changes of words. Unlike stemming, lemmatization made that 'crying' changed to 'cry' or 'acquisitions' changed to 'acquisition'.

## Data Visualization

After normalized text, we made data visualization by using word cloud. 

Figure shows wordcloud about disaster.

![image](https://github.com/YoonjungChoi/CMPE255_TEAM10/blob/main/paper/images/fig.wordcloud_1.png)

**Observation:** we discovered disaster tweets' related words; suicide, police, news, kill, attack, death, california, storm, flood. 

Figure shows wordcloud about non-disaster.

![image](https://github.com/YoonjungChoi/CMPE255_TEAM10/blob/main/paper/images/fig.wordcloud_0.png)

**Observation:** Non disaster tweets shows that time, want, great, feel, read, but also injury or emergency are found.

Figure represent histogram of the number of words at each sample.

![image](https://github.com/YoonjungChoi/CMPE255_TEAM10/blob/main/paper/images/fig.numWords_label.png)

**Observation:** 

## Word Embedding to transform data into numerical feature vectors

Word embedding is one of the most popular representation of document vocabulary. It is capable of capturing context of a word in a document, semantic and syntactic similarity, relation with other words. One of the biggest problems with text is that it is messy and unstructured, and machine learning algorithms need structured, properly defined fixed-length inputs. To train text on machine learning model, we need to transform 'text' feature(words or sentences) into fixed-length numerical feature vectors. There are a few method we can use to transform text into numerical feature vectors.


### CountVectorizer

Bag of Words model is a simplified representation used in natural language processing. A text is represented as the bag of its words, disregarding grammar and describes the occurrences of words with in a document. CountVectorizer can be used for bag of words model. This convert a collection of text documents to a matrics of token counts. CountVectorizer transforms the text into fixed-length vectors.

Figure shows occurrences of words by CountVectorizer .

![image](https://github.com/YoonjungChoi/CMPE255_TEAM10/blob/main/paper/images/fig.countvec.png)

**Observation:** we observed the number of occurrences of words. The 'deed' occurs around 4000.

### Term Frequency Inverse Document Frequency(Tf-Idf)

Some articles say that TF-IDF is better than Count Vectorizers because it not only focuses on the frequency of words present in the corpus but also provides the importance of the words. We can then remove the words that are less important for analysis, hence making the model building less complex by reducing the input dimensions.

Figure shows weighted occurrences of words by Tf-Idf .

![image](https://github.com/YoonjungChoi/CMPE255_TEAM10/blob/main/paper/images/fig.tfidf.png)

**Observation:** we observed the number of occurrences of words. The 'deed' occurs around 15000.

### Word2Vec

According to Wikipedia, Word2vec is a group of related models that are used to produce word embeddings. The word2vec algorithm uses a neural network model to learn word associations from a large corpus of text. Once trained, such a model can detect synonymous words or suggest additional words for a partial sentence. Word2Vec represents words in vector space in a way of similar meaning words are positioned in close locations but dissimilar words are placed far away. For non sequential models, we build a sentence embedding by averaging the values across all token embeddings output by Word2Vec. We followed the gensim-tutorial[] to visualize the relationship with words.

Figure shows visuliazation by word2vec.

![image](https://github.com/YoonjungChoi/CMPE255_TEAM10/blob/main/paper/images/fig.w2v_wildfire.png)

**Observation:** The blue are most similar words with the 'Wildfire' and green is most unrelated words with the 'Wildfire'

### Word2Vec with PCA applied

As principal component analysis is a strategy to reduce dimension, we applied PCA to reduce 100 components from Word2Vec with dimensionality of 300. 
Word2Vec of Gensim's default dimensionalty is 100. We tried to train and get better accuracy in several experiments. As a result, for instance, with Xgboost model, the case of creating 300 dimensionalty and appling PCA to create 100 pricipal components has better performance rather than word2vec with 100 dimensionality without PCA.

### Glove




We build four feature vectors from Count Vectorizer, Tf-Idf, Word2vec and Word2Vec with PCA applied. Glove and Word2Vec embedding are used for LSTM model.

# Methods
We use non-sequential models, such as Logistic Regression, SVM, Decision Tree, Random Tree, XGboost and a sequential model LSTM to solve this binary classification problem. Sequence models are the machine learning models that input or output sequences of data [].  We train four feature vectors on the basic models to find which feature vectors can yield better performance. For optimization of models, we re-train and adjust parameters on models with the selected feature vector. Even we did parameter optimization of models, some models do not result in siginificantly improvement, but the default parameter could be work well because it was designed general purpose. 

## Logistic Regression(LR)

Logistic Regression is a supervised machine learning algorithm that can be used to model the probability of a certain class or event. It is used when the data is linearly separable and the outcome is binary or dichotomous in nature. That means Logistic regression is usually used for Binary classification problems. Also, Logistic Regression is easier to implement, interpret, and very efficient to train. Logistic Regression is good accuracy for many simple data sets and it performs well when the dataset is linearly separable. So first we tried to train data sets on logistic regression.

Figure shows performance on Logistic Regression without modifying parameters.

![image](https://github.com/YoonjungChoi/CMPE255_TEAM10/blob/main/paper/images/fig.LogisticR.png)

**Observation:** We observed the count vectorizer feature set resulted in better accuracy(0.797) and precision(0.813).

From parameter optimization, we finalized parameters as C=0.15, penalty='l2', tol=0.001, solver='saga', random state=42, max iter=1000. We obtained the following result and confusion matrics:

![image](https://github.com/YoonjungChoi/CMPE255_TEAM10/blob/main/paper/images/fig.lr_final_score.png)

**Observation:** We observed that optimization does not improve significantly, but it improved to accuracy(0.800) and precision(0.837).

![image](https://github.com/YoonjungChoi/CMPE255_TEAM10/blob/main/paper/images/fig.logisticR_final_cm.png)

**Observation:** This confusion matrix shows the number of instance between prediction and actuals. This Logistic Regression model predicts 668 true positive (disaster) and 1160 true negative(non-disaster) instances.

## Support Vector Machine(SVM)
Support Vector Machine is a supervised learning model used for classification and regression problems. SVM can be used when data has exactly two classes. SVM classifies data by finding the best hyperplane that separates all data points of one class from those of the other class. The best hyperplane for an SVM means the one with the largest margin between the two classes. SVM can be used for our binary classification problem. We train four feature vectors on basic SVM, which means no changes of parameters.

Figure shows performance on SVM without modifying parameters.

![image](https://github.com/YoonjungChoi/CMPE255_TEAM10/blob/main/paper/images/fig.svm_basic.png)

**Observation:** We observed the count vectorizer feature resulted in better accuracy(0.799) rather than other feature vectors.

We adjusted parameters to yield better accuracy. In the final SVM model, it has default C value as 1, gamma value as 'auto', kernel value as 'sigmoid'. We obtained the result and confusion matrics of the model.

![image](https://github.com/YoonjungChoi/CMPE255_TEAM10/blob/main/paper/images/fig.svm_final_score.png)

**Observation:** We observed that optimization does not improve significantly. It improved as accuracy(0.800). 

![image](https://github.com/YoonjungChoi/CMPE255_TEAM10/blob/main/paper/images/fig.svm_final_cm.png)

**Observation:** This confusion matrix shows that SVM predicts 664 true positive(disaster)and 1163 true negative(non-disaster) instances.


## Decision Tree
A decision tree can be used for either regression or classification. Advantages of classification with Decision Trees are inexpensive to construct, extremely fast at classifying unknown records, easy to interpret for small-sized trees, accuracy comparable to other classification techniques for many simple data sets, excludes unimportant features. Thus, we try to train data on decision tree as well.

Figure shows performance on Decision Tree without modifying parameters.

![image](https://github.com/YoonjungChoi/CMPE255_TEAM10/blob/main/paper/images/fig.dt.png)

**Observation:** We observed the tf-idf feature set resulted in better accuracy(0.752) than other feature sets and count vectorizer feature resulted in better precision(0.731). Decision Tree decided to different node at each time, so result can be differ.

From parameter optimization, we finalized parameters as min samples split=8. We obtained the result and confusion matrics.

![image](https://github.com/YoonjungChoi/CMPE255_TEAM10/blob/main/paper/images/fig.dt_final_score.png)

**Observation:** We observed accuracy(0.0.756). We don't have significant improvement from parameter optimization.

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
[ ] Data For Everyone’ website, https://appen.com/datasets-resource-center

[ ] Kaggle Competition, https://www.kaggle.com/competitions/nlp-getting-started/data

[ ] Natural Language Toolkit, https://www.nltk.org/index.html

[ ] Naturalstemming-vs-lemmatization, https://www.baeldung.com/cs/stemming-vs-lemmatization
[1] Text Preprocessing for NLP (Natural Language Processing),Beginners to Master, https://medium.com/analytics-vidhya/text-preprocessing-for-nlp-natural-language-processing-beginners-to-master-fd82dfecf95

[2] Text Preprocessing in NLP, https://towardsdatascience.com/text-preprocessing-in-natural-language-processing-using-python-6113ff5decd8

[3] WordCloud, https://kavita-ganesan.com/python-word-cloud/#.YnnPcPPMIeU

[4] Word2Vec, gensim-word2vec-tutorial, https://www.kaggle.com/code/pierremegret/gensim-word2vec-tutorial/notebook

[5] Word2Vec, Wikipedia, https://en.wikipedia.org/wiki/Word2vec
