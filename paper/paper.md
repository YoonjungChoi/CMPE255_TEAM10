---
Title: "Natural Language Processing with Disaster Tweets"
Date: "May 2022"
Author: Team 10, CMPE 255, San José State University

header-includes: |
  \usepackage{booktabs}
  \usepackage{caption}
---

# Abstract

Social Network Services(SNS) have become not only an important source of emergency information during disasters but also a medium for expressing immediate responses of warning, evacuation or rescue. Because smartphones are so common, people can use them to broadcast an emergency in real time. As a result, more organizations (such as disaster relief organizations and news companies) are interested in programmatically monitoring tweets, however it's not always clear whether a person's statements are announcing a calamity. By analyzing context, we can utilize this study to track, monitor and predict disasters from the real time data and this study would help make prediction models.


# Introduction

Natural Language Processing (NLP) is a branch of artificial intelligence that helps computers understand and interpret human language and give computers the ability to understand written and spoken words. NLP is one of the most promising avenues for social media data processing and NLP has been widely used to analyze SNS and extract potential patterns.

Twitter is one of the popular SNS platforms and many tweets have been delivered in emergencies. Since there are demands for companies to utilize these tweets, we investigate natural language processes and develop prediction models having better performance in this paper.

The problem can be viewed as a binary classification problem and this project's goal is to figure out how to tell which tweets are about "genuine disasters" and which aren't. This project will involve experimentation on various machine learning models that will predict which tweets are about "actual disasters" and which aren't.

# Literature/Market Review

Although there are various existing analyses on this dataset like classifiers with word embeddings, there is no comparison on classifiers with word embeddings and also adaptation of PCA or ensemble with different word embeddings. We wanted to explore the impact of datasets constructed using different word vectorization methods (Countvectorizer, TF-IDF, Word2vec and Word2vec with Principal Component Analysis) on multiple models (Logistic regression, SVM, Decision Tree, Random Forest, XGBoost, LSTM Glove, Glove, LSTM with Word2Vec etc.). The results were compared based on different performance metrics such as Accuracy, Recall, Precision, F1 Score, Confusion Matrix, ROC Curve.
 

# Methods

To develop the models, we have designed a sequence of steps. The steps involved are :
1. Data Collection
2. Exploratory Data Analysis
3. Data Preprocessing 
4. Feature Selection
5. Feature Extraction
6. Data Split
7. Model Training, Parameter tuning
8. Model Evaluation 
9. Best Fit Analysis

Workflow:

![Workflow](https://user-images.githubusercontent.com/90536339/169720495-9b02d9a0-c9b2-44a6-81db-45631d38e7f5.png)

# Data Collection

The data set has been collected from the company figure-eight and originally shared on their ‘Data For Everyone’ website [2]. We found the data set from Kaggle Competition [3]. It contains 7613 tweets data with the following features:

# Exploratory Data Analysis, Data preprocessing and Feature Selection:
![image](https://github.com/YoonjungChoi/CMPE255_TEAM10/blob/main/paper/images/fig.dataset.png)
 
 Figure 1. Description of data set.

**Observation:** 
We observed that the data set has four features and one label.


![image](https://github.com/YoonjungChoi/CMPE255_TEAM10/blob/main/paper/images/fig.missing.png)
  
  Figure 2. The number of missing values in the data set.

**Observation:** 
We observed that the 'location' feature has many null values(2533) and the 'keyword' feature has null values(39).


![image](https://github.com/YoonjungChoi/CMPE255_TEAM10/blob/main/paper/images/fig.labelpie.png)
  
  Figure 3. Pie chart of 'target' label.

**Observation:** 
We cannot say that it has a perfectly balanced dataset, but slightly it is a balanced data set.


![image](https://github.com/YoonjungChoi/CMPE255_TEAM10/blob/main/paper/images/keywordSet.png)
  
  Figure 4. The number of unique words and common words at respective label(disaster and non-disaster).

**Observation:** 
1. The 'keyword' feature does not have many null values(39), but there are many common words. It has 221 words of disaster,  219 words of non-disaster.
2. There are 218 intersection words.
3. The differences that only 'disaster' tweets have are {'debris', 'wreckage', 'derailment'}. 
4. The differences that only 'non-disaster' tweets have are {'aftershock'}. 
5. Thus, we do not use keyword features.
 

![image](https://github.com/YoonjungChoi/CMPE255_TEAM10/blob/main/paper/images/locationEx.png)
  
  Figure 5. Partial samples of 'location' data set.

**Observation:** 
1. The 'location' feature has many null values(2533) and does not have format and it is not generated automatically. 
2. This feature has invalid data such as 'Happily Married with 2 kids', 'have car; will travel', 'peekskill. new york, 10566', or 'milky way'.
3.  We do not use 'location' as a feature.

**Concluded Observation of Exploratory Data Analysis and Data preprocessing:**
The 'id' feature is nominal data, which means that there is no meaningful information. Finally, it was decided to drop 'id', 'keyword', 'location' features and used only 'text' feature and 'target' label.


**Implementation:**
For cleaning text, we changed all words to lowercase, removed URL, HTML tags, Emojis, punctuation and ASCII codes [4],[5].

1. Sample of the original 'text' feature can be viewed below.

![image](https://github.com/YoonjungChoi/CMPE255_TEAM10/blob/main/paper/images/fig.oridata.png)
  
  Figure 6. Partial data set of 'text' feature.

**Observation:** We observed each sample has mixed data such as upper/lower cases, url, emojis.


2. Changes after Data Cleansing can be viewed below.

![image](https://github.com/YoonjungChoi/CMPE255_TEAM10/blob/main/paper/images/fig.cleantext.png)
  
  Figure 7. Partial data set after cleaning meaningless data.

**Observation:** Incomplete, duplicated, incorrect, and irrelevant data from the dataset is removed.

However, there is still an additional implementation that we can do to extract meaningful information from the cleaned data. We apply stemming or lemmatization to get normalized words. Natural Language Toolkit (NLTK)[6] provides easy-to-use interfaces for natural language processing.


### 3. Tokenization

Tokenization divides strings into lists of substrings. We used the library to find the words and punctuation in sentences.

Changes after applying tokenization can be viewed below.

![image](https://github.com/YoonjungChoi/CMPE255_TEAM10/blob/main/paper/images/fig.tok.png)
  
  Figure 8. Partial data set after applying tokenization on previous cleaned data.

**Observation:** We observed the separated words in each sample.


### 4. Stopwords

We removed commonly used words, such as "the", "a", "is", "in". 

Changes after applying stopwords can be viewed below.

![image](https://github.com/YoonjungChoi/CMPE255_TEAM10/blob/main/paper/images/fig.stopw.png)
  
  Figure 9. Partial data set after removing stopwords on tokenized data.

**Observation:** In the first sample, 'out', 'for', 'more', 'set', 'me' are removed.


### 5. Stemming

Stemming is the process of producing morphological variants of a root/base word. For example, words such as “Likes”, ”liked”, ”likely” and ”liking” will be reduced to “like” after stemming. There are different algorithms for stemming. Porter Stemmer, one of them, is a basic stemmer and it is straightforward and fast to run. 

Changes after applying stemming can be viewed below.

![image](https://github.com/YoonjungChoi/CMPE255_TEAM10/blob/main/paper/images/fig.stem.png)
 
 Figure 10. Partial data set after applying stemming on data without stopwords.

**Observation:** We observed some changes of words. The 'crying' changed to 'cri' or 'acquisitions' changed to 'acquisit'.


### 6. Lemmatization

Lemmatization is the process of grouping together the inflected forms of a word so they can be analyzed as a single item, identified by the word's lemma, or dictionary form. Both stemming and lemmatization are word normalization techniques, but we can find the word in the dictionary in case of lemmatization. 

For example, the original words 'populated' changed to 'popul' in Stemming, but it is not changed in lemmatization. Lemmatization has better performance than Stemming [7].

Changes after applying lemmatization can be viewed below.

![image](https://github.com/YoonjungChoi/CMPE255_TEAM10/blob/main/paper/images/fig.lem.png)
  
  Figure 11. Partial data set after applying lemmatization on data without stopwords.

**Observation:** We observed some changes of words. Unlike stemming, lemmatization made that 'crying' changed to 'cry' or 'acquisitions' changed to 'acquisition'.


### Data Visualization

After performing the aforementioned steps on the feature 'text', we made data visualization using word cloud[8]. 

Word cloud on disaster can be viewed below.

![image](https://github.com/YoonjungChoi/CMPE255_TEAM10/blob/main/paper/images/fig.wordcloud_1.png)

Figure 12. WordCloud on 'target' labeled as a disaster.

**Observation:** We discovered disaster tweets' related words : suicide, police, news, kill, attack, death, california, storm, flood.


Word cloud on non-disaster can be viewed below.

![image](https://github.com/YoonjungChoi/CMPE255_TEAM10/blob/main/paper/images/fig.wordcloud_0.png)

Figure 13. WordCloud on 'target' label as non disaster.

**Observation:** Non disaster tweets shows words unrelated of disasters: time, want, great, feel, read. Also, injury or emergency.

# Feature Extraction (feature = 'text')

## Word Embedding to transform data into numerical feature vectors

One of the biggest problems with text is that it is messy and unstructured, and machine learning algorithms need structured, properly defined fixed-length inputs. To train text on machine learning models, we need to transform 'text' features(words or sentences) into fixed-length numerical feature vectors.

Word embedding is one of the most popular representations of document vocabulary. It is capable of capturing context of a word in a document, semantic and syntactic similarity, relation with other words. There are a few methods we can use to transform text into numerical feature vectors.

### 1. CountVectorizer

The Bag-of-Words model is a simplified representation used in NLP. A text is represented as the bag of its words, disregarding grammar, and describes the occurrences of words within a document. CountVectorizer can be used for a bag of words model. 

CountVectorizer can be used for a bag of words model. This converts a collection of text documents to a matrix of token counts and transforms the text into fixed-length vectors.

Occurrences of words by CountVectorizer can be viewed below.

![image](https://github.com/YoonjungChoi/CMPE255_TEAM10/blob/main/paper/images/fig.countvec.png)

  Figure 14. Occurrences of words by CounterVectorizer.

**Observation:** We observed the frequencis of words. The  word 'deed' was present in 4000 odd occurences. The  word 'reason' was present in 10 odd occurences.

### 2. Term Frequency Inverse Document Frequency(Tf-Idf)

TF-IDF is considered to be better than Countvectorizer because it not only focuses on the frequency of words present in the corpus but also provides the importance of the words. Hence, we removed the words that are less important for analysis making the model building less complex.

Occurrences of words by Tf-Idf can be viewed below.

![image](https://github.com/YoonjungChoi/CMPE255_TEAM10/blob/main/paper/images/fig.tfidf.png)

Figure 15. Occurrences of words by Tf-Idf.

**Observation:** We observed weighted frequencis of words. The word 'deed' looks more weighted and the occurences changed to around 15000. The words 'reason' and 'reason earthquake' have appeared to weigh more as well.

### 3. Word2Vec

The word2vec algorithm uses a neural network model to learn word associations from a large corpus of text. Once trained, such a model can detect synonymous words or suggest additional words for a partial sentence [9]. Word2Vec represents words in vector space in a way of similar meaning: words are positioned in close locations but dissimilar words are placed far away. We followed the gensim-tutorial[10] to visualize the relationship with words.

Visualization of words by word2vec can be viewed below.

![image](https://github.com/YoonjungChoi/CMPE255_TEAM10/blob/main/paper/images/fig.w2v_wildfire.png)

Figure 16. Visualization of words by word2vec.

**Observation:** Blue represents most similar words associated with the 'Wildfire' and green represents most unrelated words associated with 'Wildfire'.

### 4. Word2Vec with PCA applied

Principal Component Analysis is a strategy to reduce dimensionality and to identify important relationships in data and to extract new features. According to the Gensim guide,  Word2Vec's default dimensionality is 100, so we are able to have directly 100 dimensions data set without PCA. However, we used PCA to reduce 100 components from Word2Vec with dimensionality of 300 since PCA not only reduces dimensionality but also extract new features. 

We have noticed that the implementation of  PCA has better performance than word2vec feature vector set with 100 dimensionality without PCA.

### 5. Glove

Glove stand for global vectors for word representation. It is an unsupervised learning algorithm developed by Stanford for generating word embeddings by aggregating global word-word co-occurrence matrices from a corpus [11]. It was decided to use the glove.6B.100d.txt file containing the glove vectors trained on the Wikipedia and GigaWord dataset.

The difference between the Word2Vec and Glove is the way of training. Glove is based on global word to word co-occurrence counts utilizing the entire corpus, on the other hand, Word2Vec uses co-occurrence within local neighboring words.




We build four feature vectors from Count Vectorizer, Tf-Idf, Word2vec and Word2Vec with PCA applied. Glove and Word2Vec embedding are used for the LSTM model. We are not able to apply PCA on Bag Of Words feature vector sets because of its sparsity. Transformed feature vectors sets have respective shape (7613,16270) from Count Vectorizer, shape (7613, 63245) from If-Idf, shape (7613,300) from Word2Vec, shape (7613,100) from Word2Vec applied PCA. 

# Models

We used non-sequential models, such as Logistic Regression, SVM, Decision Tree, Random Tree, XGboost, and a sequential model LSTM to solve this binary classification problem. 

For optimization of models, we re-trained and adjusted parameters on the models with the selected feature vector. However, some models did not result in significant improvement just for the simpler reason that the default parameters have worked well due to their general-purpose design.

## 1. Logistic Regression(LR)

Logistic Regression is a supervised machine learning algorithm that can be used to model the probability of a certain class or event. It is used when the data is linearly separable and the outcome is binary or dichotomous in nature [12]. That is the reason we have used LR to solve a Binary classification problem in our case.

Performance on Logistic Regression without modifying parameters can be viewed below.

|LR      |CountVectorizer|Tf-Idf|Word2Vec|Word2Vec+PCA|
|--------|---------------|------|--------|------------|
|Accuracy|          0.797| 0.776|   0.666|       0.761|

Table 1. Logistic Regression's accuracies with respective to feature vector sets.

**Observation:** We observed that the count vectorizer feature data set resulted in better accuracy(0.797). Also, Word2Vec applied PCA feature data set has better accuracy than word2vec feature data set.

**Parameter Tuning:**
We modified parameters of Logistic Regression to improve better accuracy with the count vectorizer feature data set. 

Penalty, a type of linear regression that uses shirikage, has three options; l1, l2, or elasticnet. The 'l1' is called Rasso Regression and this type of regularization can result in making coefficients zero, which means that some of the features are eliminated. On the other hand, the 'l2', called Ridge Regression, does not result in elimination of coefficients, which means that some features are affected a little. The 'elasticnet' is between 'l1' and 'l2'[13].  C is the inverse of regularization strength. The 'solver' is an algorithm to use in the optimization problem. For small datasets, ‘liblinear’ is a good choice [14]. We trained by making several models with three types of penalty, C, gamma, and other parameters.

Below is the code snippet:

```
lg_clf = LogisticRegression(C=0.45, penalty='l2', tol=0.01, solver='liblinear', random_state=42, max_iter=100)
```

Results and confusion matrix of the model can be viewed below.

|LR      | Accuracy | Recall | Precision | F1 Score |
|--------|----------|--------|-----------|----------|
|        |     0.801|   0.687|      0.826|     0.750|

Table 2. Logistic Regression's performance with Count Vectorizer feature vectors set.

**Observation:** We observed that the accuracy is improved (0.801). However, its not significant.

![image](https://github.com/YoonjungChoi/CMPE255_TEAM10/blob/main/paper/images/fig.logisticR_final_cm.png)

Figure 17. Confusion Matrix of Logistic Regression with Count Vectorizer feature vectors set.

**Observation:** This confusion matrix shows the number of samples between prediction and actuals. This Logistic Regression model predicts 683 true positive (disaster) and 1146 true negative(non-disaster) samples.

## 2. Support Vector Machine(SVM)

Support Vector Machine is a supervised learning model used for classification and regression problems. SVM can be used when data has exactly two classes. SVM classifies data by finding the best hyperplane that separates all data points of one class from those of the other class. The best hyperplane for an SVM means the one with the largest margin between the two classes [16]. That is the reason we have used SVM to solve a Binary classification problem in our case.

Performance on SVM without modifying parameters can be viewed below.

|SVM     |CountVectorizer|Tf-Idf|Word2Vec|Word2Vec+PCA|
|--------|---------------|------|--------|------------|
|Accuracy|          0.799| 0.761|   0.627|       0.712|

Table 3. SVM's accuracies with respective feature vector sets.

**Observation:** We observed that the count vectorizer feature vector set resulted in better accuracy(0.799) than other feature vectors' sets. Word2Vec applied PCA feature vector set has better accuracy than Word2vec feature vector set.

**Parameter Tuning:**

We adjusted parameters SVM to improve better accuracy with the count vectorizer feature data set. 

SVM has also parameters for regularization. C is the inverse strength of the regularization. Gamma is the kernel coefficient. If Gamma sets ‘scale’, then it uses 1 / (n_features * X.var()). SVM has a kernel parameter for kernel trick, which is a simple method where a non linear data is projected onto a higher dimension space so as to make it easier to classify the data where it could be linearly divided by a plane [17]. We trained by making several models with C(default=1), kernels, gamma(default='scale') and other parameters. Sigmoid Kernel refers to a neural network field and is equivalent to a two-layer, perceptron neural network. 

Below is the code snippet for optimazation: 

```
svm_clf = SVM(kernel='sigmoid')
```

Results and confusion matrix of the model can be viewed below.

|SVM     | Accuracy | Recall | Precision | F1 Score |
|--------|----------|--------|-----------|----------|
|        |     0.800|   0.688|      0.839|     0.744|

Table 4. SVM's  performance with the Count Vectorizer feature vector set.

**Observation:** We observed that optimization does not improve significantly as accuracy is 0.800, but default parameters could have good performance without adjusting them because they are designed for general purpose.

![image](https://github.com/YoonjungChoi/CMPE255_TEAM10/blob/main/paper/images/fig.svm_final_cm.png)

Figure 18. Confusion Matrix of SVM with Count Vectorizer feature vectors set.

**Observation:** This confusion matrix shows that SVM predicts 664 true positive(disaster)and 1163 true negative(non-disaster) samples. Compared to Logistic Regression model, SVM predict more non-disaster samples than disaster samples.

## Decision Tree
A decision tree can be used for either regression or classification. Decision Tree uses 'entropy' or 'gini' to calculate impurity of split and obtains information gain, and then decides which node splits in a way of having as much as possible higher information gain. Advantages of classification with Decision Trees are inexpensive to construct, extremely fast at classifying unknown records, easy to interpret for small-sized trees, accuracy comparable to other classification techniques for many simple data sets, and excludes unimportant features. Thus, we try to train data on decision trees as well. Table 5 shows performance on Decision Tree without modifying parameters. 


|SVM     |CountVectorizer|Tf-Idf|Word2Vec|Word2Vec+PCA|
|--------|---------------|------|--------|------------|
|Accuracy|          0.748| 0.751|   0.667|       0.671|

Table 5. Decision Tree's accuracies with respective feature vector sets.

**Observation:** We observed the tf-idf feature vectors set resulted in better accuracy(0.751) than other feature vectors sets. Since the decision Tree decides on a different node at each time, results could be differ considering that the difference with accuracy of the Countvectorizer feature vector set are small. Also, we observed that accuracies from feature vectors sets of Word2Vec and Word2Vec applied PCA are not much big difference rather than cases of Logistic Regression and SVM.

We adjusted parameters DecisionTree to improve better accuracy with the Tf-Idf feature vectors set. As default values, Decision Tree has 'gini' criterion, 'best' split strategy, consider all number of features when looking for best split. We can see the detail of parameters in Sklearn documentation[18].
The 'min_samples_split' parameter indicates the minimum number of samples required to split an internal node. For experiments, I used random_state to make reproducible results.

```
clf = DecisionTreeClassifier(min_samples_split=10, random_state=27)
```

We obtained the result and confusion matrix of the model.

|DT      | Accuracy | Recall | Precision | F1 Score |
|--------|----------|--------|-----------|----------|
|        |     0.756|   0.681|      0.737|     0.708|

Table 6. Decision Tree's  performance with If-Idf feature vectors set.

![image](https://github.com/YoonjungChoi/CMPE255_TEAM10/blob/main/paper/images/fig.dt_final_cm.png)

Figure 19. Confusion Matrix of Decision Tree with If-Idf feature vectors set.

**Observation:** This confusion matrix shows that SVM predicts 677 true positive(disaster) and 1049 true negative (non-disaster) samples. Compared to Logistic Regression and SVM, Decision Tree predict less non-disaster samples.

## Random Forest

Random Forest is a supervised learning algorithm. It can be used for both classification and regression. However, it's mainly used for classification problems. A forest comprises trees and it's said that the more trees it has, the more robust the forest is. Random Forest is a set of multiple decision trees. Random Forest creates decision trees on randomly selected data samples, gets predictions from each tree and selects the best solution by means of voting.

Decision trees may suffer from overfitting but random forest prevents overfitting by creating trees on random subsets. Decision trees are computationally faster.

[table, image for score, confusion matrix]

**Observation:** 

## Xgboost

[table, image for score, confusion matrix]

**Observation:** 

## LSTM

Long Short-Term Memory (LSTM) networks are a type of recurrent neural network capable of learning order dependence in sequence prediction problems. This is a behavior required in complex problem domains like machine translation, speech recognition, and more. LSTMs are a complex area of deep learning. The Long Short Term Memory architecture was motivated by an analysis of error flow in existing RNNs which found that long time lags were inaccessible to existing architectures, because backpropagated error either blows up or decays exponentially. 

An LSTM layer consists of a set of recurrently connected blocks, known as memory blocks. These blocks can be thought of as a differentiable version of the memory chips in a digital computer. Each one contains one or more recurrently connected memory cells and three multiplicative units – the input, output and forget gates – that provide continuous analogues of write, read and reset operations for the cells. … The net can only interact with the cells via the gates.

**Working**

An LSTM has four “gates”: forget, remember, learn and use(or output). It also has three inputs: long-term memory, short-term memory, and E. (E is some training example/new data). Step 1: When the 3 inputs enter the LSTM they go into either the forget gate, or learn gate. The long-term info goes into the forget gate, where, shocker, some of it is forgotten (the irrelevant parts). The short-term info and “E” go into the learn gate. This gate decides what info will be learned. Step 2: information that passes the forget gate (it is not forgotten, forgotten info stays at the gate) and info that passes learn gate (it is learned) will go to the remember gate (which makes up the new long term memory) and the use gate (which updates short term memory +is the outcome of the network).

![image](https://user-images.githubusercontent.com/46517523/169108963-a9b16f57-9ca6-4239-bb4e-d3d48bd41551.png)
@Ishaan this image looks not understandable.
```
@Ishaan : code here
```

@Ishaan : code explanations here

![image](https://user-images.githubusercontent.com/46517523/169112300-5acf0cf2-6c2e-48ce-82ac-b12135fd2c92.png)

**Observation:**

We tried to set random seed to make reproducible results, still LSTM model fluctuates its accuracy score from 0.803 to 0.813.

| LSTM+Glove | Accuracy | Recall  | Precision | F1 Score |
|------------|----------|--------|-----------|-----------|
|            |     0.810|   0.791|      0.764|      0.777|

**Observation:** LSTM has the higher accuracy rather than other models that we tried so far.

We trained the same LSTM model with Word2Vec for comparison. Results are shown the below Table.

|LSTM  + W2V | Accuracy | Recall | Precision | F1 Score |
|------------|----------|--------|-----------|----------|
|            |     0.635|   0.548|      0.927|     0.688|

**Observation:** In this case, the model with Word2Vec has worst performance rather than the model with Glove. However, this model have many possibility to improve considering that we do not use optimization. We will apply optimization in further study since we found that there are many parameters to improve performance [keras-].


## Ensemble
Ensemble methods are techniques that create multiple models and then combine them to produce improved results. Ensemble methods usually produce more accurate solutions than a single model would.  We have four different feature sets and random_state parameters enable split feature vector sets in the same way, which means we can use ensemble models on our own. Based on the voting way, First ensemble model consisted of non sequential models; Logistic Regression with Count vectorizer, SVM with Counter vectorizer, Decision Tree with Tf-Idf, RandomForeset with counter vectorizer, Xgboost with word2vec applied PCA.

Table shows the accuracy, recall, precision, and f1 score of the ensemble model.

|Ensemble  | Accuracy | Recall | Precision | F1 Score |
|----------|----------|--------|-----------|----------|
|          |     0.811|   0.701|      0.838|     0.764|

![image](https://github.com/YoonjungChoi/CMPE255_TEAM10/blob/main/paper/images/en1_cm.png)

**Observation:** When we use ensemble model, accuracy is better than accuracy from respective non sequential models.

Next ensemble model is the first ensemble model adding the sequential model, LSTM with Glove. This ensemble model also fluctuate performance depends on performance of LSTM model. The below table shows the accuracy, recall, precision, and f1 score when LSTM with Glove model has 0.811 accuracy. 

|Ensemble  | Accuracy | Recall | Precision | F1 Score |
|----------|----------|--------|-----------|----------|
|          |     0.813|   0.687|      0.856|     0.762|

![image](https://github.com/YoonjungChoi/CMPE255_TEAM10/blob/main/paper/images/en2_cm.png)

**Observation:** When we use ensemble model, accuracy is better than accuracy from respective models. 

# Comparisons

## Performance Metrics

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

From non sequential modeling on one classifier and four word embeddings, we expected to find the best combination having better accuracy. We found that when Logistic Regression, SVM, Random Tree, XGBoost trained on CountVectorizer feature vectors set, Decision Tree trained on Tf-Idf feature vectors set, and XGBoost trained on Word2Vec applied PCA feature vectors set, they resulted in the highest accuracy at each experiment. In addition, although we were not able to acquire much improvement by optimizing parameters, we obtained a little improvement at least and we learned that sometimes default parameters could have good accuracy because they are supposed to work for general purpose. On the ensemble model of non sequential models, we found that the ensemble model has better performance than cases using a single model.

![image](https://github.com/YoonjungChoi/CMPE255_TEAM10/blob/main/paper/images/performance.png)

From sequential modeling on LSTM and two word embeddings; word2vec and glove, when LSTM trained on Glove feature vectors set, we were able to have the highest accuracy. LSTM has better accuracy rather than other models. However, we can also think that given that the difference is not very big, non sequential models can have as much performance as the LSTM model in this experiment.

We obtained ROC Curve and AUC(Area under the ROC Curve) of respective combinations of models and word embeddings. Since we used voting way of ensemble, we was not able to make ROC curve and AUC of ensemble model. As we can see in Figure and Table, LSTM with Glove has the highest area under the curve(0.881).
<!-- ![image]([https://github.com/YoonjungChoi/CMPE255_TEAM10/blob/main/paper/images/ROCCurve.png]) -->
<!-- ![image]([https://github.com/YoonjungChoi/CMPE255_TEAM10/blob/main/paper/images/ROC_Curves.png]) -->
<!-- ![image](https://github.com/YoonjungChoi/CMPE255_TEAM10/blob/main/paper/images/ROC_Curves.png) -->
![image](https://github.com/YoonjungChoi/CMPE255_TEAM10/blob/main/paper/images/ROCCurve.png)

| Models  | LR + CV | SVM + CV | DT + Tf-IDF | RT + CV | Xgb + W2v + PCA | LSTM+Glove |
|---------|---------|----------|-------------|---------|-----------------|------------|
| AUC     |    0.859|     0.854|        0.767|    0.852|            0.831|       0.881|   


Also, other submissions of Kaggle have done similar steps for preprocessing and applying models and many participants have 1 score in Leaderboard. They tried to train a single model or even in case of ensemble, they trained ensemble classifiers with the same data set. However, we have a little bit of a different direction. We expect that there would be a suitable combination of feature vector sets and models. So we will try to build models with different feature vector sets to find combinations to get better performance. We evaluated diverse models and we tried to make a comparison.

# Conclusions

In this analysis we experienced four prominent word embeddings and seven classification techniques of using an Fiure-Eight Company data set. LSTM with glove have good performance among individual models and Ensemble model with combined different feature vectors and classifiers in our experiment outperfomanced the other classifiers on each data set.

# Limitations And Future research

We obtained the qualified data set from the company, so we assumed that data is reliable. However, the fact that data could be not truthful is the main limitation of this study. Overcoming these limitations can be done in future research. By distinguishing reliability of data first, we can analyze and predict emergency situations properly. Additional study on deep learning algorithms should be continued. When we made models on LSTM with word embeddings, we faced many difficulties about understanding complicated algorithms itself and choosing diverse optimazation options. In furthur study, we will continue dealing with our concerns.

# References

[1] NLP Market Research, https://www.statista.com/statistics/607891/worldwide-natural-language-processing-market-revenues/

[2] Data For Everyone’ website, https://appen.com/datasets-resource-center

[3] Kaggle Competition, https://www.kaggle.com/competitions/nlp-getting-started/data

[4] Text Preprocessing for NLP (Natural Language Processing),Beginners to Master, https://medium.com/analytics-vidhya/text-preprocessing-for-nlp-natural-language-processing-beginners-to-master-fd82dfecf95

[5] Text Preprocessing in NLP, https://towardsdatascience.com/text-preprocessing-in-natural-language-processing-using-python-6113ff5decd8

[6] Natural Language Toolkit, https://www.nltk.org/index.html

[7] Natural Stemming-vs-lemmatization, https://www.baeldung.com/cs/stemming-vs-lemmatization

[8] WordCloud, https://kavita-ganesan.com/python-word-cloud/#.YnnPcPPMIeU

[9] Word2Vec Wikipedia, https://en.wikipedia.org/wiki/Word2vec

[10] Word2Vec, gensim-word2vec-tutorial, https://www.kaggle.com/code/pierremegret/gensim-word2vec-tutorial/notebook

[11] What is Glove?, https://medium.com/analytics-vidhya/word-vectorization-using-glove-76919685ee0b

[12] Glove File from Kaggle, https://www.kaggle.com/datasets/danielwillgeorge/glove6b100dtxt

[13] Logistic Regression, https://www.sciencedirect.com/topics/computer-science/logistic-regression

[14] Logstic Regression API, https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

[15] Logistic Regression Sparsity, https://scikit-learn.org/stable/auto_examples/linear_model/plot_logistic_l1_l2_sparsity.html

[16] SVM How it works, https://www.analyticsvidhya.com/blog/2017/09/understaing-support-vector-machine-example-code/

[17] SVM Kernel Trick, https://datamites.com/blog/support-vector-machine-algorithm-svm-understanding-kernel-trick/

[18] Decision Tree, https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html

[keras-] Keras Optimizers API, https://keras.io/api/optimizers/
