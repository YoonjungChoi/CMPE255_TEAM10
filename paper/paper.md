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

Natural Language Processing (NLP) is a branch of artificial intelligence that helps computers understand and interpret human language and give computers the ability to understand written and spoken words. Worldwide revenue from the NLP market forecasts to be almost 14 times larger in 2025 than it was in 2017, increasing from around three billion U.S. dollars in 2017 to over 43 billion in 2025 [1]. NLP is one of the most promising avenues for social media data processing and NLP has been widely used to analyze SNS and extract potential patterns.

SNS allows people to connect with each other and give us easy and instant communication tools in real time. SNS has been playing a crucial role in interacting in our society. SNS is able to be utilized as an important vehicle of emergency information during disasters to deliver immediate responses of warning, evacuation or rescue, providing immediate assistance, assessing damage, continuing assistance and the immediate restoration or construction of infrastructure.

Twitter is one of the popular SNS platforms and many tweets have been delivered in emergency situations. Since there are demands for companies to utilize these tweets, we investigate natural language processes and develop prediction models having better performance in this paper. Many submissions used a few classifiers with word embeddings, but there is no comparison on classifiers with word embeddings and additional adaptation  of PCA or ensemble with different word embeddings. So in this paper, we evaluate each model having different word embeddings. For preprocessing, we clean the data from unnecessary data such as URL, Emojis or HTML tags and normalize the data by using useful algorithms; tokenizer, stopwords and lemmatization. We transform cleaned data into four feature vectors sets by using Countvectorizer, Term Frequency Inverse Document Frequency, Word2Vec and Word2Vec with principal component analysis(PCA) applied, and trained them on non sequential models; Logistic Regression, Support Vector Machine, Decision Tree, RandomForeset, XGboost. We also train a sequential model such as LSTM with Word2Vec and Glove. We found each combination between word embeddings and classifiers having better performance. Lastly, we build an ensemble model consisting of the combinations we found unlike an existing ensemble model using the same data set. The results were compared based on different performance metrics such as Accuracy, Recall, Precision, F1 Score, Confusion Matrix, ROC Curve.

# Data Exploration

## Data Set

The data set has been collected from the company figure-eight and originally shared on their ‘Data For Everyone’ website [2]. We found the data set from Kaggle Competition [3]. It contains 7613 tweets data with the following features:

![image](https://github.com/YoonjungChoi/CMPE255_TEAM10/blob/main/paper/images/fig.dataset.png)

Figure 1. Description of data set.

**Observation:** We observed that the data set has four features and one label.

Figure 2 shows counts of missing values.

![image](https://github.com/YoonjungChoi/CMPE255_TEAM10/blob/main/paper/images/fig.missing.png)

Figure 2. The number of missing values in the data set.

**Observation:** We observed that the 'location' feature has many null values(2533) and the 'keyword' feature has null values(39).

Figure 3 shows the percentage of 'target' labels.

![image](https://github.com/YoonjungChoi/CMPE255_TEAM10/blob/main/paper/images/fig.labelpie.png)

Figure 3. Pie chart of 'target' label.

**Observation:** we cannot say that it has a perfectly balanced dataset, but slightly it is a balanced data set.

Figure 4 shows the number of unique words of the 'keyword' feature based on each target.

![image](https://github.com/YoonjungChoi/CMPE255_TEAM10/blob/main/paper/images/keywordSet.png)

Figure 4. The number of unique words and common words at respective label(disaster and non-disaster).

**Observation:** the keyword feature does not have many null values(39), but there are many common words. The 'keyword' feature of disaster has 221 words, and The 'keyword' feature of non-disaster has 219 words. There are 218 intersection words. The differences that only 'disaster' tweets have are {'debris', 'wreckage', 'derailment'}. The difference that only 'non-disaster' tweets have are {'aftershock'}. Thus, we do not use keyword features.

Figure 5 shows a few samples of the 'location' feature.

![image](https://github.com/YoonjungChoi/CMPE255_TEAM10/blob/main/paper/images/locationEx.png)

Figure 5. Partial samples of 'location' data set.

**Observation:** 'location' feature has many null values(2533) and does not have format and it is not generated automatically. This feature has invalid data such as 'Happily Married with 2 kids', 'have car; will travel', 'peekskill. new york, 10566', or 'milky way'. We do not use 'location' as a feature.

The 'id' feature is nominal data, which means that there is no meaningful information. Finally, we dropped 'id', 'keyword', 'location' features and used only 'text' feature and 'target' label.


## Data Preprocessing
We should clean data to remove meaningless data because properly cleaned data enable us to do good text analysis and help in making accurate decisions for our problem. For cleaning text, we changed all words to lowercase, removed URL, HTML tags, Emojis, punctuation and ASCII codes [4],[5].

Figure 6 shows parital of the original 'text' feature.

![image](https://github.com/YoonjungChoi/CMPE255_TEAM10/blob/main/paper/images/fig.oridata.png)

Figure 6. Partial data set of 'text' feature.

**Observation:** We observed each sample has mixed data such as upper/lower cases, url, emojis.

Figure 7 shows changes after cleaning meaningless data.

![image](https://github.com/YoonjungChoi/CMPE255_TEAM10/blob/main/paper/images/fig.cleantext.png)

Figure 7. Partial data set after cleaning meaningless data.

**Observation:** We observed there is removal of unnecessary data.

We have cleaned data through the previous step, but there are still a few ways that we can do to extract meaningful information from the cleaned data. We apply stemming or lemmatization to get normalized words. Natural Language Toolkit (NLTK)[6] provides easy-to-use interfaces for natural language processing.

### Tokenization

Tokenization divides strings into lists of substrings. We can use the library to find the words and punctuation in sentences.
The Figure 8 shows changes after applying tokenization.

![image](https://github.com/YoonjungChoi/CMPE255_TEAM10/blob/main/paper/images/fig.tok.png)

Figure 8. Partial data set after applying tokenization on previous cleaned data.

**Observation:** We observed the separated words in each sample.

### Stopwords

We remove commonly used words, such as "the", "a", "is", "in". Figure 9 shows changes after applying stopwords.

![image](https://github.com/YoonjungChoi/CMPE255_TEAM10/blob/main/paper/images/fig.stopw.png)

Figure 9. Partial data set after removing stopwords on tokenized data.

**Observation:** We observed the removal of stop words. In the first sample, 'out', 'for', 'more', 'set', 'me' is removed.

### Stemming

Stemming is the process of producing morphological variants of a root/base word. For example, words such as “Likes”, ”liked”, ”likely” and ”liking” will be reduced to “like” after stemming. There are different algorithms for stemming. Porter Stemmer, one of them, is a basic stemmer and it is straightforward and fast to run. Figure shows changes after applying stemming.

![image](https://github.com/YoonjungChoi/CMPE255_TEAM10/blob/main/paper/images/fig.stem.png)

Figure 10. Partial data set after applying stemming on data without stopwords.

**Observation:** We observed some changes of words. The 'crying' changed to 'cri' or 'acquisitions' changed to 'acquisit'.

### Lemmatization

Lemmatization is the process of grouping together the inflected forms of a word so they can be analyzed as a single item, identified by the word's lemma, or dictionary form. Both stemming and lemmatization are word normalization techniques, but we can find the word in the dictionary in case of lemmatization. For example, the original words 'populated' changed to 'popul' in Stemming, but it is not changed in lemmatization. Lemmatization is better performed than Stemming [7]. We decided to apply lemmatization.

![image](https://github.com/YoonjungChoi/CMPE255_TEAM10/blob/main/paper/images/fig.lem.png)

Figure 11. Partial data set after applying lemmatization on data without stopwords.

**Observation:** We observed some changes of words. Unlike stemming, lemmatization made that 'crying' changed to 'cry' or 'acquisitions' changed to 'acquisition'.

## Data Visualization

After normalized text, we made data visualization by using word cloud[8]. 

Figure 12 shows a word cloud about disaster.

![image](https://github.com/YoonjungChoi/CMPE255_TEAM10/blob/main/paper/images/fig.wordcloud_1.png)

Figure 12. WordCloud on 'target' labeled as a disaster.

**Observation:** we discovered disaster tweets' related words; suicide, police, news, kill, attack, death, california, storm, flood.

Figure 13 shows the word cloud about non-disaster.

![image](https://github.com/YoonjungChoi/CMPE255_TEAM10/blob/main/paper/images/fig.wordcloud_0.png)

Figure 13. WordCloud on 'target' label as non disaster.

**Observation:** Non disaster tweets shows words unrelated of disasters; time, want, great, feel, read. Also, injury or emergency, which can be used for non-disasters situations, are found as well.

## Word Embedding to transform data into numerical feature vectors

Word embedding is one of the most popular representations of document vocabulary. It is capable of capturing context of a word in a document, semantic and syntactic similarity, relation with other words. One of the biggest problems with text is that it is messy and unstructured, and machine learning algorithms need structured, properly defined fixed-length inputs. To train text on machine learning models, we need to transform 'text' features(words or sentences) into fixed-length numerical feature vectors. There are a few methods we can use to transform text into numerical feature vectors.

### CountVectorizer

Bag of Words model is a simplified representation used in natural language processing. A text is represented as the bag of its words, disregarding grammar and describes the occurrences of words within a document. CountVectorizer can be used for a bag of words model. This converts a collection of text documents to a matrix of token counts. CountVectorizer transforms the text into fixed-length vectors.

Figure 14 shows occurrences of words by CountVectorizer .

![image](https://github.com/YoonjungChoi/CMPE255_TEAM10/blob/main/paper/images/fig.countvec.png)

Figure 14. Occurrences of words by CounterVectorizer.

**Observation:** We observed frequencis of words. The 'deed' occurs around 4000. The 'reason' word occurs alot among 10 words in Figure.

### Term Frequency Inverse Document Frequency(Tf-Idf)

Some articles say that TF-IDF is better than Countvectorizer because it not only focuses on the frequency of words present in the corpus but also provides the importance of the words. We can then remove the words that are less important for analysis, hence making the model building less complex by reducing the input dimensions.

Figure 15 shows weighted occurrences of words by Tf-Idf.

![image](https://github.com/YoonjungChoi/CMPE255_TEAM10/blob/main/paper/images/fig.tfidf.png)

Figure 15. Occurrences of words by Tf-Idf.

**Observation:** We observed weighted frequencis of words. The 'deed' word looks weighted more and changed to around 15000. The 'reason' and 'reason earthquake' have more weighted rather than other words among 10 words in Figure.

### Word2Vec

According to Wikipedia, Word2vec is a group of related models that are used to produce word embeddings. The word2vec algorithm uses a neural network model to learn word associations from a large corpus of text. Once trained, such a model can detect synonymous words or suggest additional words for a partial sentence [9]. Word2Vec represents words in vector space in a way of similar meaning: words are positioned in close locations but dissimilar words are placed far away. For non sequential models, we build a sentence embedding by averaging the values across all token embeddings output by Word2Vec. We followed the gensim-tutorial[10] to visualize the relationship with words.

Figure 16 shows visualization by word2vec.

![image](https://github.com/YoonjungChoi/CMPE255_TEAM10/blob/main/paper/images/fig.w2v_wildfire.png)

Figure 16. Visualization of words by word2vec.

**Observation:** The blue are most similar words with the 'Wildfire' and green is most unrelated words with the 'Wildfire'.

### Word2Vec with PCA applied

Principal Component Analysis is a strategy to reduce dimensionality, to identify important relationships in data and to extract new features. According to the Gensim guide,  Word2Vec's default dimensionality is 100, so we are able to have directly 100 dimensions data set without PCA. However, we used PCA to reduce 100 components from Word2Vec with dimensionality of 300 since PCA not only reduces dimensionality but also extract new features. We tried to train and get better accuracy in several experiments. As a result, in this data set, usage of PCA has better performance rather than word2vec feature vector set with 100 dimensionality without PCA.

### Glove

Glove stand for global vectors for word representation. It is an unsupervised learning algorithm developed by Stanford for generating word embeddings by aggregating global word-word co-occurrence matrices from a corpus [11]. The resulting embeddings show interesting linear substructures of the word in vector space. Files with the pre-trained vectors Glove can be found in many sites like Kaggle[12]. We will use the glove.6B.100d.txt file containing the glove vectors trained on the Wikipedia and GigaWord dataset.

The difference between the Word2Vec and Glove is the way of training, which yield vectors with subtly different properties. Glove is based on global word to word co-occurrence counts utilizing the entire corpus, on the other hand, Word2Vec uses co-occurrence within local neighboring words.


We build four feature vectors from Count Vectorizer, Tf-Idf, Word2vec and Word2Vec with PCA applied. Glove and Word2Vec embedding are used for the LSTM model. We are not able to apply PCA on Bag Of Words feature vector sets because of its sparsity. Transformed feature vectors sets have respective shape (7613,16270) from Count Vectorizer, shape (7613, 63245) from If-Idf, shape (7613,300) from Word2Vec, shape (7613,100) from Word2Vec applied PCA. 

# Methods
We use non-sequential models, such as Logistic Regression, SVM, Decision Tree, Random Tree, XGboost and a sequential model LSTM to solve this binary classification problem. Sequence models are the machine learning models that input or output sequences of data [].  We train four feature vectors on the basic models to find which feature vectors can yield better performance. For optimization of models, we re-train and adjust parameters on models with the selected feature vector. Even though we did parameter optimization of models, some models did not result in significant improvement, but the default parameter could work well because it was designed for general purpose. 

## Logistic Regression(LR)

Logistic Regression is a supervised machine learning algorithm that can be used to model the probability of a certain class or event. It is used when the data is linearly separable and the outcome is binary or dichotomous in nature [12]. That means Logistic regression is usually used for Binary classification problems. Also, Logistic Regression is easier to implement, interpret, and very efficient to train. Logistic Regression is good accuracy for many simple data sets and it performs well when the dataset is linearly separable. So first we tried to train data sets on logistic regression.

Table 1 shows performance on Logistic Regression without modifying parameters.

|LR      |CountVectorizer|Tf-Idf|Word2Vec|Word2Vec+PCA|
|--------|---------------|------|--------|------------|
|Accuracy|          0.797| 0.776|   0.666|       0.761|

Table 1. Logistic Regression's accuracies with respective feature vector sets.

**Observation:** We observed the count vectorizer feature data set resulted in better accuracy(0.797). Also, Word2Vec applied PCA feature data set has better accuracy than word2vec feature data set.

We modified parameters of Logistic Regression to improve better accuracy with the count vectorizer feature data set. Logistic Regression has parameters for regularization, which can be used to train models that generalize better on unseen data, by preventing the algorithm from overfitting the training dataset. Penalty, a type of linear regression that uses shirikage, has three options; l1, l2, or elasticnet. The 'l1' is called Rasso Regression and this type of regularization can result in making coefficients zero, which means that some of the features are eliminated. On the other hand, the 'l2', called Ridge Regression, does not result in elimination of coefficients, which means that some features are affected a little. The 'elasticnet' is between 'l1' and 'l2'[13].  C is the inverse of regularization strength. The 'solver' is an algorithm to use in the optimization problem. For small datasets, ‘liblinear’ is a good choice [14]. We trained by making several models with three types of penalty, C, gamma, and other parameters.

This is the code snippet.

```
lg_clf = LogisticRegression(C=0.45, penalty='l2', tol=0.01, solver='liblinear', random_state=42, max_iter=100)
```

We obtained the below result and confusion matrices of the model.

|LR      | Accuracy | Recall | Precision | F1 Score |
|--------|----------|--------|-----------|----------|
|        |     0.801|   0.687|      0.826|     0.750|

Table 2. Logistic Regression's performance with Count Vectorizer feature vectors set.

**Observation:** We observed that optimization does not improve significantly, but it improved to accuracy(0.801).

![image](https://github.com/YoonjungChoi/CMPE255_TEAM10/blob/main/paper/images/fig.logisticR_final_cm.png)

Figure 17. Confusion Matrix of Logistic Regression with Count Vectorizer feature vectors set.

**Observation:** This confusion matrix shows the number of samples between prediction and actuals. This Logistic Regression model predicts 683 true positive (disaster) and 1146 true negative(non-disaster) samples.

## Support Vector Machine(SVM)
Support Vector Machine is a supervised learning model used for classification and regression problems. SVM can be used when data has exactly two classes. SVM classifies data by finding the best hyperplane that separates all data points of one class from those of the other class. The best hyperplane for an SVM means the one with the largest margin between the two classes [16]. SVM can be used for our binary classification problem. We train four feature vectors on basic SVM, which means no changes of parameters.

Table shows performance on SVM without modifying parameters.

|SVM     |CountVectorizer|Tf-Idf|Word2Vec|Word2Vec+PCA|
|--------|---------------|------|--------|------------|
|Accuracy|          0.799| 0.761|   0.627|       0.712|

Table 3. SVM's accuracies with respective feature vector sets.

**Observation:** We observed that the count vectorizer feature vector set resulted in better accuracy(0.799) rather than other feature vectors sets. Word2Vec applied PCA feature vector set has better accuracy than Word2vec feature vector set.

We adjusted parameters SVM to improve better accuracy with the count vectorizer feature data set. SVM has also parameters for regularization. C is the inverse strength of the regularization. Gamma is the kernel coefficient. If Gamma sets ‘scale’, then it uses 1 / (n_features * X.var()). SVM has a kernel parameter for kernel trick, which is a simple method where a non linear data is projected onto a higher dimension space so as to make it easier to classify the data where it could be linearly divided by a plane [17]. We trained by making several models with C(default=1), kernels, gamma(default='scale') and other parameters. Sigmoid Kernel refers to a neural network field and is equivalent to a two-layer, perceptron neural network. The below is the code snippet for optimazation. As a result, default parameters have better accuracy. 

```
svm_clf = SVM(kernel='sigmoid')
```

We obtained the result and confusion matrix of the model.

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

![image](https://github.com/YoonjungChoi/CMPE255_TEAM10/blob/main/paper/images/ROCCurve.png)

| Models  | LR + CV | SVM + CV | DT + Tf-IDF | RT + CV | Xgb + W2v + PCA | LSTM+Glove |
|---------|---------|----------|-------------|---------|-----------------|------------|
| AUC     |    0.859|     0.854|        0.767|    0.852|            0.831|       0.881|   


Also, other submissions of Kaggle have done similar steps for preprocessing and applying models and many participants have 1 score in Leaderboard. They tried to train a single model or even in case of ensemble, they trained ensemble classifiers with the same data set. However, we have a little bit of a different direction. We expect that there would be a suitable combination of feature vector sets and models. So we will try to build models with different feature vector sets to find combinations to get better performance. We evaluated diverse models and we tried to make a comparison.

# Conclusions

We obtained the qualified data set from the company, so we assumed that data is reliable. However, the fact that data could be not truthful is the main limitation of this study. Overcoming these limitations can be done in future research. By distinguishing reliability of data first, we can analyze and predict emergency situations properly. Additional study on deep learning algorithms should be continued. When we made models on LSTM with word embeddings, we faced many difficulties about understanding complicated algorithms itself and choosing diverse optimazation options. In furthur study, we will continue dealing with our concerns.

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
