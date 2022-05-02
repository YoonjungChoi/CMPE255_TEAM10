# CMPE255_TEAM10 [Natural Language Processing With Disaster Tweets]

**Team Members**

* Yoonjung Choi
* Abhiteja Mandava
* Ishaan Bhalla
* Sakruthi Avirineni

[Paper](https://www.overleaf.com/read/gnxxgdkfggzs)

## Introduction Dataset
Social Network Service have become not only an important sources of emergency information during disaster but also a medium for expressing immediate responses of warning, evacuation or rescue providing immediate assistance, assessing damage, continuing assistance and the immediate restoration or construction of infrastructure. As a result, predicting context of SNS is a crucial concern in our society. Also, more agencies want to monitor or track Twitter intelligently by using technologies. This paper can be utilized to track, monitor and predict disasters from content of SNS and would help making prediction model.

Twitter is one of the popular communication medium and people would start to annouce information via tweeter in disaster situation. This paper on Tweetter analysis is about prediction problem on whether a person's words are actually telling a disaster. Here is example:
![image](https://user-images.githubusercontent.com/20979517/164597834-91e22330-7d3c-49b1-87cc-af17eea57aba.png)
If someone says that 'On plus side LOOK AT THE SKY LIST NIGHT IT WAS ABLAZE',  People can understand it does not mean 'disaster' but metaphorically somthing. However, It is not clear for machine. 

Thus, our TEAM10 will investigate what techniques are for NLP and explore them. Other submission of Kaggle used similar preprocessing but used single model or even in case of ensemble, it trained with the same data set.

**However, we tried to find best combination of feature vector(one of CountVectorizer, Tf-idf, Word2Vec or Word2Vec with PCA) and model and made custom voting classifier combined each combination, e.g. SVM with CountVectorizer, Decision Tree with Tf-Idf, Logistic Regression with CountVectorizer.**


**Source Dataset**

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
| target    | int64  | 0:non-disaster, 1:disaster  | => **label**


## Problem
This Problem is supervised learning, Binary classification Problem and Natural Language Processing.

**This dataset have 'target' label having 0 or 1**
![image](https://user-images.githubusercontent.com/20979517/164575693-d0ee93c4-d68e-4697-a108-d616754b6eed.png)


We will use **Logistic Regression, SVM, Decision Tree** to solve this classification problem.

* Logistic Regression

* SVM
SVM works relatively well when there is a clear margin of separation between classes.
SVM is more effective in high dimensional spaces.
SVM is effective in cases where the number of dimensions is greater than the number of samples.
SVM is relatively memory efficient.

* Decision Tree
A decision tree algorithm can be used to solve both regression and classification problems.



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
* apply CountVector
* apply TF-IDF
* apply Word2Vec
* apply Word2Vec with PCA

**5) Data Split**
* split each type of feature

**6) Modeling**:
* Build training model
* Train each type of t in model
* Find best tuning values of model
* Make sure to save all information(F1 Score, Precision, Recall, Accuracy)

**7) Emsemble**
* create best funing values of model
* create ensemble model
* Train/Test ensemble model with each type of data
* Make sure to save all information(F1 Score, Precision, Recall, Accuracy)

**8) Visualization**
* Visualize results
* Compare each models to find best results.

