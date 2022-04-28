# CMPE255_TEAM10 [Natural Language Processing With Disaster Tweets]

**Team Members**

* Yoonjung Choi
* Abhiteja Mandava
* Ishaan Bhalla
* Sakruthi Avirineni

[Report](https://www.overleaf.com/read/gnxxgdkfggzs)

## Introduction Dataset
In our society, Social Network has been playing a crucial role in communication ways and analyzing social network has been important.
Especially, in case of emergency, people would start to annouce information via social network. Twitter is one of the popular communication medium.
Since more agencies want to monitor or track Twitter intelligently by using technologies.

![image](https://user-images.githubusercontent.com/20979517/164597834-91e22330-7d3c-49b1-87cc-af17eea57aba.png)

This Tweetter analysis is about prediction problem on whether a person's words are actually telling a disaster.
If someone says that 'On plus side LOOK AT THE SKY LIST NIGHT IT WAS ABLAZE',  People can unserstand it does not mean 'disaster' but metaphorically somthing. However, It is not clear for machine. Thus, our team10 will investigate what techniques are for NLP and explore them.

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
| target    | int64  | 0:non-disaster, 1:disaster  |



## Problem
This Problem is supervised learning, Binary classification Problem and Natural Language Processing.


This dataset have 'target' label having 0 or 1 

![image](https://user-images.githubusercontent.com/20979517/164575693-d0ee93c4-d68e-4697-a108-d616754b6eed.png)


We will use **Logistic Regression, SVM, Decision Tree** to solve this classification problem.

**Data Exploration**
* Loading dataset and understanding data

**Data Cleaning**
* Change all characters to lowercase
* Makes sure to remove URL, HTML, Emojis, ASCII and Punctuation. 

**Data Preprocessing Part1 Using [NLTK](https://www.nltk.org/index.html)**:
* Tokenize
* Remove Stopwords(Common words, example: are)
* PorterStemmer (Stemming is the important in NLP, example: asked -> ask)
* WordNetLemmatizer (example: peopl => people)

**Data Preprocessing Part2 to transform text to numerical value**
* apply CountVector
* apply TF-IDF
* apply Word2Vec

**Data Split**
* split each type of feature

**Modeling**:
* Build training model
* Train each type of t in model
* Find best tuning values of model
* Make sure to save all information(F1 Score, Precision, Recall, Accuracy)

**Emsemble**
* create best funing values of model
* create ensemble model
* Train/Test ensemble model with each type of data
* Make sure to save all information(F1 Score, Precision, Recall, Accuracy)

**Visualization**
* Visualize results
* Compare each models to find best results.

