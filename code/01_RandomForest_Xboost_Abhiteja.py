#!/usr/bin/env python
# coding: utf-8

# ### Importing requires libraries



pip install xgboost




import numpy as np
import pandas as pd
import os
import re
import nltk
import string
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score
import xgboost as xgb
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Reading csv into pandas dataframe


tweets = pd.read_csv('train.csv')
tweets.head(5)


tweets.info()

tweets.shape


# #### Location wise values

tweets['location'].value_counts()

# #### Keywords count

tweets['keyword'].value_counts()


# ### Data Cleaning


cols_to_drop = ['location','id']

tweets = tweets.drop(cols_to_drop, axis = 1)


tweets["CleanText"] = tweets["text"].apply(lambda x: x.lower())
tweets.head()



tweets["CleanText"] = tweets["CleanText"].apply(lambda x: re.sub(r"https?://\S+|www\.\S+", "",x))


def removeHTML(text):
    html = re.compile(r"<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});")
    return re.sub(html, "", text)

tweets["CleanText"] = tweets["CleanText"].apply(lambda x: removeHTML(x))


def removeEmojis(text):
    emoji_pattern = re.compile("["
                       u"\U0001F600-\U0001F64F"  # emoticons
                       u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                       u"\U0001F680-\U0001F6FF"  # transport & map symbols
                       u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                       u"\U00002702-\U000027B0"
                       u"\U000024C2-\U0001F251"
                       u"\U0001f926-\U0001f937"
                       u"\u200d"
                       u"\u2640-\u2642" 
                       "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

tweets["CleanText"] = tweets["CleanText"].apply(lambda x: removeEmojis(x))


def RemovePunctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

tweets["CleanText"] = tweets["CleanText"].apply(lambda x: RemovePunctuation(x))


def RemoveASCII(text):
  return re.sub(r'[^\x00-\x7f]', "", text)

tweets["CleanText"] = tweets["CleanText"].apply(lambda x: RemoveASCII(x))


tweets.head(30)

# #### There are 61 missing values in Keyword feature

features = ['keyword']
for feat in features : 
    print("The number of missing values in "+ str(feat)+" are "+str(tweets[feat].isnull().sum()))
    
# ### Analysis


plot_size = plt.rcParams["figure.figsize"]
print(plot_size[0])
print(plot_size[1])

plot_size[0]=8
plot_size[1]=6
plt.rcParams["figure.figsize"]=plot_size


# ##### target = 1 is when disaster occurs and target = 0 is when disaster does not occur

# In[21]:


x=tweets.target.value_counts()
sns.barplot(x.index,x)
plt.gca().set_ylabel('samples')




def _corpus(target):
    corpus=[] # document
    
    for x in tweets[tweets['target']==target]['text'].str.split():
        for i in x:
            corpus.append(i)
    return corpus


