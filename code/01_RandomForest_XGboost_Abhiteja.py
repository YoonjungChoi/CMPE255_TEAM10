#!/usr/bin/env python
# coding: utf-8

# ### Importing requires libraries

# In[1]:


pip install xgboost


# In[3]:


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
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Reading csv into pandas dataframe

# In[4]:


tweets = pd.read_csv('train.csv')
tweets.head(5)


# In[5]:


#Displaying a concise summary of the DataFrame
tweets.info()


# In[5]:


##Displaying number of rows and columns of the DataFrame
tweets.shape


# #### Location wise values

# In[6]:


tweets['location'].value_counts()


# #### Keywords count

# In[7]:


tweets['keyword'].value_counts()


# ### Data Cleaning

# In[8]:


cols_to_drop = ['location','id']


# In[9]:


tweets = tweets.drop(cols_to_drop, axis = 1)


# In[10]:


tweets["CleanText"] = tweets["text"].apply(lambda x: x.lower())
tweets.head()


# In[11]:


tweets["CleanText"] = tweets["CleanText"].apply(lambda x: re.sub(r"https?://\S+|www\.\S+", "",x))


# In[12]:


def removeHTML(text):
    html = re.compile(r"<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});")
    return re.sub(html, "", text)

tweets["CleanText"] = tweets["CleanText"].apply(lambda x: removeHTML(x))


# In[13]:


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


# In[14]:


def RemovePunctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))


# In[15]:


tweets["CleanText"] = tweets["CleanText"].apply(lambda x: RemovePunctuation(x))


# In[16]:


def RemoveASCII(text):
  return re.sub(r'[^\x00-\x7f]', "", text)

tweets["CleanText"] = tweets["CleanText"].apply(lambda x: RemoveASCII(x))


# In[17]:


# def RemoveSpecial(text):
#     try:
#         return text.remove('#')
#     except:
#         return text


# In[18]:


# tweets["CleanText"] = tweets["CleanText"].apply(lambda x: RemoveSpecial(x))


# In[19]:


tweets.head(30)


# #### There are 61 missing values in Keyword feature

# In[20]:


features = ['keyword']
for feat in features : 
    print("The number of missing values in "+ str(feat)+" are "+str(tweets[feat].isnull().sum()))


# ### Analysis

# In[21]:


plot_size = plt.rcParams["figure.figsize"]
print(plot_size[0])
print(plot_size[1])

plot_size[0]=8
plot_size[1]=6
plt.rcParams["figure.figsize"]=plot_size


# ##### target = 1 is when disaster occurs and target = 0 is when disaster does not occur

# In[22]:


x=tweets.target.value_counts()
sns.barplot(x.index,x)
plt.gca().set_ylabel('samples')


# In[23]:


def _corpus(target):
    corpus=[] # document
    
    for x in tweets[tweets['target']==target]['text'].str.split():
        for i in x:
            corpus.append(i)
    return corpus


# In[24]:


from collections import defaultdict
plt.figure(figsize=(10,5))
corpus=_corpus(0)
import string
d=defaultdict(int)
s = string.punctuation
for i in (corpus):
    if i in s:
        d[i]+=1
        
x,y=zip(*d.items())
plt.bar(x,y,color='blue')


# In[25]:


# Observing the stop words like a, the, in, is etc
from nltk.corpus import stopwords
stop=set(stopwords.words('english'))


# In[26]:


from collections import  Counter
c=Counter(corpus)
m=c.most_common()
x=[]
y=[]
for w,c in m[:]:
    if (w not in stop):
        x.append(w)
        y.append(c)

#sns.barplot(x=y,y=x)


# Tokenize CleanText

# In[32]:


tweets['TokenizedText'] = tweets['CleanText'].apply(nltk.word_tokenize)

display(tweets.head())


# Remove stopwords

# In[33]:


tweets['RemoveStopWords'] = tweets['TokenizedText'].apply(lambda x: [word for word in x if word not in stop])
tweets.head()


# Stemming

# In[34]:


def doPorterStemmer(text):
    stemmer = nltk.PorterStemmer()
    stems = [stemmer.stem(i) for i in text]
    return stems

tweets['PorterStemmer'] = tweets['RemoveStopWords'].apply(lambda x: doPorterStemmer(x))
tweets.head()


# Lemmatization

# In[36]:


from nltk.stem import WordNetLemmatizer

def doLemmatizeWord(text):
    lemmatizer = WordNetLemmatizer()
    lemma = [lemmatizer.lemmatize(word) for word in text]
    return lemma

tweets['LemmatizedText'] = tweets['RemoveStopWords'].apply(lambda x: doLemmatizeWord(x))
tweets.head()


# In[38]:


# Using WordNetLemmatizer for FINAL text
tweets['FINAL']=tweets['LemmatizedText'].apply(lambda x: ''.join(i+' ' for i in x))
tweets.head()


# Converting Text into Numeric format using different approaches

# In[39]:


# 1. CounterVector Numerical Dataset
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
cv_df = vectorizer.fit_transform(tweets['FINAL'])

vectorizer.get_feature_names_out()


# In[40]:


# 2. Numerical Dataset using TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer

tfvectorizer = TfidfVectorizer(ngram_range = (1,2))
tfidf_df = tfvectorizer.fit_transform(tweets['FINAL'])


# In[49]:


# 3. Numerical Dataset using Word2Vec
from gensim.models import Word2Vec, KeyedVectors
import multiprocessing

sentences = tweets['FINAL'].values
sentences = [nltk.word_tokenize(sent) for sent in sentences]

w2v_size = 300

cores = multiprocessing.cpu_count() # Count the number of cores in a computer

w2v_model = Word2Vec(
  min_count=1,
  window=2,
  vector_size=w2v_size,
  sample=6e-5, 
  alpha=0.03, 
  min_alpha=0.0007, 
  negative=20,
  workers=cores-1
  )

w2v_model.build_vocab(sentences, progress_per=10000)
w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)

words = list(w2v_model.wv.key_to_index)
w2v_df = []
for sent in sentences:
    vw = np.zeros(w2v_size) 
    #one sentence has many words
    for word in sent:
        vw +=  w2v_model.wv[word]
    #average
    vw = vw/len(sent)
    w2v_df.append(np.array(vw))


# In[51]:


# 4. Reduce Dimension with Principal Component Analysis
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

trans_w2v_df = StandardScaler().fit_transform(w2v_df)

#reduce dimention half
pca = PCA(n_components=100)
pca_trans_w2v_df = pca.fit_transform(trans_w2v_df)


# Data Split

# In[53]:


RANDOM_STATE = 2022
TEST_SIZE = 0.3

cv_train_X, cv_test_X, cv_train_Y, cv_test_y = train_test_split(cv_df, tweets['target'], test_size=TEST_SIZE, random_state=RANDOM_STATE)
tfidf_train_X, tfidf_test_X, tfidf_train_Y, tfidf_test_y = train_test_split(tfidf_df, tweets['target'], test_size=TEST_SIZE, random_state=RANDOM_STATE)
w2v_train_X, w2v_test_X, w2v_train_Y, w2v_test_y = train_test_split(w2v_df, tweets['target'], test_size=TEST_SIZE, random_state=RANDOM_STATE)
pca_w2v_train_X, pca_w2v_test_X, pca_w2v_train_Y, pca_w2v_test_y = train_test_split(pca_trans_w2v_df, tweets['target'], test_size=TEST_SIZE, random_state=RANDOM_STATE)


# Random Forest

# In[60]:


from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def model_RandomForest(train_X, test_X, train_y, test_y):
  rf = RandomForestClassifier()
  rf_model = rf.fit(train_X, train_y.values.ravel())
  pred_y = rf_model.predict(test_X)

  #Accuracy
  print('accuracy_score: %.3f' % accuracy_score(test_y, pred_y))
  print('Recall: %.3f' % recall_score(test_y, pred_y))
  print('Precision: %.3f' % precision_score(test_y, pred_y))
  print('F1 Score: %.3f' % f1_score(test_y, pred_y))


# In[62]:


print ('---------------------- RANDOM FOREST | DATASET: COUNTER-VECTOR ----------------------')
model_RandomForest(cv_train_X, cv_test_X, cv_train_Y, cv_test_y)
print ('---------------------- RANDOM FOREST | DATASET: TF-IDF ----------------------')
model_RandomForest(tfidf_train_X, tfidf_test_X, tfidf_train_Y, tfidf_test_y)
print ('---------------------- RANDOM FOREST | DATASET: WORD2VEC ----------------------')
model_RandomForest(w2v_train_X, w2v_test_X, w2v_train_Y, w2v_test_y)
print ('---------------------- RANDOM FOREST | DATASET: WORD2VEC with PCA ----------------------')
model_RandomForest(pca_w2v_train_X, pca_w2v_test_X, pca_w2v_train_Y, pca_w2v_test_y)


# ##### Random Forest with Counter Vector has the best accuracy: 0.788

# In[66]:


from sklearn.metrics import confusion_matrix

rf = RandomForestClassifier()
rf_model = rf.fit(cv_train_X, cv_train_Y.values.ravel())
pred_y = rf_model.predict(cv_test_X)

conf_matrix = confusion_matrix(y_true=cv_test_y, y_pred=pred_y)

fig, ax = plt.subplots(figsize=(5, 5))
ax.matshow(conf_matrix, cmap=plt.cm.Oranges, alpha=0.3)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')

plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()


# ##### XG-BOOST

# In[63]:


import xgboost as xgb
from sklearn.model_selection import GridSearchCV

def model_XGBoost(train_X, test_X, train_y, test_y):
  xgb_cl = xgb.XGBClassifier()
  xg_model = xgb_cl.fit(train_X, train_y.values.ravel())
  pred_y = xg_model.predict(test_X)

  #Accuracy
  print('accuracy_score: %.3f' % accuracy_score(test_y, pred_y))
  print('Recall: %.3f' % recall_score(test_y, pred_y))
  print('Precision: %.3f' % precision_score(test_y, pred_y))
  print('F1 Score: %.3f' % f1_score(test_y, pred_y))


# In[64]:


print ('---------------------- XGBOOST | DATASET: COUNTER-VECTOR ----------------------')
model_XGBoost(cv_train_X, cv_test_X, cv_train_Y, cv_test_y)
print ('---------------------- XGBOOST | DATASET: TF-IDF ----------------------')
model_XGBoost(tfidf_train_X, tfidf_test_X, tfidf_train_Y, tfidf_test_y)
print ('---------------------- XGBOOST | DATASET: WORD2VEC ----------------------')
model_XGBoost(w2v_train_X, w2v_test_X, w2v_train_Y, w2v_test_y)
print ('---------------------- XGBOOST | DATASET: WORD2VEC with PCA ----------------------')
model_XGBoost(pca_w2v_train_X, pca_w2v_test_X, pca_w2v_train_Y, pca_w2v_test_y)


# ##### XGBoost with Counter Vector has the best accuracy: 0.773

# In[67]:


xgb_cl = xgb.XGBClassifier()
xg_model = xgb_cl.fit(cv_train_X, cv_train_Y.values.ravel())
pred_y = xg_model.predict(cv_test_X)

conf_matrix = confusion_matrix(y_true=cv_test_y, y_pred=pred_y)

fig, ax = plt.subplots(figsize=(5, 5))
ax.matshow(conf_matrix, cmap=plt.cm.Oranges, alpha=0.3)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')

plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()

