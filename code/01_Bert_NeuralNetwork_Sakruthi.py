#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from google.colab import drive


# In[ ]:


drive.mount('/content/drive')


# In[ ]:


# loading the dataset 
import pandas as pd
df = pd.read_csv("/content/drive/MyDrive/255dataset/train.csv",encoding='ISO-8859-1')
df.head(5)


# In[ ]:


#dropping 3 columns.
df = df.drop(['id','keyword','location'],axis=1)
df.head()


# In[ ]:


df['target'].value_counts()


# In[ ]:


#Balancing the dataset
df_0_class = df[df['target']==0]
df_1_class = df[df['target']==1]
df_0_class_undersampled = df_0_class.sample(df_1_class.shape[0])
df = pd.concat([df_0_class_undersampled, df_1_class], axis=0)


# In[ ]:


#Splitting the dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df['text'],df['target'], stratify=df['target'])


# In[ ]:


#Installing tensorflow_text
 pip install tensorflow-text


# In[ ]:


#Bert preprocessor and Encoder.
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text


# In[ ]:


preprocess = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4")


# In[ ]:


text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text-layer')
preprocessed_text = preprocess(text_input)
outputs = encoder(preprocessed_text)
d_layer = tf.keras.layers.Dropout(0.1, name="dropout-layer")(outputs['pooled_output'])
d_layer = tf.keras.layers.Dense(1, activation='sigmoid', name="output")(d_layer)
model = tf.keras.Model(inputs=[text_input], outputs = [d_layer])


# In[ ]:


model.summary()


# In[ ]:


m= [
      tf.keras.metrics.BinaryAccuracy(name='accuracy'),
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall')
]
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=m)


# In[ ]:


#Evaluation of Model
model.fit(X_train, y_train, epochs=10)


# In[ ]:


model.evaluate(X_test, y_test)


# In[ ]:


#Here is the classification report.
import numpy as np
y_predicted = model.predict(X_test)
y_predicted = y_predicted.flatten()
y_predicted = np.where(y_predicted > 0.5, 1, 0)
from sklearn.metrics import confusion_matrix, classification_report
matrix = confusion_matrix(y_test, y_predicted)
matrix


# In[ ]:


print(classification_report(y_test, y_predicted))


# In[ ]:


from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


# In[ ]:


#Precision and Accuracy for the Bert 
print('accuracy_score: %.3f' % accuracy_score(y_test, y_predicted))
print('Recall: %.3f' % recall_score(y_test, y_predicted))
print('Precision: %.3f' % precision_score(y_test, y_predicted))
print('F1 Score: %.3f' % f1_score(y_test, y_predicted))


# In[44]:


import sklearn.metrics as metrics

def ROC_Curve(y_train, y_train_predict):
    x,y = np.arange(0,1.1,0.1),np.arange(0,1.1,0.1)
    plt.plot(x, y, '--')

    #### Plot for train
    fpr_train, tpr_train, thresholds = metrics.roc_curve(y_train, y_train_predict)
    roc_auc_train = metrics.auc(fpr_train, tpr_train)
    plt.plot(fpr_train, tpr_train, marker='o', label = 'Train AUC = %0.3f' % roc_auc_train)

    
    plt.legend(loc = 'lower right')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    return(roc_auc_train)


# In[45]:


roc_auc_train_LR = ROC_Curve(y_test,y_predicted)


# In[ ]:




