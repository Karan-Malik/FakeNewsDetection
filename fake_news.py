# -*- coding: utf-8 -*-
"""
@author: Karan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
ps=PorterStemmer()

dataset=pd.read_csv('train.csv')

dataset.head()

print(dataset.isnull().sum())

#Removing rows having article text as nan
null=[]
for i in range(len(dataset)):
  if str(dataset.values[i,3])=='nan':
    null.append(i)

dataset=dataset.drop(null)

print(dataset.isnull().sum())

#Replacing all nan values with nil
dataset=dataset.fillna('nil')

print(dataset.isnull().sum())

#Comparing Frequency of positive and negative categories in dataset
print(dataset['label'].value_counts())
dataset['label'].value_counts().plot.bar()
plt.xlabel('Category')
plt.ylabel('Frequency')
plt.show()

#Combining title and text to use for classification
dataset['text']=dataset['title']+dataset['text']

dataset.head()

#Text Cleaning and Preprocessing
nltk.download('stopwords')
words=set(stopwords.words('english'))
words=words - set(['no','not','against'])

corpus=[]
for i in range(len(dataset)):
  news=re.sub('[^a-zA-Z]',' ',dataset.values[i,3])
  news=news.lower()
  news=news.split()
  news=[ps.stem(word) for word in news if word not in words]
  news=' '.join(news)
  corpus.append(news)

print(corpus[0])

#Preparing X_train and y_train
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=10000)
X=cv.fit_transform(corpus).toarray()

y=dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1)

#Logistic Regression

from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()
classifier.fit(X_train,y_train)

y_pred=classifier.predict(X_test)

#Confusion Matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_pred,y_test)
cm

#K-Fold Cross validation
from sklearn.model_selection import cross_val_score
acc=cross_val_score(classifier,X_train,y_train,cv=10)
print(np.mean(acc())

      
#Accuracy of 95.8%
