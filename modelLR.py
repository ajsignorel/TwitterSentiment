
#Utility Libaries
import pickle
import numpy as np
import pandas as pd

#Plotting Libraries
import seaborn as sns
import matplotlib.pyplot as plt


#Sklearn Libraries
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report


#Loading Dataset for training, CSV File
encodingData = "ISO-8859-1"
data = pd.read_csv('C:/Datasets/TwitterData/processedData.csv', encoding=encodingData , low_memory=False)

#Graph the distribution, used for checking the dataset for distribution
#ax = data.groupby('sentiment').count().plot(kind='bar', title='Distribution of data', legend=False)
#ax.set_xticklabels(['Negative','Positive'], rotation=0)

#Print the DataFrame
print(data)


#text, sentiment = list(data['text']), list(data['sentiment'])

#Splitting Data into Training and Testing Data
X_train, X_test, y_train, y_test = train_test_split(text, sentiment, test_size = 0.05, random_state = 0)
print(f'Data Split Complete')


                      #Model
#Vector                
vector = TfidfVectorizer(ngram_range=(1,2), max_features=500000)
vector.fit(X_train)
print(f'Fitted')
print('Number of feature words: ', len(vector.get_feature_names()))

#Data Transformation
X_train = vector.transform(X_train)
X_test = vector.transform(X_test)
print(f'Data Transformed')

#Logistic Regression
LRmodel = LogisticRegression(C = 2, max_iter = 1000, n_jobs = -1)
LRmodel.fit(X_train, y_train)

