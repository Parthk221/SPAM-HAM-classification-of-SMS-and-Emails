#!/usr/bin/env python
# coding: utf-8

# # SPAM-HAM classification of SMS and Emails<br>
# <br>
# Importing basic libraries to access and manage our Dataset<br>
# Pandas used to import our data from tsv file into a dataframe

import numpy as np
import pandas as pd


# ## Importing Our Dataset

from sklearn.model_selection import train_test_split
dataset = pd.read_csv('smsspamcollection.tsv', sep='\t')


# ## Our dataset contains Label, Message , Length of String & Punctuations

dataset.head()


# ## Dividing our data into Feature data and Label data
# - We take 'length' and 'punct' as our feature data(X).
# - Our label data(y) will be 'label'

X = dataset[['length','punct']]
y = dataset['label']


# ## Import train_test_split from sklearn.model_selection
# - Dividing our data into training and test set.
# - Taking 30% of our data into test set.


from sklearn.model_selection import train_test_split 



X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state = 42)


# ## Training model using Logistic Regression 
# - Importing LogisticRegression from sklearn.linear_model
# - Fitting our training set to the model
# - Using Lbfgs solver to handle multinomial problem
# - Multinomial logistic regression is a classification method that generalizes logistic regression to multiclass problems, i.e. with two or more than two possible discrete outcomes.


from sklearn.linear_model import LogisticRegression



model = LogisticRegression(solver = "lbfgs")



model.fit(X_train,y_train)


# ## Using our test data set to predict outcomes from our model trained on training dataset
# - Using our X_test variable to check predict the values of 'label'.


y_pred = model.predict(X_test)


# - Creating a confusion metrics to calculate how our model performed in classifying the test data.
# - importing metrics from sklearn


from sklearn import metrics



pd.DataFrame(metrics.confusion_matrix(y_test,y_pred), index = ['ham','spam'], columns=['ham','spam'])


# ### Now we'll calculate the precision, recall, f1-score , support and accuracy of our model


print(metrics.classification_report(y_test,y_pred))



print(f"Acuracy Score : {metrics.accuracy_score(y_test,y_pred).round(4) * 100}%" )


# ## Now using Navie Bayes Multinomial Model to see how it performs


from sklearn.naive_bayes import MultinomialNB
model_nb = MultinomialNB()
model_nb.fit(X_train,y_train)



y_pred = model_nb.predict(X_test)
pd.DataFrame(metrics.confusion_matrix(y_test,y_pred), index = ['ham','spam'], columns=['ham','spam'])



print(metrics.classification_report(y_test,y_pred))



print(f"Acuracy Score : {metrics.accuracy_score(y_test,y_pred).round(4) * 100}%" )


# ## Now using SVM to see how it performs


from sklearn.svm import SVC



model_svc = SVC(gamma='auto')
model_svc.fit(X_train,y_train)



y_pred = model.predict(X_test)
pd.DataFrame(metrics.confusion_matrix(y_test,y_pred), index = ['ham','spam'], columns=['ham','spam'])



print(metrics.classification_report(y_test,y_pred))



print(f"Acuracy Score : {metrics.accuracy_score(y_test,y_pred).round(4) * 100}%" )


# # Problem with this Classification
# 
# - Applying these models dosen't give us an accurate prediction
# - It classifies the data on the basis of length and punctations in the text which can give us false positives because an message can be a HAM and still be formal as seen in the models above
# 
# ## Solution
# - In order to solve the problem we use text classification to extract features from the text in order to classify it better.

# # Using Term Frequency Inverse Document Frequency (TD-IDF)
# 
# - Basically what it does covert our text to matrix form in order to extract features from it and input that matrix into out machine learning model
# - This functions count the frequnecy of each word in the text and inverses it : The reseaon to inverse the term freqeuncy is to diminish the weights of the words that occur too freqeuntly such as 'a','the' in the document and increase the weight of the terms that occur rarely.
# - TD-IDF helps us understand the significance of a word in an entire corpus rather than relative to a document.
# 
# ## Creating new X,y for Text Classification from our previous dataset


X = dataset['message']
y = dataset ['label']


# ## Splitting our data into Test and Training sets


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state = 42)


# ## We'll create a pipeline
# - This will covert our entire text to TF-IDF in Vectorised format
# - Then it will perform Linear SVC on our trainig set(X_train)


from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC



text_clf = Pipeline([('text_idf',TfidfVectorizer()), ('clf',LinearSVC())])



text_clf.fit(X_train,y_train)



y_pred = text_clf.predict(X_test)



print(metrics.classification_report(y_test,y_pred))



print(f"Acuracy Score : {metrics.accuracy_score(y_test,y_pred).round(4) * 100}%" )


# ## Using this model helped us achive a much better accuracy using the power of text extraction with the help of TF-IDF
# 
# ## Contributing
# Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
# 
# Please make sure to update tests as appropriate.



