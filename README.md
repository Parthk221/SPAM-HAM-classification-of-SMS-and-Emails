
# SPAM-HAM classification of SMS and Emails<br>
## This program is created to train our machine learning model to detect wether a message is SPAM or not. To do so we take help of <br>
Importing basic libraries to access and manage our Dataset<br>
Pandas used to import our data from tsv file into a dataframe


```python
import numpy as np
import pandas as pd
```

## Importing Our Dataset


```python
from sklearn.model_selection import train_test_split
dataset = pd.read_csv('smsspamcollection.tsv', sep='\t')
```

## Our dataset contains Label, Message , Length of String & Punctuations


```python
dataset.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>label</th>
      <th>message</th>
      <th>length</th>
      <th>punct</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ham</td>
      <td>Go until jurong point, crazy.. Available only ...</td>
      <td>111</td>
      <td>9</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ham</td>
      <td>Ok lar... Joking wif u oni...</td>
      <td>29</td>
      <td>6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>spam</td>
      <td>Free entry in 2 a wkly comp to win FA Cup fina...</td>
      <td>155</td>
      <td>6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ham</td>
      <td>U dun say so early hor... U c already then say...</td>
      <td>49</td>
      <td>6</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ham</td>
      <td>Nah I don't think he goes to usf, he lives aro...</td>
      <td>61</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



## Dividing our data into Feature data and Label data
- We take 'length' and 'punct' as our feature data(X).
- Our label data(y) will be 'label'


```python
X = dataset[['length','punct']]
y = dataset['label']
```

## Import train_test_split from sklearn.model_selection
- Dividing our data into training and test set.
- Taking 30% of our data into test set.


```python
from sklearn.model_selection import train_test_split 
```


```python
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state = 42)
```

## Training model using Logistic Regression 
- Importing LogisticRegression from sklearn.linear_model
- Fitting our training set to the model
- Using Lbfgs solver to handle multinomial problem
- Multinomial logistic regression is a classification method that generalizes logistic regression to multiclass problems, i.e. with two or more than two possible discrete outcomes.


```python
from sklearn.linear_model import LogisticRegression
```


```python
model = LogisticRegression(solver = "lbfgs")
```


```python
model.fit(X_train,y_train)
```




    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='warn',
              n_jobs=None, penalty='l2', random_state=None, solver='lbfgs',
              tol=0.0001, verbose=0, warm_start=False)



## Using our test data set to predict outcomes from our model trained on training dataset
- Using our X_test variable to check predict the values of 'label'.


```python
y_pred = model.predict(X_test)
```

- Creating a confusion metrics to calculate how our model performed in classifying the test data.
- importing metrics from sklearn


```python
from sklearn import metrics
```


```python
pd.DataFrame(metrics.confusion_matrix(y_test,y_pred), index = ['ham','spam'], columns=['ham','spam'])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ham</th>
      <th>spam</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ham</th>
      <td>1404</td>
      <td>44</td>
    </tr>
    <tr>
      <th>spam</th>
      <td>219</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>



### Now we'll calculate the precision, recall, f1-score , support and accuracy of our model


```python
print(metrics.classification_report(y_test,y_pred))
```

                  precision    recall  f1-score   support
    
             ham       0.87      0.97      0.91      1448
            spam       0.10      0.02      0.04       224
    
       micro avg       0.84      0.84      0.84      1672
       macro avg       0.48      0.50      0.48      1672
    weighted avg       0.76      0.84      0.80      1672
    



```python
print(f"Acuracy Score : {metrics.accuracy_score(y_test,y_pred).round(4) * 100}%" )
```

    Acuracy Score : 84.27%


## Now using Navie Bayes Multinomial Model to see how it performs


```python
from sklearn.naive_bayes import MultinomialNB
model_nb = MultinomialNB()
model_nb.fit(X_train,y_train)
```




    MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)




```python
y_pred = model_nb.predict(X_test)
pd.DataFrame(metrics.confusion_matrix(y_test,y_pred), index = ['ham','spam'], columns=['ham','spam'])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ham</th>
      <th>spam</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ham</th>
      <td>1438</td>
      <td>10</td>
    </tr>
    <tr>
      <th>spam</th>
      <td>224</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(metrics.classification_report(y_test,y_pred))
```

                  precision    recall  f1-score   support
    
             ham       0.87      0.99      0.92      1448
            spam       0.00      0.00      0.00       224
    
       micro avg       0.86      0.86      0.86      1672
       macro avg       0.43      0.50      0.46      1672
    weighted avg       0.75      0.86      0.80      1672
    



```python
print(f"Acuracy Score : {metrics.accuracy_score(y_test,y_pred).round(4) * 100}%" )
```

    Acuracy Score : 86.0%


## Now using SVM to see how it performs


```python
from sklearn.svm import SVC
```


```python
model_svc = SVC(gamma='auto')
model_svc.fit(X_train,y_train)
```




    SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
      max_iter=-1, probability=False, random_state=None, shrinking=True,
      tol=0.001, verbose=False)




```python
y_pred = model.predict(X_test)
pd.DataFrame(metrics.confusion_matrix(y_test,y_pred), index = ['ham','spam'], columns=['ham','spam'])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ham</th>
      <th>spam</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ham</th>
      <td>1404</td>
      <td>44</td>
    </tr>
    <tr>
      <th>spam</th>
      <td>219</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(metrics.classification_report(y_test,y_pred))
```

                  precision    recall  f1-score   support
    
             ham       0.87      0.97      0.91      1448
            spam       0.10      0.02      0.04       224
    
       micro avg       0.84      0.84      0.84      1672
       macro avg       0.48      0.50      0.48      1672
    weighted avg       0.76      0.84      0.80      1672
    



```python
print(f"Acuracy Score : {metrics.accuracy_score(y_test,y_pred).round(4) * 100}%" )
```

    Acuracy Score : 84.27%


# Problem with this Classification

- Applying these models dosen't give us an accurate prediction
- It classifies the data on the basis of length and punctations in the text which can give us false positives because an message can be a HAM and still be formal as seen in the models above

## Solution
- In order to solve the problem we use text classification to extract features from the text in order to classify it better.

# Using Term Frequency Inverse Document Frequency (TD-IDF)

- Basically what it does covert our text to matrix form in order to extract features from it and input that matrix into out machine learning model
- This functions count the frequnecy of each word in the text and inverses it : The reseaon to inverse the term freqeuncy is to diminish the weights of the words that occur too freqeuntly such as 'a','the' in the document and increase the weight of the terms that occur rarely.
- TD-IDF helps us understand the significance of a word in an entire corpus rather than relative to a document.

## Creating new X,y for Text Classification from our previous dataset


```python
X = dataset['message']
y = dataset ['label']
```

## Splitting our data into Test and Training sets


```python
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state = 42)
```

## We'll create a pipeline
- This will covert our entire text to TF-IDF in Vectorised format
- Then it will perform Linear SVC on our trainig set(X_train)


```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
```


```python
text_clf = Pipeline([('text_idf',TfidfVectorizer()), ('clf',LinearSVC())])
```


```python
text_clf.fit(X_train,y_train)
```




    Pipeline(memory=None,
         steps=[('text_idf', TfidfVectorizer(analyzer='word', binary=False, decode_error='strict',
            dtype=<class 'numpy.float64'>, encoding='utf-8', input='content',
            lowercase=True, max_df=1.0, max_features=None, min_df=1,
            ngram_range=(1, 1), norm='l2', preprocessor=None, smooth_idf=Tr...ax_iter=1000,
         multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
         verbose=0))])




```python
y_pred = text_clf.predict(X_test)
```


```python
print(metrics.classification_report(y_test,y_pred))
```

                  precision    recall  f1-score   support
    
             ham       0.99      1.00      1.00      1448
            spam       0.99      0.96      0.97       224
    
       micro avg       0.99      0.99      0.99      1672
       macro avg       0.99      0.98      0.98      1672
    weighted avg       0.99      0.99      0.99      1672
    



```python
print(f"Acuracy Score : {metrics.accuracy_score(y_test,y_pred).round(4) * 100}%" )
```

    Acuracy Score : 99.22%


## Using this model helped us achive a much better accuracy using the power of text extraction with the help of TF-IDF


```python

```
