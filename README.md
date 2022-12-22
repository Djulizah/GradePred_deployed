# Grade Prediction Test
## Abstract
Education has vital and increasing importance almost for all countries in order to accelerate their development. Well-educated persons provide more benefits to their countries and for that reason, classification of students’ performance before they enter exams or taking courses is also gained an importance. Improvement of education quality must be performed during the active semester to improve students’ personal performance to response this expectation. To provide this, some of the main indicators are students’ personal information, educational preferences and family properties. In this paper, artificial intelligence techniques are applied to the questionnaire results that consists these main indicators, of three different courses of two faculties in order to classify students’ final grade performances and to determine the most efficient machine learning algorithm for this task. Several experiments are performed and results suggests that Radial-Basis Function Neural Network can be used effectively for this and helps to classify student performance with accuracy of 70%–88%.

> The data was collected from the Faculty of Engineering and Faculty of Educational Sciences students in 2019. The purpose is to predict students' end-of-term performances using ML techniques.

## Code Explanatory
This is how to make the models and prediction. In this project we'll be using **Flask** as our bridge to connect python source and the web app. In `App.py` file started of by importing all necessary package for our project.

``` python
from flask import Flask, render_template, request, redirect
import numpy as np # math function for python
import pandas as pd # machine learning library
from sklearn.tree import DecisionTreeClassifier 
```

There are a total of 31 data(column), in this **App** we'll only use 7, which will be our input for predicting the output(grade).
``` pyhton
age = request.form['age']
sex = request.form['sex']
graduate = request.form['graduate']
scholarship = request.form['scholarship']
artsport = request.form['artsport']
hours = request.form['hours']
attend = request.form['attend']
gpa = request.form['gpa']
```

This is the machine learning part where we started of by reading the data and deciding the data(columns) to use for prediction. The model for this project are all already turned to numbers and so we'll adapt to it.
``` pyhton
dataset = pd.read_csv("csv/DATA.csv", delimiter=";")
X = dataset[['1', '2', '3', '4', '6', '17', '22','29']].values
y =  dataset['GRADE'].values       

from sklearn.model_selection import train_test_split
X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.2, random_state=3)
```

Model Devition Tree
``` pyhton
drugTree = DecisionTreeClassifier(criterion="gini", max_depth = 10)
drugTree.fit(X_trainset, y_trainset)
```

We've listed the data for prediction input and here we'll take the corresponding data to process it.
``` pyhton
x_new = np.array((age, sex, graduate, scholarship, artsport, hours, attend, gpa))
x_new = np.reshape(x_new, (1, -1))

predTree = drugTree.predict(x_new)

output = predTree[0] # prediction output would be stored here
```

## Related Links
[Dataset from UCI Machine Learning](https://archive.ics.uci.edu/ml/datasets/Higher+Education+Students+Performance+Evaluation+Dataset)
