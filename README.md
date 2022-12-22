# Grade Prediction Test
## Abstract
Education has vital and increasing importance almost for all countries in order to accelerate their development. Well-educated persons provide more benefits to their countries and for that reason, classification of students’ performance before they enter exams or taking courses is also gained an importance. Improvement of education quality must be performed during the active semester to improve students’ personal performance to response this expectation. To provide this, some of the main indicators are students’ personal information, educational preferences and family properties. In this paper, artificial intelligence techniques are applied to the questionnaire results that consists these main indicators, of three different courses of two faculties in order to classify students’ final grade performances and to determine the most efficient machine learning algorithm for this task. Several experiments are performed and results suggests that Radial-Basis Function Neural Network can be used effectively for this and helps to classify student performance with accuracy of 70%–88%.

> The data was collected from the Faculty of Engineering and Faculty of Educational Sciences students in 2019. The purpose is to predict students' end-of-term performances using ML techniques.

## Code Explanatory

``` python
from flask import Flask, render_template, request, redirect
import numpy as np 
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
```
