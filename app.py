from flask import Flask, render_template, request, redirect
import numpy as np 
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])

def index():
    if request.method == 'POST':

        # Dari 31 data yang akan digunakan: 
        age = request.form['age']
        sex = request.form['sex']
        graduate = request.form['graduate']
        scholarship = request.form['scholarship']
        artsport = request.form['artsport']
        hours = request.form['hours']
        attend = request.form['attend']
        gpa = request.form['gpa']

        # ML #
        dataset = pd.read_csv("csv/DATA.csv", delimiter=";")
        X = dataset[['1', '2', '3', '4', '6', '17', '22','29']].values
        y =  dataset['GRADE'].values       

        from sklearn.model_selection import train_test_split
        X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.2, random_state=3)

        # model Devition Tree
        drugTree = DecisionTreeClassifier(criterion="gini", max_depth = 10)
        drugTree.fit(X_trainset, y_trainset)

        # Prediksi
        x_new = np.array((age, sex, graduate, scholarship, artsport, hours, attend, gpa))
        x_new = np.reshape(x_new, (1, -1))

        predTree = drugTree.predict(x_new)

        output = predTree[0]

        return render_template('index.html', output = output)
    else:
        return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)