from flask import Flask, render_template,request,redirect

import pandas as pd
import pickle
import numpy as np


app=Flask(__name__,template_folder="templets")

model = pickle.load(open("LinearRegressionModellll.pkl",'rb'))
bike=pd.read_csv("bikessss.csv")

@app.route('/')
def index():
    ModelName = sorted(bike['Model_Name'].unique())
    Mileage = sorted(str(bike['Mileage'].unique()))
    CC = sorted(bike['CC'].unique(), reverse=True)
    Weight = bike['Weight'].unique()
    ModelName.insert(0,"select Model")
    return render_template('index.html',Mileage=Mileage, ModelName=ModelName, CC=CC, Weight=Weight)

@app.route('/predict', methods=['POST'])
def predict():
    ModelName = request.form.get('Model_Name')
    Mileage = int(request.form.get('Mileage'))
    CC = request.form.get('CC')
    Weight = request.form.get('Weight')


    print(ModelName, Mileage, CC, Weight)

    prediction = model.predict(pd.DataFrame([[ModelName,CC,Mileage,Weight]], columns=['Model_Name', 'CC', 'Mileage', 'Weight']))



    return str(np.round(prediction[0],1))


if __name__=="__main__":
    app.run(debug=True)