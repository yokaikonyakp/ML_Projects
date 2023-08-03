from flask import Flask, jsonify, render_template, request
from flask_cors import CORS, cross_origin
import joblib
import os
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)
cors=CORS(app)
model=pickle.load(open('LinearRegressionModel.pkl','rb'))
BigMart=pd.read_csv('Cleaned_BigMart.csv')
@app.route("/")
def index():
    return render_template("index.html")

@app.route('/predict',methods=['POST'])
@cross_origin()
def predict():

    item_weight= float(request.form['item_weight'])
    item_fat_content=float(request.form['item_fat_content'])
    item_visibility= float(request.form['item_visibility'])
    item_type= float(request.form['item_type'])
    item_mrp = float(request.form['item_mrp'])
    outlet_establishment_year= float(request.form['outlet_establishment_year'])
    outlet_size= float(request.form['outlet_size'])
    outlet_location_type= float(request.form['outlet_location_type'])
    outlet_type= float(request.form['outlet_type'])

    model_filename = 'LinearRegressionModel.pkl'
    with open(model_filename, 'rb') as model_file:
        model = pickle.load(model_file)

    prediction=model.predict([[ item_weight,item_fat_content,item_visibility,item_type,item_mrp,
                  outlet_establishment_year,outlet_size,outlet_location_type,outlet_type ]]),
    data = np.array([[item_weight, item_fat_content, item_visibility, item_type, item_mrp,
                      outlet_establishment_year, outlet_size, outlet_location_type, outlet_type]]).reshape((1, 9))

    print(prediction)

    return str(np.round(prediction[0], decimals=2))



if __name__ == '__main__':
    app.run(debug=True, port=9457)