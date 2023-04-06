import os
import sys 
sys.path.append(os.getcwd())
from src.exception import CustomException

from flask import Flask, render_template, request
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application = Flask(__name__)

app = application

#Route for home page

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/predict',methods=['GET','POST']) # type: ignore
def predictt():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data = CustomData(
            month = request.form.get('month'),
            visitortype = request.form.get('visitortype'),
            operatingsystems = int(request.form.get('operatingsystems')),
            browser = int(request.form.get('browser')),
            region = int(request.form.get('region')),
            traffictype = int(request.form.get('traffictype')),
            weekend = int(request.form.get('weekend')),
            administrative = float(request.form.get('administrative')),
            administrative_duration = float(request.form.get('administrative_duration')),
            informational = float(request.form.get('informational')),
            informational_duration = float(request.form.get('informational_duration')),
            productrelated = float(request.form.get('productrelated')),
            productrelated_duration = float(request.form.get('productrelated_duration')),
            bouncerates = float(request.form.get('bouncerates')),
            exitrates = float(request.form.get('exitrates')),
            pagevalues = float(request.form.get('pagevalues')),
            specialday = float(request.form.get('specialday')) 
                )
        pred_df = data.get_data_as_dataframe()
        print(pred_df)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        return render_template('home.html',results=results[0])


#Run
if __name__ == "__main__":
    app.run(debug=True,port=8080)