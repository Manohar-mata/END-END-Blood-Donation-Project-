from flask import Flask,request,render_template
import numpy as np
import pandas as pd
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

from sklearn.preprocessing import StandardScaler

application=Flask(__name__)
app=application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            Recency_months=int(request.form.get('recency_months')),
            Frequency_times=int(request.form.get('frequency_times')),
            Monetary_cc_blood=int(request.form.get('monetary_cc_blood')),
            Time_months=int(request.form.get('time_months'))
        )

        
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        pred_pipeline=PredictPipeline()
        print("Mid Prediction")
        results=pred_pipeline.predict(pred_df)
        print("after Prediction")
        return render_template('home.html',results=results[0])
    
if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)  
    