#flask:Used to create your web application ,ex:application = Flask(__name__) (creates your Flask application)
#request:Used to access information sent by the user's browser,ex:request.form.get('gender')
#gets the value entered/selected for gender
#render_template:Used to display an HTML page,ex:return render_template('index.html') 
#Find index.html inside the templates folder and display it
from flask import Flask,request,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

#Create the Flask application
application=Flask(__name__) #This creates your Flask application.

#Create another variable for the application
app=application #This simply makes another variable pointing to the same Flask application

#First route — home page
@app.route('/') #This is a route.The / represents the root/home URL

#The function associated with that route
def index(): #When somebody visits /, Flask executes this function
    return render_template('index.html') #flask displays index.html

#Prediction route
@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET': #Did the user simply open the page? if yes The form page is displayed.
        return render_template('home.html')
    else:
        data=CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('reading_score')),
            writing_Score=float(request.form.get('writing_score'))
        )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)

        predict_pipeline=PredictPipeline()
        results=predict_pipeline.predict(pred_df)
        return render_template('home.html',results=results[0])

if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)


