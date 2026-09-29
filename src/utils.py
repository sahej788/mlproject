import os 
import sys 
import numpy as np 
import pandas as pd 
import dill 
import pickle
from src.exception import CustomException
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

def save_object(file_path,obj): #file_path:Where should the object be saved for my project it is should be 
    #artifacts/proprocessor.pkl,obj:What should be saved for my case preprocessing_obj
    try:
        dir_path=os.path.dirname(file_path) #Get the folder
        os.makedirs(dir_path,exist_ok=True) #Create the folder
        with open(file_path,"wb") as file_obj:#This opens:artifacts/proprocessor.pkl,The "wb" means:w → write,b → binary,
            #So you're opening the file for writing binary data.Why binary? Because you're saving a Python object using pickle.
            
            dill.dump(obj,file_obj) #This is the actual saving operation 
            #The .pkl file contains the serialized Python object.
            #Later, you can load it back and use the same preprocessing object.
    except Exception as e:
        raise CustomException(e,sys)

def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report={} #Store the test R² score for every model
        for i in range(len(list(models))):
            model=list(models.values())[i]

            #Get the parameters for that model
            para=param[list(models.keys())[i]]

            #GridSearchCV
            gs=GridSearchCV(model,para,cv=3) #This creates a Grid Search object.Try different combinations of hyperparameters and determine which combination performs best.
            
            gs.fit(X_train,y_train) #GridSearchCV knows which parameters performed best.

            model.set_params(**gs.best_params_)#Configure this model using the hyperparameters that GridSearchCV found to be best,The ** unpacks the dictionary into keyword arguments.
            
            model.fit(X_train,y_train) #Now the model is trained using the best hyperparameters found by GridSearchCV.
            
            y_train_pred=model.predict(X_train) #The trained model predicts the target for training data.
            y_test_pred=model.predict(X_test) #Now the model predicts the unseen test data.

            #Calculate training R²
            train_model_score = r2_score(y_train,y_train_pred)

            #Calculate test R²
            test_model_score = r2_score(y_test,y_test_pred)

            #Store the result
            report[list(models.keys())[i]] = test_model_score
        return report

    except Exception as e:
        raise CustomException(e,sys)

def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys)


        