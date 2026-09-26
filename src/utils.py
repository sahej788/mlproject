import os 
import sys 
import numpy as np 
import pandas as pd 
import dill 
import pickle
from src.exception import CustomException

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
        