##The main job of this code is:
#Read train.csv and test.csv
#Separate input features from the target (math_score)
#Handle missing values
#Scale numerical features
#Convert categorical features into numbers
#Combine processed features with the target
#Save the preprocessing object for later use
##fit-The preprocessing object learns information from the training data.
#For example:
#median values for missing numerical data
#most frequent categorical values
#category mappings
#means/std deviations for scaling
##transform-It then applies those learned rules to the training data.

import sys
from dataclasses import dataclass #This allows you to create a configuration class easily.

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer #ColumnTransformer says:Apply this preprocessing to numerical columns, and a different preprocessing to categorical columns
from sklearn.impute import SimpleImputer #SimpleImputer handles missing values
from sklearn.pipeline import Pipeline #A Pipeline allows you to put multiple preprocessing steps together.Instead of manually doing each operation separately, Pipeline packages them together
from sklearn.preprocessing import OneHotEncoder,StandardScaler #OneHotEncoder converts categorical text into numerical columns.ML algorithms generally need numerical input rather than raw strings
#StandardScaler puts numerical features on a comparable scale
from src.exception import CustomException
from src.logger import logging
import os#Used for creating the path where the preprocessing object will be saved
from src.utils import save_object

@dataclass 
class DataTransformationConfig:#This is your configuration class.
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')#This defines where the preprocessing object will be saved

class DataTransformation: #This class contains all the preprocessing logic
    def __init__(self): #Constructor
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformation_object(self): #This method's job is to build the preprocessing object.It doesn't actually transform the dataset yet.It creates the instructions for how the data should be transformed.
        '''
        This function is responsible for data trnasformation
        '''
        try:
            numerical_columns=["writing_score","reading_score"]#These are my numerical input features
            categorical_columns = ["gender","race_ethnicity","parental_level_of_education","lunch","test_preparation_course"]
            #These columns contain categories/text.These need to be converted into numerical representations

            num_pipeline=Pipeline( #This creates a pipeline specifically for numerical columns
                steps=[
                    ("imputer",SimpleImputer(strategy="median")), #If a numerical value is missing:the median value is used to fill it.Why median?Median is relatively resistant to extreme values/outliers compared with the mean.
                    ('scaler',StandardScaler())#After missing values have been handled, the numerical features are standardized.
                ]
            )
            cat_pipeline=Pipeline(#This pipeline handles categorical columns
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ('one hot encoder',OneHotEncoder()),#This converts categories into numerical columns
                    ('scaler',StandardScaler(with_mean=False)) #The encoded categorical data is scaled
                    #Why with_mean=False?:One-hot encoding can produce a sparse matrixA,sparse matrix stores mostly-zero data
                    #efficiently,If you subtract the mean from every value, you can turn that sparse matrix into a dense.matrix,
                    #which can consume much more memory,allows the sparse representation to remain efficient
                ]
            )
            #These messages are written into your log file.
            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            #Creating ColumnTransformer
            #This is where everything comes together
            #ColumnTransformer combines both pipelines into one preprocessing object
            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_columns),
                    ('cat_pipeline',cat_pipeline,categorical_columns)
                ]
            )
            return preprocessor#This returns the object you just created
            #Remember:At this stage you have not transformed the data yet.You've created the instructions for transformation
        
        except Exception as e: #Exception handling
            raise CustomException(e,sys) #If anything goes wrong while creating the preprocessing object, your custom exception
            #handles it and gives you useful traceback information

    #Main transformation method
    def initiate_data_transformation(self,train_path,test_path): #This is the method that actually performs the transformation.
        try:
            train_df=pd.read_csv(train_path)#Read training data
            test_df=pd.read_csv(test_path)#Read testing data
            logging.info("Read train and test data completed")
            
            logging.info("Obtaining preprocessing object")
            preprocessing_obj=self.get_data_transformation_object() #This calls the method we just discussed.

            target_column_name="math_score"
            numerical_columns = ["writing_score", "reading_score"]

            #Separate X and y for training

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1) #This becomes your X_train equivalent.
            target_feature_train_df=train_df[target_column_name]#This becomes your y_train equivalent.

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info("Applying preprocessing object on training dataframe and testing dataframe.")

            #The most important transformation step

            #Why fit_transform on training data?:because the preprocessing parameters must be learned only from training data.This
            #prevents information from the test set from leaking into the training process.This is called avoiding data leakage.
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df) #Transform test data

            #Combine X and y
            train_arr=np.c_[input_feature_train_arr,np.array(target_feature_train_df)]#np.c_ combines arrays column-wise.
            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            logging.info("Saved preprocessing object.")

            #Save the preprocessing object
            save_object( #Take a Python object and save it to a file so that you can use it later.
                #This is extremely important for deployment.
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj #You're saving:preprocessing_obj to artifacts/proprocessor.pkl
                #Why?Imagine later your model is deployed and a new student comes in so You need to apply the exact same 
                #preprocessing that was used during training.You don't want to create a completely new encoder/scaler.So you save
                #the fitted preprocessing object and can load it later.
            )
            #Return three things
            return (train_arr,test_arr,self.data_transformation_config.preprocessor_obj_file_path,)

            #Final exception handling



        except Exception as e:
            raise CustomException(e,sys)


