import os
import sys
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import (AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor,)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts','model.pkl') #When I save the final trained model, save it here.

class ModelTrainer: #This class contains the model-training logic
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):# train_array and test_array comes from data transformation
        try:
            logging.info('Split training and test input data')
            X_train,y_train,X_test,y_test=( 
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models={ # This is basically a collection of models.
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),# verbose=False tells CatBoost:Don't continuously print training information to the console.Otherwise CatBoost can produce a lot of output.
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            #A hyperparameter is a setting you choose before/during model training, rather than something the model learns 
            #directly from the training data.
            params={ #This dictionary contains hyperparameters that you want to try for each model.
                "Decision Tree":{
                    'criterion':['squared_error','friedman_mse','absolute_error','poisson'],#Try different criteria for deciding how the Decision Tree should split data.
                    'splitter':['best','random'],
                    'max_features':['sqrt','log2']
                },
                "Random Forest":{
                    'criterion':['squared_error','friedman_mse','absolute_error','poisson'],
                    'max_features':['sqrt','log2',None],
                    'n_estimators':[8,16,32,64,128,256] #How many decision trees should the Random Forest contain
                },
                "Gradient Boosting":{
                    'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001], #Controls how strongly each new boosting stage contributes.
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],#Controls the fraction of samples used for fitting each boosting stage.
                    'criterion':['squared_error', 'friedman_mse'],
                    'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256] #Number of boosting stages/trees
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10], #How deep the trees can become
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100] #Number of boosting iterations
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                }
             
            model_report: dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,
                                               param=params)
            
            ## To get best model score from dict
            best_model_score=max(sorted(model_report.values()))

            ## To get best model name from dict
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)]

            #Then get the actual model
            best_model=models[best_model_name]

            #Minimum acceptable score
            if best_model_score<0.6: #If none of the models achieves an R² of at least 0.6, stop the pipeline.
                raise CustomException('No best model found',sys)
            logging.info("Best found model on both training and testing dataset")

            #Save the best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            ) 

            #Final Prediction
            predicted = best_model.predict(X_test)

            #Calculate final R² 
            r2_square = r2_score(y_test, predicted)
            return r2_square
        
        except Exception as e:
            raise CustomException(e,sys)