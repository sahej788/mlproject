import os #os allows Python to interact with the operating system.These help create folders and file paths.

import sys #sys gives access to Python/system information.This allows your custom exception class to access the error's traceback
           #information.

from src.exception import CustomException #You're importing the custom exception class you created earlier

from src.logger import logging #This imports the logging system you created earlier & the information gets stored in your log file.

import pandas as pd

from sklearn.model_selection import train_test_split

from dataclasses import dataclass #makes it easier to create classes that mainly store data/configuration.
                    #Instead of writing a lot of boilerplate code, dataclass automatically creates things like the constructor.

#Creating DataIngestionConfig
@dataclass#The @dataclass tells Python:This class is mainly going to store configuration values.(Decorator function)
class DataIngestionConfig:
    train_data_path:str=os.path.join('artifacts','train.csv') #This defines where your training data will be saved.
    test_data_path:str=os.path.join('artifacts','test.csv') #os.path.join('artifacts', "train.csv") it creates path(artifacts/train.csv),because it handles operating-system path separators.
    raw_data_path:str=os.path.join('artifacts','data.csv')#This is where the original dataset will be copied.

#Creating DataIngestion class
#flow: DataIngestion->read data->save raw data->split train/test->save train/test->return their paths
class DataIngestion: #This class contains the actual logic for bringing the dataset into your ML pipeline.
    def __init__(self): #__init__() runs automatically when you create an object.
        self.ingestion_config=DataIngestionConfig() #This creates an object of DataIngestionConfig and stores it inside the DataIngestion object.

    def initiate_data_ingestion(self):#This method performs the actual data ingestion(Start the data ingestion process)
        logging.info('Entered the data ingestion method or component')#This writes a message to your log file it says Data ingestion has started.This is useful when debugging because later you can look at the log and know where the program reached.
        try:#Everything inside the try block is code that might generate an error.
            df=pd.read_csv('notebook\data\stud.csv')
            logging.info('Read the dataset as Dataframe')#Again, this writes information into your log it says The CSV was successfully read into a DataFrame

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)#Creating the artifacts folder
            #Saving the raw dataset 
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True) #This saves the original DataFrame into:artifacts/data.csv,index=False:Don't save Pandas row numbers as an extra column,header=True:Save the column names

            #Logging train-test split
            logging.info("Train test split initiated")#this records: Train/test splitting has started.

            #Splitting the dataset
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)

            #Saving training data
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)#This saves the training dataset to:artifacts/train.csv
            #Saving test data
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)#This saves the test dataset to:artifacts/test.csv
            logging.info("Ingestion of the data is completed")#This records that ingestion has finished

            #Returning the paths
            return(#The function returns:artifacts/train.csv,artifacts/test.csv
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:#If anything goes wrong inside the try block, catch the error and store it in e
            raise CustomException(e,sys) #passes the original error and sys to your custom exception class.your CustomException then creates a detailed message including:python file,line number,error message

#Main block
if __name__=="__main__":#Run the following code only when this Python file is executed directly
    obj = DataIngestion()#Create DataIngestion object:This creates an object from your class
    obj.initiate_data_ingestion()

        