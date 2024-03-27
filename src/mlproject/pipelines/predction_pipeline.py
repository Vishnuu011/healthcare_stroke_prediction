import os
import sys
import pandas as pd
from src.mlproject.logger import logging
from src.mlproject.exception import CustomException
from src.mlproject.utils import load_object

class PredictPipeline:

    
    def __init__(self):
        print("init.. the object")

    def predict(self,features):
        try:
            preprocessor_path=os.path.join("artifacts","preprocessor.pkl")
            model_path=os.path.join("artifacts","model.pkl")

            preprocessor=load_object(preprocessor_path)
            model=load_object(model_path)

            scaled_fea=preprocessor.transform(features)
            pred=model.predict(scaled_fea)

            return pred

        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(self,
                 gender:str,
                 age:float,
                 hypertension:float,
                 heart_disease:float,
                 ever_married:str,
                 work_type:str,
                 Residence_type:str,
                 avg_glucose_level:float,
                 bmi:float,
                 smoking_status:str):

        self.gender=gender
        self.age=age
        self.hypertension=hypertension
        self.heart_disease=heart_disease
        self.ever_married=ever_married
        self.work_type=work_type
        self.Residence_type = Residence_type
        self.avg_glucose_level = avg_glucose_level
        self.bmi = bmi
        self.smoking_status = smoking_status

    def get_data_as_dataframe(self):
        
        try:
            custom_data_input_dict = {
                'gender':[self.gender],
                'age':[self.age],
                'hypertension':[self.hypertension],
                'heart_disease':[self.heart_disease],
                'ever_married':[self.ever_married],
                'work_type':[self.work_type],
                'Residence_type':[self.Residence_type],
                'avg_glucose_level':[self.avg_glucose_level],
                'bmi':[self.bmi],
                'smoking_status':[self.smoking_status]
            }
            df = pd.DataFrame(custom_data_input_dict)

            return df 
        except Exception as e:
            logging.info('Exception Occured in prediction pipeline')
            raise CustomException(e,sys)               