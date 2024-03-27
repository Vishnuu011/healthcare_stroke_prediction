import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from src.mlproject.utils import save_object
from src.mlproject.exception import CustomException
from src.mlproject.logger import logging



@dataclass
class DataTransformationCofig:
    preprocessor_obj_file_path:str =os.path.join("artifacts","preprocessor.pkl")


class DataTransformation:

    def __init__(self):

        self.data_tansformation_config=DataTransformationCofig()

    def get_data_transformation(self):


        try:
            categorical_col=['gender', 'ever_married', 'work_type', 'Residence_type','smoking_status']
            numerical_col=['age', 'hypertension', 'heart_disease', 'avg_glucose_level','bmi']

            logging.info("Data ingestion started................")

        #numerical pipeline
            num_pipeline=Pipeline(
               steps=[
                   ('imputer', SimpleImputer(strategy='median')),
                   ('scaler', StandardScaler())
                ]
            )
        #categorical Pipeline
            cat_pipeline=Pipeline(
                 steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('ordinalencoder', OrdinalEncoder()),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )
            preprocessor=ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_col),
                ('cat_pipeline', cat_pipeline, categorical_col)
            ])
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)    

    def initialize_data_transformation(self,train_path,test_path):
        
        try:

            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("read train and test data complete")
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head : \n{test_df.head().to_string()}')

            preprocessing_obj = self.get_data_transformation()

            target_column_name = 'stroke'
            drop_columns = [target_column_name,'id']

            input_feature_train_df = train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df=train_df[target_column_name]


            input_feature_test_df=test_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df=test_df[target_column_name]

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)

            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            logging.info("Applying preprocessing object on training and testing datasets.")

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

        

            save_object(
               file_path=self.data_tansformation_config.preprocessor_obj_file_path,
               obj=preprocessing_obj
            )

            logging.info("preprocessing pickle file saved")

            return (
               train_arr,
               test_arr
            )
        except Exception as e:
            raise CustomException(e, sys)    