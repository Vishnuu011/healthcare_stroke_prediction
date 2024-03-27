import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass

from sklearn.metrics import accuracy_score
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from src.mlproject.exception import CustomException
from src.mlproject.logger import logging
from src.mlproject.utils import evaluate_models, save_object
from sklearn.metrics import accuracy_score

@dataclass
class ModelTrainerConfig:
    trained_model_trainer_config=os.path.join("artifacts", "model.pkl")

class ModelTrainer:

    def __init__(self):

        self.trained_model_config=ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):


        try:
            logging.info("splitting training and testing input data.....")
            X_train, y_train, X_test, y_test=(
               train_array[:,:-1],
               train_array[:,-1],
               test_array[:,:-1],
               test_array[:,-1]
            )
            models={
                "Rrandom forest":RandomForestClassifier(),
                "Log Regression": LogisticRegression(),
                'KNN':KNeighborsClassifier(),
                'Tree':DecisionTreeClassifier(),
                'adaboost':AdaBoostClassifier(),
                'gradboost':GradientBoostingClassifier(),
                'xboost':XGBClassifier()
            }


            model_report:dict=evaluate_models(X_train,y_train,X_test,y_test,models)
            print(model_report)

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            best_model = models[best_model_name]
            print(f'Best Model Found , Model Name : {best_model_name} , Accu : {best_model_score}')
            print('\n====================================================================================\n')


            save_object(
                file_path=self.trained_model_config.trained_model_trainer_config,
                obj=best_model
            ) 

        except Exception as e:
            raise CustomException(e, sys)    