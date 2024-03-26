import os
import sys
import pandas as pd
import numpy as np

from src.mlproject.logger import logging
from src.mlproject.exception import CustomException

from dataclasses import dataclass
from sklearn.model_selection import train_test_split


@dataclass
class DataIngestionConfig:

    raw_data_path: str = os.path.join("artifacts", "raw.csv")
    train_data_path: str = os.path.join("artifacts","train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")



class DataIngestion:

    def __init__(self):

        self.data_ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            logging.info("data ingestion started ............")

            data=pd.read_csv("https://raw.githubusercontent.com/Vishnuu011/datastore/main/healthcare-dataset-stroke-data.csv")

            os.makedirs(os.path.dirname(os.path.join(self.data_ingestion_config.raw_data_path)), exist_ok=True)
            data.to_csv(self.data_ingestion_config.raw_data_path, index=False)

            logging.info("raw data splited train and test .....................")

            train_data, test_data=train_test_split(data, test_size=0.2)
            train_data.to_csv(self.data_ingestion_config.train_data_path, index=False)
            test_data.to_csv(self.data_ingestion_config.test_data_path, index=False)

            logging.info("train and test have saved in atifact ................")

            logging.info("data ingestion completed ................")

            return (
                self.data_ingestion_config.train_data_path,
                self.data_ingestion_config.test_data_path
            )


        except Exception as e:
            raise CustomException(e, sys)