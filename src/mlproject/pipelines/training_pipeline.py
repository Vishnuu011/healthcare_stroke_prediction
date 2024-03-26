from src.mlproject.logger import logging
from src.mlproject.components.data_ingestion import DataIngestion

data_ingestion=DataIngestion()
train_data_path, test_data_path=data_ingestion.initiate_data_ingestion()