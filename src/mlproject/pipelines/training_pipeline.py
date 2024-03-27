from src.mlproject.logger import logging
from src.mlproject.components.data_ingestion import DataIngestion
from src.mlproject.components.data_transformation import DataTransformation
from src.mlproject.components.model_trainer import ModelTrainer

data_ingestion=DataIngestion()
train_data_path, test_data_path=data_ingestion.initiate_data_ingestion()

data_transfom=DataTransformation()
train_arr, test_arr =data_transfom.initialize_data_transformation(train_data_path, test_data_path)

model_trainer_obj=ModelTrainer()
model_trainer_obj.initiate_model_trainer(train_arr,test_arr)