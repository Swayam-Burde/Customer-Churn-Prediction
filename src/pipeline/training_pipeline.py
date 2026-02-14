import sys
import os

# Add project root to system path to resolve 'src' module
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(project_root)

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException
from src.logger import logging

if __name__ == "__main__":
    try:
        logging.info("Training pipeline started")
        
        # 1. Data Ingestion
        obj = DataIngestion()
        train_data, test_data = obj.initiate_data_ingestion()
        logging.info("Data Ingestion Completed")

        # 2. Data Transformation
        data_transformation = DataTransformation()
        train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)
        logging.info("Data Transformation Completed")

        # 3. Model Training
        model_trainer = ModelTrainer()
        accuracy = model_trainer.initiate_model_trainer(train_arr, test_arr)
        logging.info(f"Model Training Completed with Accuracy: {accuracy}")
        
    except Exception as e:
        logging.error("Exception occurred in Training Pipeline")
        raise CustomException(e, sys)
