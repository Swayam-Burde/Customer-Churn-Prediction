import sys
import os
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")

class FeatureEngineering(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        try:
            df = X.copy()
            
            # Numeric conversion if not already (handled in ingestion but good for safety)
            if 'TotalCharges' in df.columns and df['TotalCharges'].dtype == 'object':
                 df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
            
            # Tenure Group
            if 'tenure' in df.columns:
                bins = [0, 12, 24, 48, 60, 72]
                labels = ['0-1 Yr', '1-2 Yrs', '2-4 Yrs', '4-5 Yrs', '5+ Yrs']
                df['TenureGroup'] = pd.cut(df['tenure'], bins=bins, labels=labels, right=False)
                df['TenureGroup'] = df['TenureGroup'].astype(str) # Ensure string for OHE

            # Interactions
            if 'MonthlyCharges' in df.columns and 'tenure' in df.columns:
                df['Monthly_Tenure_Interaction'] = df['MonthlyCharges'] * df['tenure']
            
            if 'TotalCharges' in df.columns and 'MonthlyCharges' in df.columns:
                 df['Ratio_Total_Monthly'] = df['TotalCharges'] / (df['MonthlyCharges'] + 1)

            # Service Count
            services = ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 
                        'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
            
            # Check availability of columns
            available_services = [col for col in services if col in df.columns]
            if available_services:
                df['ServiceCount'] = df[available_services].apply(lambda x: x.isin(['Yes', 'Fiber optic', 'DSL']).sum(), axis=1)
            else:
                df['ServiceCount'] = 0

            return df
            
        except Exception as e:
            raise CustomException(e, sys)

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation
        '''
        try:
            # Define columns
            numerical_columns = ["tenure", "MonthlyCharges", "TotalCharges", 
                                 "Monthly_Tenure_Interaction", "Ratio_Total_Monthly", "ServiceCount"]
            
            categorical_columns = [
                "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService", "MultipleLines", 
                "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", 
                "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod", "TenureGroup"
            ]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            # Main Preprocessor (Applied AFTER Feature Engineering)
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipelines", cat_pipeline, categorical_columns)
                ]
            )
            
            # Full Pipeline
            full_pipeline = Pipeline(
                steps=[
                    ("feature_engineering", FeatureEngineering()),
                    ("preprocessor", preprocessor)
                ]
            )

            return full_pipeline

        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "Churn"
            
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe.")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e, sys)
