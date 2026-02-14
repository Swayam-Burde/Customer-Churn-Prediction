import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            # Define best models from notebook findings
            xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)
            lgbm = LGBMClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42, verbose=-1)
            cat = CatBoostClassifier(iterations=200, depth=5, learning_rate=0.05, verbose=0, random_state=42)
            rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)

            estimators = [
                ('xgb', xgb),
                ('lgbm', lgbm),
                ('cat', cat),
                ('rf', rf)
            ]

            logging.info("Initializing Stacking Classifier")
            model = StackingClassifier(
                estimators=estimators,
                final_estimator=LogisticRegression(),
                cv=5,
                n_jobs=-1
            )
            
            logging.info("Training Model...")
            model.fit(X_train, y_train)

            logging.info("Model Training Complete. Evaluating...")
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            
            accuracy = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_prob)
            
            logging.info(f"Model Accuracy: {accuracy:.4f}")
            logging.info(f"Model ROC-AUC: {auc:.4f}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=model
            )

            logging.info("Model Saved")
            
            return accuracy

        except Exception as e:
            raise CustomException(e, sys)
