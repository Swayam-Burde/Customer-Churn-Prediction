import pandas as pd
import joblib
import os
from src.components.data_transformation import FeatureEngineering

# Load data and artifacts
try:
    test_df = pd.read_csv('artifacts/test.csv')
    model = joblib.load('artifacts/model.pkl')
    preprocessor = joblib.load('artifacts/preprocessor.pkl')
except FileNotFoundError as e:
    print(f"Error: {e}")
    exit()

# Separate features
X_test = test_df.drop('Churn', axis=1)

# Preprocess (must use the pipeline which includes FeatureEngineering)
# Note: The 'preprocessor.pkl' saved in DataTransformation.get_data_transformer_object() 
# INCLUDES the FeatureEngineering step inside the pipeline.
# So we can just call preprocessor.transform(X_test) directly.

try:
    processed_data = preprocessor.transform(X_test)
    
    # Predict Probabilities
    probs = model.predict_proba(processed_data)[:, 1]
    
    # Add to dataframe
    X_test['Churn_Probability'] = probs
    
    # Sort and get top 3
    top_churners = X_test.sort_values(by='Churn_Probability', ascending=False).head(3)
    
    print("\n--- High Churn Profiles ---")
    for i, row in top_churners.iterrows():
        print(f"\nProfile {i+1} (Prob: {row['Churn_Probability']:.2f}):")
        print(row[['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 
                   'PhoneService', 'InternetService', 'OnlineSecurity', 'TechSupport', 
                   'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges']].to_string())
                   
except Exception as e:
    print(f"Error during inference: {e}")
