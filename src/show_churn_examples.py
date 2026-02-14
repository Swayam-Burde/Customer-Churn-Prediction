import pandas as pd

try:
    # Read test data
    df = pd.read_csv('artifacts/test.csv')
    
    # Filter for churners
    churners = df[df['Churn'] == 1]
    
    print(f"\nFound {len(churners)} actual churners in test set.")
    print("Here are 3 example configurations of customers who churned:\n")
    
    # Select 3 examples
    examples = churners.head(3)
    
    for i, row in examples.iterrows():
        print(f"--- Example {i+1} ---")
        print(f"Gender: {row['gender']}")
        print(f"Senior Citizen: {row['SeniorCitizen']}")
        print(f"Partner: {row['Partner']}")
        print(f"Dependents: {row['Dependents']}")
        print(f"Tenure: {row['tenure']}")
        print(f"Phone Service: {row['PhoneService']}")
        print(f"Multiple Lines: {row['MultipleLines']}")
        print(f"Internet Service: {row['InternetService']}")
        print(f"Online Security: {row['OnlineSecurity']}")
        print(f"Online Backup: {row['OnlineBackup']}")
        print(f"Device Protection: {row['DeviceProtection']}")
        print(f"Tech Support: {row['TechSupport']}")
        print(f"Streaming TV: {row['StreamingTV']}")
        print(f"Streaming Movies: {row['StreamingMovies']}")
        print(f"Contract: {row['Contract']}")
        print(f"Paperless Billing: {row['PaperlessBilling']}")
        print(f"Payment Method: {row['PaymentMethod']}")
        print(f"Monthly Charges: {row['MonthlyCharges']}")
        print(f"Total Charges: {row['TotalCharges']}")
        print("\n")

except Exception as e:
    print(f"Error reading data: {e}")
