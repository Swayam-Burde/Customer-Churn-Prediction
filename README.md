# 🔮 Customer Churn Prediction

An end-to-end machine learning project to predict customer churn in the telecommunications industry. This project utilizes a modular architecture to ingest data, transform features, train advanced ensemble models, and deploy a user-friendly Streamlit web application for real-time predictions.

## 🚀 Key Features

*   **Modular Architecture**: Clean, maintainable code structure separated into data ingestion, transformation, and model training components.
*   **Advanced Modeling**: Implements a Stacking Ensemble model (combining XGBoost, Gradient Boosting, Random Forest, etc.) achieving an **ROC-AUC of ~0.85**.
*   **Robust Preprocessing**: Automated pipelines for handling missing values, scaling numerical features, and encoding categorical variables.
*   **Interactive Dashboard**: A professional **Streamlit** web app allowing users to input customer details and get instant churn risk assessments.
*   **Explainability**: Incorporates feature importance analysis and probability scores to understand risk factors.

## 🛠️ Tech Stack

*   **Language**: Python 3.9+
*   **Web Framework**: Streamlit
*   **Machine Learning**: Scikit-Learn, XGBoost, CatBoost, LightGBM
*   **Data Manipulation**: Pandas, NumPy
*   **Visualization**: Plotly, Matplotlib, Seaborn
*   **DevOps**: Docker (optional), GitHub Actions (ready for CI/CD)

## 📂 Directory Structure

```plaintext
Customer_Churn_Prediction/
├── .gitignore               # Git exclusion rules
├── README.md                # Project documentation
├── app.py                   # Streamlit Web Application entry point
├── requirements.txt         # Project dependencies
├── artifacts/               # Generated models and preprocessors (Ignored by Git)
│   ├── model.pkl
│   ├── preprocessor.pkl
│   ├── train.csv
│   └── test.csv
├── notebooks/               # Jupyter Notebooks for experimentation
│   ├── 01_EDA.ipynb
│   ├── 02_Preprocessing.ipynb
│   └── 03_Modeling.ipynb
└── src/                     # Source Code
    ├── __init__.py
    ├── exception.py         # Custom Exception Handling
    ├── logger.py            # Logging Configuration
    ├── utils.py             # Helper utility functions
    ├── components/          # ML Components
    │   ├── __init__.py
    │   ├── data_ingestion.py
    │   ├── data_transformation.py
    │   └── model_trainer.py
    └── pipeline/            # Execution Pipelines
        ├── __init__.py
        ├── predict_pipeline.py
        └── training_pipeline.py
```

## ⚙️ Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/Customer_Churn_Prediction.git
    cd Customer_Churn_Prediction
    ```

2.  **Create and activate a virtual environment**:
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # Mac/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## 🏃 Usage

### 1. Training the Model
To re-train the model from scratch using the raw data, run the training pipeline:
```bash
python src/pipeline/training_pipeline.py
```
This will generate `model.pkl` and `preprocessor.pkl` in the `artifacts/` folder.

### 2. Running the Web App
Launch the Streamlit dashboard to interact with the model:
```bash
streamlit run app.py
```
The app will open in your default browser at `http://localhost:8501`.

## 📊 Model Performance

The final **Stacking Ensemble** model was selected after extensive experimentation.

*   **ROC-AUC Score**: 0.8456
*   **Accuracy**: ~80%
*   **Key Drivers**: Contract type, Tenure, Internet Service (Fiber Optic), and Electronic Check payment method.

## 🤝 Contributing

Contributions are welcome! Please fork this repository and submit a pull request for any features, bug fixes, or enhancements.
