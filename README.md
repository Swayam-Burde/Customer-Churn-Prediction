# Customer Churn Prediction

## Overview
This project focuses on predicting customer churn using machine learning. It utilizes a notebook-first approach for data exploration and model development, aiming to eventually deploy a Streamlit web application.

## Directory Structure
- `data/`:
    - `raw/`: Original dataset (Telco-Customer-Churn.csv).
    - `processed/`: Cleaned and ready-to-model data.
- `notebooks/`:
    - `01_EDA.ipynb`: Exploratory Data Analysis.
    - `02_Preprocessing.ipynb`: Data cleaning, encoding, and splitting.
    - `03_Modeling.ipynb`: Model training and evaluation.
- `src/`: Source code for the web application and shared utilities.

## Setup
1.  **Environment**: ensure you are in your project's virtual environment.
2.  **Dependencies**: Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run Notebooks**:
    ```bash
    jupyter notebook
    ```
    Navigate to the `notebooks/` directory to begin.
