import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        color: white;
        background-color: #007bff;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #0056b3;
    }
    h1 {
        color: #343a40;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    h2, h3 {
        color: #495057;
    }
    .css-1d391kg {
        padding-top: 3rem;
    }
    .metric-container {
        border: 1px solid #dee2e6;
        border-radius: 5px;
        padding: 1rem;
        background-color: white;
        text-align: center;
        box-shadow: 0 0.125rem 0.25rem rgba(0,0,0,0.075);
    }
    </style>
    """, unsafe_allow_html=True)

# Application Title
st.title("🔮 Customer Churn Prediction Dashboard")
st.markdown("### Predict customer retention with AI-powered insights")
st.divider()

# Load Model and Preprocessor
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load('artifacts/model.pkl')
        preprocessor = joblib.load('artifacts/preprocessor.pkl')
        return model, preprocessor
    except FileNotFoundError:
        st.error("⚠️ Model files not found! Please run the training pipeline first.")
        return None, None

model, preprocessor = load_artifacts()

if model and preprocessor:
    # Sidebar: Input Features
    st.sidebar.header("📝 Customer Profile")
    
    def user_input_features():
        # Group 1: Demographics
        with st.sidebar.expander("👤 Demographics", expanded=True):
            gender = st.selectbox("Gender", ["Male", "Female"], index=None, placeholder="Select Gender...")
            senior = st.selectbox("Senior Citizen", ["0", "1"], format_func=lambda x: "Yes" if x == "1" else "No", index=None, placeholder="Select Status...")
            partner = st.selectbox("Partner", ["Yes", "No"], index=None, placeholder="Select Partner Status...")
            dependents = st.selectbox("Dependents", ["Yes", "No"], index=None, placeholder="Select Dependents...")

        # Group 2: Services
        with st.sidebar.expander("📡 Services", expanded=False):
            phone = st.selectbox("Phone Service", ["Yes", "No"], index=None, placeholder="Select Phone Service...")
            multiple = st.selectbox("Multiple Lines", ["No", "Yes"], index=None, placeholder="Select Multiple Lines...")
            internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"], index=None, placeholder="Select Internet Service...")
            security = st.selectbox("Online Security", ["No", "Yes"], index=None, placeholder="Select Online Security...")
            backup = st.selectbox("Online Backup", ["No", "Yes"], index=None, placeholder="Select Online Backup...")
            protection = st.selectbox("Device Protection", ["No", "Yes"], index=None, placeholder="Select Device Protection...")
            support = st.selectbox("Tech Support", ["No", "Yes"], index=None, placeholder="Select Tech Support...")
            tv = st.selectbox("Streaming TV", ["No", "Yes"], index=None, placeholder="Select Streaming TV...")
            movies = st.selectbox("Streaming Movies", ["No", "Yes"], index=None, placeholder="Select Streaming Movies...")

        # Group 3: Account Info
        with st.sidebar.expander("💳 Account Information", expanded=True):
            tenure = st.slider("Tenure (Months)", 0, 72, 0, help="Number of months the customer has stayed with the company")
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"], index=None, placeholder="Select Contract Type...")
            billing = st.selectbox("Paperless Billing", ["Yes", "No"], index=None, placeholder="Select Billing Preference...")
            payment = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
            ], index=None, placeholder="Select Payment Method...")
            
            # Key fix: Initial value 0.00, step 1.0, format %.2f
            monthly = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=500.0, value=0.00, step=1.0, format="%.2f")
            total = st.number_input("Total Charges ($)", min_value=0.0, max_value=20000.0, value=0.00, step=1.0, format="%.2f")

        data = {
            'gender': gender,
            'SeniorCitizen': int(senior) if senior is not None else None,
            'Partner': partner,
            'Dependents': dependents,
            'tenure': tenure,
            'PhoneService': phone,
            'MultipleLines': multiple,
            'InternetService': internet,
            'OnlineSecurity': security,
            'OnlineBackup': backup,
            'DeviceProtection': protection,
            'TechSupport': support,
            'StreamingTV': tv,
            'StreamingMovies': movies,
            'Contract': contract,
            'PaperlessBilling': billing,
            'PaymentMethod': payment,
            'MonthlyCharges': monthly,
            'TotalCharges': total
        }
        
        return pd.DataFrame(data, index=[0]), data

    input_df, input_dict = user_input_features()

    # Main Panel Layout
    col1, col2 = st.columns([1, 1.5], gap="medium")

    with col1:
        st.subheader("🔍 Customer Summary")
        st.dataframe(input_df.T.rename(columns={0: 'Value'}), use_container_width=True)

    with col2:
        st.subheader("📊 Prediction Model")
        
        # Validation Check
        missing_fields = [k for k, v in input_dict.items() if v is None]
        
        if missing_fields:
            st.warning("⚠️ Please fill in all fields in the sidebar to proceed with prediction.")
            st.info(f"Missing: {', '.join(missing_fields)}")
        else:
            st.markdown("Click the button below to analyze the churn risk for this profile.")
            
            if st.button("🚀 Predict Churn Risk", use_container_width=True):
                with st.spinner('Analyzing...'):
                    try:
                        # Use PredictPipeline for inference
                        from src.pipeline.predict_pipeline import CustomData, PredictPipeline
                        
                        custom_data = CustomData(**input_dict)
                        features_df = custom_data.get_data_as_data_frame()
                        
                        predict_pipeline = PredictPipeline()
                        prediction, probability = predict_pipeline.predict(features_df)
                        
                        prediction = prediction[0]
                        probability = probability[0][1]
                        
                        # Display Results
                        st.divider()
                        st.markdown("### Analysis Result")
                        
                        if prediction == 1:
                            st.error(f"⚠️ **High Churn Risk Detected**")
                            st.markdown(f"This customer has a **{probability:.1%}** probability of leaving.")
                        else:
                            st.success(f"✅ **Low Churn Risk**")
                            st.markdown(f"This customer is likely to stay (Probability of churn: **{probability:.1%}**).")
                        
                        # Gauge Chart
                        fig = go.Figure(go.Indicator(
                            mode = "gauge+number",
                            value = probability * 100,
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            title = {'text': "Churn Probability (%)", 'font': {'size': 24}},
                            gauge = {
                                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                                'bar': {'color': "rgba(0,0,0,0)"}, # hide default bar, use steps
                                'bgcolor': "white",
                                'borderwidth': 2,
                                'bordercolor': "gray",
                                'steps': [
                                    {'range': [0, 50], 'color': "#28a745"},
                                    {'range': [50, 80], 'color': "#ffc107"},
                                    {'range': [80, 100], 'color': "#dc3545"}
                                ],
                                'threshold': {
                                    'line': {'color': "black", 'width': 4},
                                    'thickness': 0.75,
                                    'value': probability * 100
                                }
                            }
                        ))
                        fig.update_layout(height=400, margin=dict(l=20, r=20, t=50, b=20))
                        st.plotly_chart(fig, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"An error occurred during prediction: {e}")
                        st.error(f"Details: {str(e)}")

else:
    st.info("Awaiting model generation... Please ensure the pipeline has been run.")
