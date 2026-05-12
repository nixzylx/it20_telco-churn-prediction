import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Telco Churn Prediction System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .main-header h1 {
        color: white !important;
    }
    .metric-card {
        background-color: #1e1e2f;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        border: 1px solid #4a4a6a;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header"><h1>📊 Telecommunications Customer Churn Prediction System</h1></div>', unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 1.2rem;'>Supervised Learning System using Gradient Boosting and Random Forest</p>", unsafe_allow_html=True)

# ============================================
# DATA LOADING AND PREPROCESSING
# ============================================
@st.cache_data
def load_and_preprocess_data():
    """Load and preprocess the telco churn dataset"""
    import os
    
    # Try multiple paths
    possible_paths = [
        "WA_Fn-UseC_-Telco-Customer-Churn.csv",
        "telco_churn.csv",
        "data/WA_Fn-UseC_-Telco-Customer-Churn.csv",
        "data/telco_churn.csv",
        "../WA_Fn-UseC_-Telco-Customer-Churn.csv",
        "../data/WA_Fn-UseC_-Telco-Customer-Churn.csv",
        "../../WA_Fn-UseC_-Telco-Customer-Churn.csv"
    ]
    
    df = None
    used_path = None
    
    for path in possible_paths:
        if os.path.exists(path):
            df = pd.read_csv(path)
            used_path = path
            break
    
    if df is None:
        st.error("❌ Could not find the Telco Churn CSV file!")
        st.info("Please ensure the file is in the same directory as app.py")
        st.stop()
    
    # Display original shape
    st.write(f"Original dataset shape: {df.shape}")
    
    # Remove customerID
    if "customerID" in df.columns:
        df.drop("customerID", axis=1, inplace=True)
    
    # Convert TotalCharges to numeric (critical step!)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    
    # Drop rows with missing TotalCharges (there are 11 such rows)
    initial_rows = len(df)
    df = df.dropna(subset=['TotalCharges'])
    st.write(f"Removed {initial_rows - len(df)} rows with missing TotalCharges")
    
    # Remove any duplicates
    df = df.drop_duplicates()
    
    # Convert Churn to binary (Yes=1, No=0)
    df['Churn'] = (df['Churn'] == 'Yes').astype(int)
    
    return df, used_path

# Load data
with st.spinner("Loading and preprocessing data..."):
    df, data_path = load_and_preprocess_data()

st.success(f"✅ Data loaded successfully from: `{data_path}`")
st.write(f"Final dataset shape: {df.shape}")

# ============================================
# ONE-HOT ENCODING
# ============================================
# Get categorical columns (excluding Churn)
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

# One-hot encode categorical variables
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=False)

# Prepare features and target
X = df_encoded.drop('Churn', axis=1)
y = df_encoded['Churn']

# Verify no NaN values remain
if X.isnull().any().any():
    st.warning(f"Found {X.isnull().sum().sum()} missing values. Filling with 0...")
    X = X.fillna(0)

st.success(f"✅ Data encoded! Features: {X.shape[1]}, Samples: {X.shape[0]}")

# ============================================
# TRAIN-TEST SPLIT
# ============================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================
# TRAIN MODELS
# ============================================
@st.cache_resource
def train_models():
    """Train models with optimized parameters"""
    
    # Model 1: Random Forest (handles class imbalance)
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    # Model 2: Gradient Boosting
    gb_model = GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=4,
        min_samples_split=5,
        min_samples_leaf=2,
        subsample=0.8,
        random_state=42
    )
    
    # Train models
    rf_model.fit(X_train_scaled, y_train)
    gb_model.fit(X_train_scaled, y_train)
    
    # Make predictions
    rf_pred = rf_model.predict(X_test_scaled)
    gb_pred = gb_model.predict(X_test_scaled)
    rf_proba = rf_model.predict_proba(X_test_scaled)[:, 1]
    gb_proba = gb_model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate metrics
    metrics = {
        "Random Forest": {
            "Accuracy": accuracy_score(y_test, rf_pred),
            "Precision": precision_score(y_test, rf_pred),
            "Recall": recall_score(y_test, rf_pred),
            "F1-Score": f1_score(y_test, rf_pred),
            "ROC-AUC": roc_auc_score(y_test, rf_proba)
        },
        "Gradient Boosting": {
            "Accuracy": accuracy_score(y_test, gb_pred),
            "Precision": precision_score(y_test, gb_pred),
            "Recall": recall_score(y_test, gb_pred),
            "F1-Score": f1_score(y_test, gb_pred),
            "ROC-AUC": roc_auc_score(y_test, gb_proba)
        }
    }
    
    # Calculate ensemble predictions
    ensemble_proba = (rf_proba + gb_proba) / 2
    ensemble_pred = (ensemble_proba >= 0.5).astype(int)
    
    metrics["Ensemble"] = {
        "Accuracy": accuracy_score(y_test, ensemble_pred),
        "Precision": precision_score(y_test, ensemble_pred),
        "Recall": recall_score(y_test, ensemble_pred),
        "F1-Score": f1_score(y_test, ensemble_pred),
        "ROC-AUC": roc_auc_score(y_test, ensemble_proba)
    }
    
    return {
        "models": {
            "Random Forest": rf_model,
            "Gradient Boosting": gb_model
        },
        "metrics": metrics,
        "feature_names": X.columns.tolist(),
        "scaler": scaler
    }

# Train models
with st.spinner("Training models..."):
    model_data = train_models()

models = model_data["models"]
metrics = model_data["metrics"]
feature_names = model_data["feature_names"]
scaler = model_data["scaler"]

# ============================================
# DISPLAY MODEL PERFORMANCE
# ============================================
st.subheader("📈 Model Performance Comparison")

metrics_df = pd.DataFrame(metrics).T.round(4)
st.dataframe(
    metrics_df.style.highlight_max(axis=0, color='lightgreen'),
    use_container_width=True
)

best_model_name = max(metrics, key=lambda x: metrics[x]["F1-Score"])
st.success(f"✅ Best model: **{best_model_name}** (F1-Score: {metrics[best_model_name]['F1-Score']:.2%})")

# ============================================
# USER INPUT FORM
# ============================================
st.sidebar.header("📝 Customer Information")
st.sidebar.markdown("Enter customer details below")

# Load original data for reference
original_df = pd.read_csv(data_path)

# Create input form
input_data = {}

with st.sidebar.form(key="prediction_form"):
    st.markdown("### 👤 Demographics")
    
    # Gender
    input_data['gender'] = st.selectbox("Gender", ['Male', 'Female'])
    
    # Senior Citizen
    input_data['SeniorCitizen'] = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes (65+)" if x == 1 else "No")
    
    # Partner
    input_data['Partner'] = st.selectbox("Partner", ['Yes', 'No'])
    
    # Dependents
    input_data['Dependents'] = st.selectbox("Dependents", ['Yes', 'No'])
    
    st.markdown("---")
    st.markdown("### 📞 Services")
    
    # Tenure
    input_data['tenure'] = st.slider("Tenure (months)", 0, 72, 12, 1)
    
    # Phone Service
    input_data['PhoneService'] = st.selectbox("Phone Service", ['Yes', 'No'])
    
    # Multiple Lines
    if input_data['PhoneService'] == 'Yes':
        input_data['MultipleLines'] = st.selectbox("Multiple Lines", ['Yes', 'No'])
    else:
        input_data['MultipleLines'] = 'No phone service'
    
    # Internet Service
    input_data['InternetService'] = st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])
    
    # Internet add-ons (only if customer has internet)
    if input_data['InternetService'] != 'No':
        col1, col2 = st.columns(2)
        with col1:
            input_data['OnlineSecurity'] = st.selectbox("Online Security", ['Yes', 'No'])
            input_data['OnlineBackup'] = st.selectbox("Online Backup", ['Yes', 'No'])
            input_data['DeviceProtection'] = st.selectbox("Device Protection", ['Yes', 'No'])
        with col2:
            input_data['TechSupport'] = st.selectbox("Tech Support", ['Yes', 'No'])
            input_data['StreamingTV'] = st.selectbox("Streaming TV", ['Yes', 'No'])
            input_data['StreamingMovies'] = st.selectbox("Streaming Movies", ['Yes', 'No'])
    else:
        # Set all to 'No internet service'
        for service in ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']:
            input_data[service] = 'No internet service'
    
    st.markdown("---")
    st.markdown("### 💰 Billing")
    
    # Contract
    input_data['Contract'] = st.selectbox("Contract Type", ['Month-to-month', 'One year', 'Two year'])
    
    # Paperless Billing
    input_data['PaperlessBilling'] = st.selectbox("Paperless Billing", ['Yes', 'No'])
    
    # Payment Method
    input_data['PaymentMethod'] = st.selectbox("Payment Method", [
        'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'
    ])
    
    # Monthly Charges
    input_data['MonthlyCharges'] = st.number_input("Monthly Charges ($)", 18.0, 120.0, 65.0, 5.0)
    
    # Total Charges
    input_data['TotalCharges'] = st.number_input("Total Charges ($)", 0.0, 9000.0, 1000.0, 100.0)
    
    submitted = st.form_submit_button("🔮 Predict Churn", type="primary", use_container_width=True)

# ============================================
# PREDICTION
# ============================================
if submitted:
    # Convert input to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # One-hot encode
    input_encoded = pd.get_dummies(input_df)
    
    # Ensure all columns from training are present
    for col in feature_names:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    
    # Reorder columns to match training
    input_encoded = input_encoded[feature_names]
    
    # Scale
    input_scaled = scaler.transform(input_encoded)
    
    # Make predictions
    st.markdown("---")
    st.subheader("🎯 Prediction Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        rf_prob = models["Random Forest"].predict_proba(input_scaled)[0][1]
        rf_pred = models["Random Forest"].predict(input_scaled)[0]
        st.markdown(f"""
        <div class='metric-card'>
            <h3>🌲 Random Forest</h3>
            <h2>{rf_prob:.1%}</h2>
            <p>Churn Probability</p>
            <h3>{'⚠️ Will Churn' if rf_pred == 1 else '✅ Will Stay'}</h3>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        gb_prob = models["Gradient Boosting"].predict_proba(input_scaled)[0][1]
        gb_pred = models["Gradient Boosting"].predict(input_scaled)[0]
        st.markdown(f"""
        <div class='metric-card'>
            <h3>🚀 Gradient Boosting</h3>
            <h2>{gb_prob:.1%}</h2>
            <p>Churn Probability</p>
            <h3>{'⚠️ Will Churn' if gb_pred == 1 else '✅ Will Stay'}</h3>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        ensemble_prob = (rf_prob + gb_prob) / 2
        ensemble_pred = 1 if ensemble_prob > 0.5 else 0
        st.markdown(f"""
        <div class='metric-card'>
            <h3>🎲 Ensemble</h3>
            <h2>{ensemble_prob:.1%}</h2>
            <p>Churn Probability</p>
            <h3>{'⚠️ Will Churn' if ensemble_pred == 1 else '✅ Will Stay'}</h3>
        </div>
        """, unsafe_allow_html=True)
    
    # Final recommendation
    st.markdown("---")
    if ensemble_pred == 1:
        st.error(f"## ⚠️ HIGH RISK: Customer Likely to Churn\n### Probability: {ensemble_prob:.1%}")
        st.progress(ensemble_prob)
        
        if ensemble_prob > 0.7:
            st.warning("🚨 **Immediate Action Required!** Offer loyalty discount or upgrade contract.")
        elif ensemble_prob > 0.4:
            st.warning("⚠️ **Medium Risk** - Consider sending retention offer.")
    else:
        st.success(f"## ✅ LOW RISK: Customer Likely to Stay\n### Probability: {ensemble_prob:.1%}")
        st.progress(ensemble_prob)
        st.info("💡 Continue providing excellent service and consider upsell opportunities.")
    
    # Show input summary
    with st.expander("📋 Customer Information Summary"):
        st.dataframe(pd.DataFrame([input_data]), use_container_width=True)

# ============================================
# FEATURE IMPORTANCE
# ============================================
st.markdown("---")
st.subheader("📊 Feature Importance Analysis")

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": models["Gradient Boosting"].feature_importances_
}).sort_values(by="Importance", ascending=False).head(20)

fig = px.bar(importance_df, x="Importance", y="Feature", orientation='h',
             title="Top 20 Most Important Features",
             color="Importance", color_continuous_scale="Viridis")
fig.update_layout(height=600)
st.plotly_chart(fig, use_container_width=True)

# ============================================
# DATA PREVIEW
# ============================================
st.markdown("---")
st.subheader("🔍 Dataset Preview")
original_df = pd.read_csv(data_path)
st.dataframe(original_df.head(20), use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 1rem;'>
    <p>🎓 <strong>Predictive Analytics Project</strong> | Telco Customer Churn Prediction System</p>
    <p>Built with Streamlit • Random Forest • Gradient Boosting • Python</p>
</div>
""", unsafe_allow_html=True)