# ============================================
# LOAN DEFAULT PREDICTOR - WEB APPLICATION
# Run with: streamlit run loan_app.py
# ============================================

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Loan Default Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load trained model and scaler
@st.cache_resource
def load_model():
    model = joblib.load('loan_default_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

try:
    model, scaler = load_model()
    model_loaded = True
except:
    model_loaded = False
    st.error("⚠️ Model not found. Please train the model first using loan_default.py")

# Custom CSS for better styling and dark mode compatibility
st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    /* Risk cards - using background colors that work in both light/dark mode */
    .risk-high {
        background-color: rgba(220, 38, 38, 0.15);
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #DC2626;
        color: #DC2626;
    }
    .risk-low {
        background-color: rgba(34, 197, 94, 0.15);
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #22C55E;
        color: #22C55E;
    }
    .risk-medium {
        background-color: rgba(245, 158, 11, 0.15);
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #F59E0B;
        color: #F59E0B;
    }
    
    /* Text inside risk cards - using theme-aware colors */
    .risk-content {
        color: inherit;
        font-size: 1rem;
    }
    
    .risk-content strong {
        color: inherit;
    }
    
    /* Metric styling */
    .metric-good {
        color: #22C55E;
        font-weight: bold;
    }
    .metric-bad {
        color: #DC2626;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">🏦 Loan Default Risk Predictor</div>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar for model info (removed "How to Run" section)
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/loan.png", width=80)
    st.header("About")
    st.info("""
    This AI-powered tool predicts the likelihood of loan default based on borrower characteristics.
    
    **Features used:**
    - Credit score & history
    - Income & debt ratios
    - Loan terms
    - Employment history
    
    Built with Random Forest ML model achieving 85%+ accuracy.
    """)
    
    st.header("Interpretation")
    st.markdown("""
    - **< 30%** → Low risk ✅
    - **30% - 60%** → Medium risk ⚠️
    - **> 60%** → High risk 🔴
    """)

# Main content area
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Borrower Financial Profile")
    
    credit_score = st.slider("Credit Score", 300, 850, 680, help="Higher is better. Above 670 is generally good.")
    annual_income = st.number_input("Annual Income (£)", min_value=10000, max_value=500000, value=50000, step=5000)
    debt_to_income = st.slider("Debt-to-Income Ratio (%)", 0, 100, 35, help="Monthly debt payments ÷ monthly income. Lower is better.")
    employment_years = st.slider("Years at Current Job", 0, 30, 5)
    
    st.subheader("💳 Credit History")
    num_late_payments = st.number_input("Number of Late Payments (last 2 years)", 0, 20, 1)
    credit_utilization = st.slider("Credit Utilization (%)", 0, 100, 30, help="Credit used ÷ credit available. Keep under 30%.")
    credit_age_years = st.slider("Length of Credit History (years)", 0, 40, 8)
    num_credit_inquiries = st.number_input("Recent Credit Inquiries", 0, 10, 1)

with col2:
    st.subheader("💰 Loan Details")
    loan_amount = st.number_input("Loan Amount (£)", min_value=1000, max_value=200000, value=25000, step=1000)
    interest_rate = st.slider("Interest Rate (%)", 3.0, 30.0, 12.0, 0.5)
    loan_term = st.selectbox("Loan Term", [12, 24, 36, 48, 60], format_func=lambda x: f"{x} months")
    num_open_accounts = st.slider("Number of Open Credit Accounts", 1, 20, 8)
    homeowner = st.radio("Homeowner Status", ["Yes", "No"])
    
    st.subheader("📈 Additional Context")
    
    # Determine delta color for DTI metric
    dti_delta = "Good" if debt_to_income < 36 else "High"
    dti_delta_color = "normal" if debt_to_income < 36 else "inverse"
    st.metric("Debt-to-Income Ratio", f"{debt_to_income}%", 
              delta=dti_delta, 
              delta_color=dti_delta_color)
    
    estimated_monthly_payment = (loan_amount * (interest_rate/100/12) * (1 + interest_rate/100/12)**loan_term) / ((1 + interest_rate/100/12)**loan_term - 1)
    st.metric("Estimated Monthly Payment", f"£{estimated_monthly_payment:.0f}")

# Create feature array for prediction
if model_loaded:
    homeowner_value = 1 if homeowner == "Yes" else 0
    
    input_data = pd.DataFrame([[
        credit_score,
        num_late_payments,
        credit_utilization,
        credit_age_years,
        loan_amount,
        interest_rate,
        loan_term,
        annual_income,
        debt_to_income,
        employment_years,
        num_open_accounts,
        num_credit_inquiries,
        homeowner_value
    ]], columns=[
        'credit_score', 'num_late_payments', 'credit_utilization', 'credit_age_years',
        'loan_amount', 'interest_rate', 'loan_term_months', 'annual_income',
        'debt_to_income', 'employment_years', 'num_open_accounts',
        'num_credit_inquiries', 'homeowner'
    ])
    
    # Scale and predict
    input_scaled = scaler.transform(input_data)
    probability = model.predict_proba(input_scaled)[0, 1]
    prediction = "Default" if probability > 0.5 else "No Default"
    
    st.markdown("---")
    st.subheader("🎯 Prediction Result")
    
    # Risk gauge using plotly
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = probability * 100,
        title = {'text': "Default Risk (%)", 'font': {'size': 24}},
        domain = {'x': [0, 1], 'y': [0, 1]},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkgray"},
            'bar': {'color': "#1E3A8A"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': "#DCFCE7"},
                {'range': [30, 60], 'color': "#FEF3C7"},
                {'range': [60, 100], 'color': "#FEE2E2"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': probability * 100
            }
        }
    ))
    fig.update_layout(height=300, paper_bgcolor='rgba(0,0,0,0)', font={'color': 'currentColor'})
    st.plotly_chart(fig, use_container_width=True)
    
    # Risk assessment card with improved dark mode compatibility
    risk_percentage = probability * 100
    
    if probability < 0.3:
        st.markdown(f"""
        <div class="risk-low">
            <div class="risk-content">
                <strong>✅ Low Risk ({risk_percentage:.1f}%)</strong><br>
                This loan application is likely to be approved. The borrower demonstrates strong creditworthiness.
            </div>
        </div>
        """, unsafe_allow_html=True)
    elif probability < 0.6:
        st.markdown(f"""
        <div class="risk-medium">
            <div class="risk-content">
                <strong>⚠️ Medium Risk ({risk_percentage:.1f}%)</strong><br>
                This loan requires additional review. Consider requesting documentation or adjusting terms.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="risk-high">
            <div class="risk-content">
                <strong>🔴 High Risk ({risk_percentage:.1f}%)</strong><br>
                This loan application shows significant risk factors. Recommend decline or secured loan.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Feature contribution analysis
    st.subheader("🔍 Key Risk Factors")
    
    # Highlight concerning factors
    concerns = []
    if credit_score < 620:
        concerns.append("⚠️ **Low credit score** - Below 620 significantly increases risk")
    if num_late_payments > 2:
        concerns.append(f"⚠️ **Frequent late payments** - {num_late_payments} late payments in recent history")
    if debt_to_income > 43:
        concerns.append(f"⚠️ **High debt burden** - DTI of {debt_to_income}% exceeds recommended 43%")
    if credit_utilization > 50:
        concerns.append(f"⚠️ **High credit utilization** - Using {credit_utilization}% of available credit")
    if employment_years < 1:
        concerns.append("⚠️ **Short employment history** - Less than 1 year at current job")
    
    if concerns:
        for concern in concerns:
            st.markdown(concern)
    else:
        st.success("✅ No major risk factors identified. All key metrics within acceptable ranges.")

# Footer
st.markdown("---")
st.caption("⚠️ Disclaimer: This tool is for educational purposes. Real lending decisions require comprehensive underwriting.")