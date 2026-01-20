"""
KenyaCredit AI - Streamlit Application
Main dashboard for credit scoring with alternative data
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from credit_model import CreditScoringModel
from alt_data import AlternativeDataCollector
from utils import (
    create_sample_data, 
    format_currency, 
    get_risk_category, 
    get_risk_color,
    calculate_var
)

# Page configuration
st.set_page_config(
    page_title="KenyaCredit AI",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = CreditScoringModel()
    st.session_state.alt_data_collector = AlternativeDataCollector()
    st.session_state.trained = False

# Header
st.markdown('<p class="main-header">KenyaCredit AI</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Alternative Data Credit Scoring for Financial Inclusion</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://via.placeholder.com/300x100/1f77b4/ffffff?text=KenyaCredit+AI", use_container_width=True)
    
    st.markdown("---")
    
    page = st.radio(
        "Navigation",
        ["Home", "Credit Assessment", "Portfolio Analytics", "About"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    st.markdown("### MSCFE Project")
    st.markdown("**Author:** Peterson Mutegi")
    st.markdown("**Modules:**")
    st.markdown("- M1: Credit Risk")
    st.markdown("- M3: Portfolio Theory")
    st.markdown("- M4: Alternative Data")
    st.markdown("- M5: News Sentiment")
    
    st.markdown("---")
    
    # Train model button
    if st.button("Train/Retrain Model", use_container_width=True):
        with st.spinner("Training model..."):
            # Create sample data
            train_data = create_sample_data(n_samples=500)
            
            # Train model
            metrics = st.session_state.model.train(train_data)
            st.session_state.trained = True
            
            st.success(f"Model trained successfully!")
            st.info(f"Test Accuracy: {metrics['test_accuracy']:.2%}")

# Main content area
if page == "Home":
    # Home page
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### Mission")
        st.write("""
        Expand credit access to underbanked Kenyans using innovative 
        alternative data sources combined with traditional credit analysis.
        """)
    
    with col2:
        st.markdown("### Technology")
        st.write("""
        Machine learning models trained on traditional financial data plus 
        social media sentiment, Google Trends, and news analysis.
        """)
    
    with col3:
        st.markdown("### Impact")
        st.write("""
        Reduce default rates by 15-20% while increasing approval rates 
        for creditworthy but data-sparse borrowers.
        """)
    
    st.markdown("---")
    
    # Key metrics (demo)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Applicants Scored", "12,450", "+1,250")
    
    with col2:
        st.metric("Model Accuracy", "87.5%", "+2.3%")
    
    with col3:
        st.metric("Default Rate", "8.2%", "-3.1%")
    
    with col4:
        st.metric("Avg Processing Time", "45 sec", "-15 sec")
    
    st.markdown("---")
    
    # How it works
    st.markdown("### How It Works")
    
    flow_col1, flow_col2, flow_col3, flow_col4 = st.columns(4)
    
    with flow_col1:
        st.markdown("#### 1 Data Collection")
        st.write("Gather traditional + alternative data")
    
    with flow_col2:
        st.markdown("#### 2 Feature Engineering")
        st.write("TF-IDF, sentiment analysis, risk indicators")
    
    with flow_col3:
        st.markdown("#### 3 ML Prediction")
        st.write("Random Forest model predicts default probability")
    
    with flow_col4:
        st.markdown("#### 4 Credit Score")
        st.write("Generate 300-850 score + recommendations")
    
    st.markdown("---")
    
    # Sample data view
    st.markdown("### Sample Training Data")
    sample_df = create_sample_data(n_samples=10)
    st.dataframe(sample_df, use_container_width=True)

elif page == "Credit Assessment":
    st.markdown("## Individual Credit Assessment")
    
    if not st.session_state.trained:
        st.warning("Please train the model first using the sidebar button.")
    else:
        # Input form
        st.markdown("### Applicant Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Traditional Data")
            
            income = st.number_input(
                "Monthly Income (KES)",
                min_value=10000,
                max_value=500000,
                value=50000,
                step=5000
            )
            
            existing_debt = st.number_input(
                "Existing Debt (KES)",
                min_value=0,
                max_value=300000,
                value=20000,
                step=5000
            )
            
            employment_months = st.number_input(
                "Employment Duration (months)",
                min_value=0,
                max_value=360,
                value=24,
                step=1
            )
            
            age = st.number_input(
                "Age",
                min_value=18,
                max_value=70,
                value=30,
                step=1
            )
        
        with col2:
            st.markdown("#### Alternative Data")
            
            st.info("These values are automatically collected from various sources")
            
            # Get alternative data
            alt_data = st.session_state.alt_data_collector.get_financial_stress_indicator()
            
            social_sentiment = st.slider(
                "Social Media Sentiment",
                min_value=-1.0,
                max_value=1.0,
                value=float(alt_data['social_sentiment']),
                step=0.1,
                help="Sentiment from financial discussions (-1=negative, 1=positive)"
            )
            
            google_trends_score = st.slider(
                "Google Trends Financial Stress",
                min_value=0.0,
                max_value=100.0,
                value=float(alt_data['google_trends_score']),
                step=1.0,
                help="Search volume for financial stress keywords (0=low stress, 100=high stress)"
            )
            
            news_sentiment = st.slider(
                "News Sentiment",
                min_value=-1.0,
                max_value=1.0,
                value=float(alt_data['news_sentiment']),
                step=0.1,
                help="Sentiment from recent economic news (-1=negative, 1=positive)"
            )
        
        st.markdown("---")
        
        # Predict button
        if st.button("Assess Credit Risk", type="primary", use_container_width=True):
            # Prepare applicant data
            applicant_data = {
                'income': income,
                'existing_debt': existing_debt,
                'employment_months': employment_months,
                'age': age,
                'social_sentiment': social_sentiment,
                'google_trends_score': google_trends_score,
                'news_sentiment': news_sentiment
            }
