"""
KenyaCredit AI - Streamlit Application
Main dashboard for credit scoring with alternative data
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import sys
from pathlib import Path

# Add src directory to path for Streamlit Cloud compatibility
src_path = Path(__file__).parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

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
    st.markdown("### KenyaCredit AI")
    
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
    if st.button("Train/Retrain Model", width="stretch"):
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
    st.dataframe(sample_df, width="stretch")

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
        if st.button("Assess Credit Risk", type="primary", width="stretch"):
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
            
            # Get prediction
            with st.spinner("Analyzing credit risk..."):
                result = st.session_state.model.predict_score(applicant_data)
            
            st.markdown("---")
            st.markdown("## Credit Assessment Results")
            
            # Display results in columns
            res_col1, res_col2, res_col3 = st.columns(3)
            
            with res_col1:
                st.metric(
                    "Credit Score",
                    result['credit_score'],
                    help="Score range: 300-850"
                )
            
            with res_col2:
                st.metric(
                    "Risk Category",
                    result['risk_category']
                )
            
            with res_col3:
                st.metric(
                    "Default Probability",
                    f"{result['default_probability']:.1%}"
                )
            
            # Credit Score Gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=result['credit_score'],
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Credit Score"},
                gauge={
                    'axis': {'range': [300, 850]},
                    'bar': {'color': get_risk_color(result['risk_category'])},
                    'steps': [
                        {'range': [300, 450], 'color': "#ffcccc"},
                        {'range': [450, 600], 'color': "#ffe6cc"},
                        {'range': [600, 750], 'color': "#fff5cc"},
                        {'range': [750, 850], 'color': "#ccffcc"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': result['credit_score']
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, width="stretch")
            
            # Recommendation
            st.markdown("### Recommendation")
            
            if result['risk_category'] == "Low Risk":
                st.success(f"Recommended for approval at {result['recommended_interest_rate']:.1f}% interest rate")
            elif result['risk_category'] == "Medium Risk":
                st.warning(f"Consider approval with enhanced monitoring at {result['recommended_interest_rate']:.1f}% interest rate")
            elif result['risk_category'] == "High Risk":
                st.warning(f"Additional documentation required. If approved: {result['recommended_interest_rate']:.1f}% interest rate")
            else:
                st.error("Application declined - very high risk profile")

elif page == "Portfolio Analytics":
    st.markdown("## Portfolio Risk Analytics")
    
    # Generate sample portfolio data
    np.random.seed(42)
    n_loans = 100
    
    portfolio_data = pd.DataFrame({
        'loan_id': [f'LN{str(i).zfill(5)}' for i in range(1, n_loans + 1)],
        'credit_score': np.random.normal(650, 100, n_loans).clip(300, 850).astype(int),
        'loan_amount': np.random.normal(100000, 50000, n_loans).clip(10000, 500000),
        'interest_rate': np.random.uniform(12, 25, n_loans),
        'default_prob': np.random.beta(2, 10, n_loans)
    })
    
    # Add risk category
    portfolio_data['risk_category'] = portfolio_data['credit_score'].apply(get_risk_category)
    
    # Portfolio metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Loans", f"{n_loans:,}")
    
    with col2:
        st.metric("Portfolio Value", format_currency(portfolio_data['loan_amount'].sum()))
    
    with col3:
        avg_score = portfolio_data['credit_score'].mean()
        st.metric("Avg Credit Score", f"{avg_score:.0f}")
    
    with col4:
        expected_default = (portfolio_data['default_prob'] * portfolio_data['loan_amount']).sum()
        st.metric("Expected Loss", format_currency(expected_default))
    
    st.markdown("---")
    
    # Distribution charts
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.markdown("### Credit Score Distribution")
        fig_hist = px.histogram(
            portfolio_data, 
            x='credit_score', 
            nbins=20,
            color_discrete_sequence=['#1f77b4']
        )
        fig_hist.update_layout(showlegend=False)
        st.plotly_chart(fig_hist, width="stretch")
    
    with chart_col2:
        st.markdown("### Risk Category Breakdown")
        risk_counts = portfolio_data['risk_category'].value_counts()
        fig_pie = px.pie(
            values=risk_counts.values, 
            names=risk_counts.index,
            color=risk_counts.index,
            color_discrete_map={
                "Low Risk": "#28a745",
                "Medium Risk": "#ffc107",
                "High Risk": "#fd7e14",
                "Very High Risk": "#dc3545"
            }
        )
        st.plotly_chart(fig_pie, width="stretch")
    
    st.markdown("---")
    
    # VaR Analysis
    st.markdown("### Value at Risk (VaR) Analysis")
    
    # Simulate daily returns
    daily_returns = np.random.normal(0.0005, 0.02, 252)  # 1 year of daily returns
    
    var_95 = calculate_var(daily_returns, 0.95)
    var_99 = calculate_var(daily_returns, 0.99)
    
    var_col1, var_col2 = st.columns(2)
    
    with var_col1:
        st.metric("VaR (95%)", f"{var_95:.2%}", help="Maximum expected loss at 95% confidence")
    
    with var_col2:
        st.metric("VaR (99%)", f"{var_99:.2%}", help="Maximum expected loss at 99% confidence")
    
    # Returns distribution
    fig_returns = px.histogram(
        x=daily_returns, 
        nbins=50,
        title="Portfolio Returns Distribution",
        labels={'x': 'Daily Returns', 'y': 'Frequency'},
        color_discrete_sequence=['#1f77b4']
    )
    fig_returns.add_vline(x=var_95, line_dash="dash", line_color="orange", annotation_text="VaR 95%")
    fig_returns.add_vline(x=var_99, line_dash="dash", line_color="red", annotation_text="VaR 99%")
    st.plotly_chart(fig_returns, width="stretch")
    
    st.markdown("---")
    
    # Correlation matrix
    st.markdown("### Feature Correlation Analysis")
    
    correlation_data = portfolio_data[['credit_score', 'loan_amount', 'interest_rate', 'default_prob']].corr()
    
    fig_corr = px.imshow(
        correlation_data,
        text_auto='.2f',
        color_continuous_scale='RdBu_r',
        aspect='auto'
    )
    fig_corr.update_layout(title="Correlation Matrix")
    st.plotly_chart(fig_corr, width="stretch")
    
    st.markdown("---")
    
    # Portfolio data table
    st.markdown("### Portfolio Data")
    st.dataframe(portfolio_data.head(20), width="stretch")

elif page == "About":
    st.markdown("## About KenyaCredit AI")
    
    st.markdown("""
    ### Project Overview
    
    KenyaCredit AI is an innovative credit scoring system designed to expand financial 
    inclusion for underbanked populations in Kenya. By combining traditional financial 
    metrics with alternative data sources, we can provide more accurate credit assessments 
    for individuals who lack traditional credit histories.
    
    ### Key Features
    
    - **Alternative Data Integration**: Incorporates social media sentiment, Google Trends 
      data, and news sentiment analysis
    - **Machine Learning**: Uses Random Forest models for robust credit risk prediction
    - **Portfolio Analytics**: Comprehensive risk metrics including VaR and correlation analysis
    - **Real-time Processing**: Fast credit assessment with immediate results
    
    ### Technology Stack
    
    - **Machine Learning**: scikit-learn, Random Forest Classifier
    - **NLP**: TextBlob for sentiment analysis, TF-IDF vectorization
    - **Data Sources**: Google Trends (pytrends), News APIs, Social Media
    - **Frontend**: Streamlit for interactive dashboard
    - **Data Processing**: pandas, numpy
    - **Visualization**: Plotly, Matplotlib
    
    ### MSCFE Modules Demonstrated
    
    | Module | Topic | Implementation |
    |--------|-------|----------------|
    | M1 | Credit Risk | Default probability modeling, credit scoring |
    | M2 | Return & Volatility | Portfolio return analysis |
    | M3 | Portfolio Theory | Correlation analysis, diversification |
    | M4 | Alternative Data | TF-IDF, social media integration |
    | M5 | News Sentiment | NLP-based news analysis |
    | M7 | Ethics | Fair lending considerations |
    
    ### Author
    
    **Peterson Mutegi**  
    MSCFE Student | Financial Engineering & Alternative Data
    
    - [LinkedIn](https://www.linkedin.com/in/peterson-muriuki)
    - [GitHub](https://github.com/Peterson-Muriuki)
    
    ### Disclaimer
    
    This is a demonstration project for educational purposes. The credit scores and 
    recommendations provided are based on simulated data and should not be used for 
    actual lending decisions.
    """)
