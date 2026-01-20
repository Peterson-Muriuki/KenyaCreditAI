"""
Utility functions for KenyaCredit AI
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re

def clean_text(text):
    """
    Clean text data for NLP processing
    
    Args:
        text (str): Raw text input
        
    Returns:
        str: Cleaned text
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove special characters but keep spaces
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text


def calculate_sharpe_ratio(returns, risk_free_rate=0.05):
    """
    Calculate Sharpe ratio for a return series
    
    Args:
        returns (pd.Series): Series of returns
        risk_free_rate (float): Annual risk-free rate (default 5%)
        
    Returns:
        float: Sharpe ratio
    """
    if len(returns) == 0 or returns.std() == 0:
        return 0.0
    
    excess_returns = returns.mean() - (risk_free_rate / 252)  # Daily risk-free rate
    return excess_returns / returns.std()


def calculate_var(returns, confidence_level=0.95):
    """
    Calculate Value at Risk using historical method
    
    Args:
        returns (pd.Series): Series of returns
        confidence_level (float): Confidence level (default 95%)
        
    Returns:
        float: VaR value
    """
    if len(returns) == 0:
        return 0.0
    
    return np.percentile(returns, (1 - confidence_level) * 100)


def create_sample_data(n_samples=100):
    """
    Create sample borrower data for demonstration
    
    Args:
        n_samples (int): Number of samples to generate
        
    Returns:
        pd.DataFrame: Sample dataset
    """
    np.random.seed(42)
    
    data = {
        'applicant_id': [f'APP{str(i).zfill(5)}' for i in range(1, n_samples + 1)],
        'income': np.random.normal(50000, 20000, n_samples).clip(15000, 200000),
        'existing_debt': np.random.normal(20000, 15000, n_samples).clip(0, 100000),
        'employment_months': np.random.randint(1, 120, n_samples),
        'age': np.random.randint(21, 65, n_samples),
        'social_sentiment': np.random.uniform(-1, 1, n_samples),
        'google_trends_score': np.random.uniform(0, 100, n_samples),
        'news_sentiment': np.random.uniform(-1, 1, n_samples),
        'default': np.random.choice([0, 1], n_samples, p=[0.85, 0.15])
    }
    
    df = pd.DataFrame(data)
    
    # Calculate debt-to-income ratio
    df['debt_to_income'] = df['existing_debt'] / df['income']
    
    return df


def format_currency(amount):
    """
    Format amount as Kenyan Shillings
    
    Args:
        amount (float): Amount to format
        
    Returns:
        str: Formatted string
    """
    return f"KES {amount:,.2f}"


def get_risk_category(score):
    """
    Categorize credit score into risk levels
    
    Args:
        score (float): Credit score (0-1000)
        
    Returns:
        str: Risk category
    """
    if score >= 750:
        return "Low Risk"
    elif score >= 600:
        return "Medium Risk"
    elif score >= 450:
        return "High Risk"
    else:
        return "Very High Risk"


def get_risk_color(category):
    """
    Get color for risk category visualization
    
    Args:
        category (str): Risk category
        
    Returns:
        str: Hex color code
    """
    colors = {
        "Low Risk": "#28a745",
        "Medium Risk": "#ffc107",
        "High Risk": "#fd7e14",
        "Very High Risk": "#dc3545"
    }
    return colors.get(category, "#6c757d")