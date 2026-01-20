"""
Credit Scoring Model for KenyaCredit AI
Combines traditional and alternative data sources
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

class CreditScoringModel:
    """
    Credit scoring model combining traditional and alternative data
    """
    
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_trained = False
        
    def engineer_features(self, df):
        """
        Create features from raw data
        
        Args:
            df (pd.DataFrame): Raw data
            
        Returns:
            pd.DataFrame: Engineered features
        """
        features = df.copy()
        
        # Traditional credit features
        features['debt_to_income'] = features['existing_debt'] / features['income']
        features['income_stability'] = features['employment_months'] / 12  # Years employed
        
        # Alternative data features (already in dataset)
        # - social_sentiment: Sentiment from social media (-1 to 1)
        # - google_trends_score: Financial stress indicator (0-100)
        # - news_sentiment: Economic news sentiment (-1 to 1)
        
        # Interaction features
        features['alt_data_composite'] = (
            features['social_sentiment'] * 0.4 +
            (100 - features['google_trends_score']) / 100 * 0.3 +  # Invert trends (low search = good)
            features['news_sentiment'] * 0.3
        )
        
        # Risk indicators
        features['high_debt_flag'] = (features['debt_to_income'] > 0.5).astype(int)
        features['young_borrower'] = (features['age'] < 25).astype(int)
        
        return features
    
    def train(self, df, target_col='default'):
        """
        Train the credit scoring model
        
        Args:
            df (pd.DataFrame): Training data with target
            target_col (str): Name of target column
            
        Returns:
            dict: Training metrics
        """
        # Engineer features
        features_df = self.engineer_features(df)
        
        # Select features for model
        feature_cols = [
            'income', 'existing_debt', 'employment_months', 'age',
            'debt_to_income', 'income_stability',
            'social_sentiment', 'google_trends_score', 'news_sentiment',
            'alt_data_composite', 'high_debt_flag', 'young_borrower'
        ]
        
        self.feature_names = feature_cols
        
        X = features_df[feature_cols]
        y = df[target_col]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        # Evaluate
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return {
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'feature_importance': feature_importance
        }
    
    def predict_score(self, applicant_data):
        """
        Predict credit score for an applicant
        
        Args:
            applicant_data (dict): Applicant information
            
        Returns:
            dict: Credit score and details
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        # Convert to DataFrame
        df = pd.DataFrame([applicant_data])
        
        # Engineer features
        features_df = self.engineer_features(df)
        
        # Select and scale features
        X = features_df[self.feature_names]
        X_scaled = self.scaler.transform(X)
        
        # Predict probability of default
        default_prob = self.model.predict_proba(X_scaled)[0, 1]
        
        # Convert to credit score (300-850 scale, inverted from default prob)
        credit_score = int(850 - (default_prob * 550))
        
        # Get risk category
        if credit_score >= 750:
            risk_category = "Low Risk"
        elif credit_score >= 600:
            risk_category = "Medium Risk"
        elif credit_score >= 450:
            risk_category = "High Risk"
        else:
            risk_category = "Very High Risk"
        
        # Calculate recommended interest rate (base rate + risk premium)
        base_rate = 10.0  # Base rate 10%
        risk_premium = default_prob * 20  # Up to 20% additional
        recommended_rate = base_rate + risk_premium
        
        return {
            'credit_score': credit_score,
            'default_probability': float(default_prob),
            'risk_category': risk_category,
            'recommended_interest_rate': float(recommended_rate),
            'features': features_df[self.feature_names].iloc[0].to_dict()
        }
    
    def save_model(self, path='models/credit_model.pkl'):
        """Save trained model to disk"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained
        }, path)
    
    def load_model(self, path='models/credit_model.pkl'):
        """Load trained model from disk"""
        if os.path.exists(path):
            saved = joblib.load(path)
            self.model = saved['model']
            self.scaler = saved['scaler']
            self.feature_names = saved['feature_names']
            self.is_trained = saved['is_trained']
            return True
        return False