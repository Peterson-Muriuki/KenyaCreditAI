"""
Alternative Data Collection Module
Handles Google Trends, news sentiment, and social media analysis
"""

import pandas as pd
import numpy as np
from textblob import TextBlob
from pytrends.request import TrendReq
import requests
from bs4 import BeautifulSoup
import time
from datetime import datetime, timedelta

class AlternativeDataCollector:
    """
    Collects and processes alternative data sources
    """
    
    def __init__(self):
        self.pytrends = TrendReq(hl='en-US', tz=180)  # Kenyan timezone
        
    def get_google_trends_score(self, keywords, timeframe='today 3-m'):
        """
        Get Google Trends data for financial stress keywords
        
        Args:
            keywords (list): List of search terms
            timeframe (str): Time range for trends
            
        Returns:
            float: Composite stress score (0-100)
        """
        try:
            self.pytrends.build_payload(
                keywords,
                cat=0,
                timeframe=timeframe,
                geo='KE'  # Kenya
            )
            
            trends_df = self.pytrends.interest_over_time()
            
            if trends_df.empty:
                return 50.0  # Neutral score if no data
            
            # Calculate average of latest values
            latest_scores = trends_df[keywords].iloc[-1].values
            composite_score = np.mean(latest_scores)
            
            return float(composite_score)
            
        except Exception as e:
            print(f"Error fetching Google Trends: {e}")
            return 50.0  # Return neutral score on error
    
    def analyze_text_sentiment(self, text):
        """
        Analyze sentiment of text using TextBlob
        
        Args:
            text (str): Text to analyze
            
        Returns:
            float: Sentiment score (-1 to 1)
        """
        try:
            blob = TextBlob(str(text))
            return blob.sentiment.polarity
        except:
            return 0.0
    
    def scrape_news_headlines(self, source_url=None):
        """
        Scrape recent financial news headlines (simulated for demo)
        
        Args:
            source_url (str): URL of news source
            
        Returns:
            list: List of headline dictionaries
        """
        # For demo purposes, return sample headlines
        # In production, this would scrape real news sites
        
        sample_headlines = [
            {"title": "Kenya's economy shows strong growth in Q4", 
             "date": datetime.now() - timedelta(days=1)},
            {"title": "Central Bank maintains interest rates", 
             "date": datetime.now() - timedelta(days=2)},
            {"title": "Digital lending platforms see increased adoption", 
             "date": datetime.now() - timedelta(days=3)},
            {"title": "Inflation remains stable at 5.2%", 
             "date": datetime.now() - timedelta(days=4)},
            {"title": "IMF approves new funding for Kenya", 
             "date": datetime.now() - timedelta(days=5)},
        ]
        
        return sample_headlines
    
    def get_news_sentiment(self):
        """
        Get aggregated news sentiment score
        
        Returns:
            float: Average sentiment (-1 to 1)
        """
        headlines = self.scrape_news_headlines()
        
        sentiments = [
            self.analyze_text_sentiment(h['title']) 
            for h in headlines
        ]
        
        if not sentiments:
            return 0.0
        
        return np.mean(sentiments)
    
    def get_financial_stress_indicator(self):
        """
        Combined financial stress indicator from multiple sources
        
        Returns:
            dict: Stress indicators
        """
        # Keywords indicating financial stress
        stress_keywords = ['emergency loan', 'debt help', 'financial crisis']
        
        # Get trends (this would be real in production)
        # For demo, we'll simulate
        trends_score = np.random.uniform(30, 70)  # Simulate trends
        
        # Get news sentiment
        news_sentiment = self.get_news_sentiment()
        
        # Social media sentiment (simulated for demo)
        social_sentiment = np.random.uniform(-0.3, 0.5)
        
        # Composite stress score
        # High trends = High stress (bad)
        # Negative news = High stress (bad)
        # Negative social = High stress (bad)
        
        stress_score = (
            trends_score * 0.4 +  # Higher trends = more stress
            (1 - (news_sentiment + 1) / 2) * 100 * 0.3 +  # Convert sentiment to 0-100
            (1 - (social_sentiment + 1) / 2) * 100 * 0.3
        )
        
        return {
            'google_trends_score': trends_score,
            'news_sentiment': news_sentiment,
            'social_sentiment': social_sentiment,
            'composite_stress_score': stress_score
        }
    
    def enrich_applicant_data(self, applicant_data):
        """
        Add alternative data to applicant information
        
        Args:
            applicant_data (dict): Basic applicant info
            
        Returns:
            dict: Enriched applicant data
        """
        # Get alternative data
        alt_data = self.get_financial_stress_indicator()
        
        # Add to applicant data
        enriched = applicant_data.copy()
        enriched['google_trends_score'] = alt_data['google_trends_score']
        enriched['news_sentiment'] = alt_data['news_sentiment']
        enriched['social_sentiment'] = alt_data['social_sentiment']
        
        return enriched