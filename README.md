# KenyaCredit AI

**Alternative Credit Scoring for Financial Inclusion in Kenya**

## Overview
KenyaCredit AI is an advanced credit scoring system that combines traditional financial data with alternative data sources (social media sentiment, Google Trends, news analysis) to expand credit access for underbanked Kenyans.
[![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.1+-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.26+-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![Plotly](https://img.shields.io/badge/Plotly-5.18+-3F4F75?logo=plotly&logoColor=white)](https://plotly.com/)
[![NLTK](https://img.shields.io/badge/NLTK-3.8+-154f3c?logo=python&logoColor=white)](https://www.nltk.org/)
[![TextBlob](https://img.shields.io/badge/TextBlob-NLP-9b59b6)](https://textblob.readthedocs.io/)
[![pytrends](https://img.shields.io/badge/pytrends-Google%20Trends-4285F4?logo=google&logoColor=white)](https://pypi.org/project/pytrends/)
[![Beautiful Soup](https://img.shields.io/badge/BeautifulSoup-Web%20Scraping-59b300)](https://www.crummy.com/software/BeautifulSoup/)
[![TF-IDF](https://img.shields.io/badge/TF--IDF-Text%20Vectorization-8e44ad)](https://scikit-learn.org/stable/modules/feature_extraction.html#tfidf-term-weighting)
[![Sentiment Analysis](https://img.shields.io/badge/Sentiment-Analysis-e74c3c)](https://en.wikipedia.org/wiki/Sentiment_analysis)
[![Random Forest](https://img.shields.io/badge/Random%20Forest-ML%20Model-27ae60)](https://scikit-learn.org/stable/modules/ensemble.html#random-forests)
[![VaR](https://img.shields.io/badge/VaR-Risk%20Metric-c0392b)](https://en.wikipedia.org/wiki/Value_at_risk)
[![Correlation](https://img.shields.io/badge/Correlation-Analysis-16a085)](https://en.wikipedia.org/wiki/Correlation)
[![Credit Scoring](https://img.shields.io/badge/Credit-Scoring-2c3e50)](https://en.wikipedia.org/wiki/Credit_score)
[![Portfolio Theory](https://img.shields.io/badge/Portfolio-Theory-34495e)](https://en.wikipedia.org/wiki/Modern_portfolio_theory)
[![Alternative Data](https://img.shields.io/badge/Alternative-Data-f39c12)](https://en.wikipedia.org/wiki/Alternative_data_(finance))


## Features
- Traditional + Alternative Data Credit Scoring
- Portfolio Risk Analytics (VaR, Correlation Analysis)
- Real-time News Sentiment Analysis
- Google Trends Financial Stress Indicators
- Interactive Streamlit Dashboard

## Tech Stack
- **Backend:** Python, scikit-learn, pandas, numpy
- **NLP:** NLTK, TextBlob, TF-IDF
- **Data Sources:** Google Trends (pytrends), News APIs, Social Media
- **Frontend:** Streamlit
- **Deployment:** Streamlit Cloud

## Installation
```bash
# Clone repository
git clone https://github.com/Peterson-Muriuki/KenyaCreditAI.git
cd KenyaCreditAI

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

## Usage
```bash
streamlit run src/app.py
```

## Project Structure
```
KenyaCreditAI/
├── data/               # Data files (not tracked in git)
├── models/             # Trained models
├── notebooks/          # Jupyter notebooks for exploration
├── src/                # Source code
│   ├── app.py         # Main Streamlit app
│   ├── credit_model.py # Credit scoring logic
│   ├── alt_data.py    # Alternative data collection
│   └── utils.py       # Helper functions
├── tests/              # Unit tests
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Modules Demonstrated
This project showcases skills from:
- M1: Credit Risk and Financing
- M2: Return and Volatility
- M3: Correlation & Portfolio Theory
- M4: Alternative Data (TF-IDF, Sentiment Analysis)
- M5: News Data and Sentiment Analysis
- M7: Ethics in Financial Engineering

## Author
**Peterson Mutegi**  
MSCFE Student | Financial Engineering & Alternative Data  
[LinkedIn](https://www.linkedin.com/in/peterson-muriuki) | [GitHub](https://github.com/Peterson-Muriuki)

## License
MIT License
