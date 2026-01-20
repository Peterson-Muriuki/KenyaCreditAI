# KenyaCredit AI

**Alternative Credit Scoring for Financial Inclusion in Kenya**

## Overview
KenyaCredit AI is an advanced credit scoring system that combines traditional financial data with alternative data sources (social media sentiment, Google Trends, news analysis) to expand credit access for underbanked Kenyans.

## Features
- 🎯 Traditional + Alternative Data Credit Scoring
- 📊 Portfolio Risk Analytics (VaR, Correlation Analysis)
- 📰 Real-time News Sentiment Analysis
- 🔍 Google Trends Financial Stress Indicators
- 📈 Interactive Streamlit Dashboard

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