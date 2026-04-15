# Fake_News_Detection
A hybrid Fake News Detection system that combines Machine Learning (Scikit-Learn) with Real-Time Web Evidence retrieval. It cross-references claims against live RSS feeds and News APIs to provide a "Truth Score" based on NLP analysis and source credibility.

# 🛡️ FakeShield v2 

**Truth Powered by Live Evidence** Developed by: Pratap Bambadi & Preetam J Hiremath  
*Dept. of Computer Science & Engineering, SGBIT College*

## 🚀 Overview
FakeShield v2 is not just a static classifier; it is a **dynamic verification engine**. Unlike traditional models that rely on outdated datasets, FakeShield fetches live evidence from the web to verify if a claim is supported or contradicted by credible news agencies.

## 🛠️ Features
- **Live Evidence Fetching:** Real-time search via NewsAPI and RSS fallbacks.
- **ML Probability Matrix:** Dual-layer classification using TF-IDF and Logistic Regression.
- **Pattern Detector:** Scans for linguistic triggers like clickbait, conspiracy jargon, and sensationalism.
- **Source Credibility Meter:** Ranks sources based on domain authority and historical reliability.

## 💻 Tech Stack
- **Backend:** Flask (Python)
- **Frontend:** HTML5, CSS3 (Modern Dark UI), JS
- **AI/NLP:** Scikit-Learn, NLTK, BeautifulSoup4
- **Data:** NewsAPI, Feedparser (RSS)
-
