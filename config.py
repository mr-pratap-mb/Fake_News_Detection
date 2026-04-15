import os

APP_NAME = "FakeShield"
TAGLINE = "Truth Powered by Evidence"
VERSION = "2.0.0"
TEAM = [
    {"name": "Pratap Bambadi", "roll": "2BU23CS100"},
    {"name": "Preetam J Hiremath", "roll": "2BU23CS104"}
]
COLLEGE = "Dept. of Computer Science & Engineering, SGBIT College"

NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")

MAX_INPUT_LENGTH = 5000
MAX_EVIDENCE_ARTICLES = 8
REQUEST_TIMEOUT = 10

RSS_FEEDS = {
    "BBC News": "http://feeds.bbci.co.uk/news/rss.xml",
    "Reuters": "https://feeds.reuters.com/reuters/topNews",
    "AP News": "https://rsshub.app/apnews/topics/apf-topnews",
    "Al Jazeera": "https://www.aljazeera.com/xml/rss/all.xml",
    "The Hindu": "https://www.thehindu.com/feeder/default.rss",
    "NDTV": "https://feeds.feedburner.com/ndtvnews-top-stories"
}

CREDIBLE_DOMAINS = [
    "bbc.com", "reuters.com", "apnews.com", "ndtv.com",
    "thehindu.com", "aljazeera.com", "theguardian.com",
    "nytimes.com", "washingtonpost.com", "hindustantimes.com"
]

SUSPICIOUS_DOMAINS = [
    "beforeitsnews.com", "infowars.com", "naturalnews.com",
    "worldnewsdailyreport.com", "empirenews.net"
]
