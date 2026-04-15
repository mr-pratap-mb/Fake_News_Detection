import requests
from bs4 import BeautifulSoup
import config
from utils.web_fetcher import get_domain

class NewsAPIFetcher:
    """
    NewsAPI.org integration module for FakeShield v2.
    """
    
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://newsapi.org/v2/"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) FakeShield/2.0'
        }
        self.timeout = 10
        
    def search_everything(self, query, max_results=8):
        """
        Call /everything endpoint using the given query keywords.
        Returns a formatted list of article dictionaries.
        """
        if not self.api_key or self.api_key == "your_newsapi_key_here":
            print("NewsAPI error: API Key not set.")
            return []
            
        url = f"{self.base_url}everything"
        # API requires URL encoding implicitly handled by requests.get params
        params = {
            'q': query if query else "breaking news",
            'language': 'en',
            'sortBy': 'relevancy',
            'pageSize': max_results,
            'apiKey': self.api_key
        }
        
        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=self.timeout)
            if response.status_code == 200:
                data = response.json()
                return self._format_articles(data.get('articles', []))
            else:
                print(f"NewsAPI /everything Error: {response.status_code} - {response.text}")
                return []
        except requests.RequestException as e:
            print(f"NewsAPI request failed: {e}")
            return []
            
    def search_top_headlines(self, query, max_results=5):
        """
        Call /top-headlines endpoint. 
        Note: NewsAPI top-headlines limits keyword query matching strictness.
        """
        if not self.api_key or self.api_key == "your_newsapi_key_here":
            return []
            
        url = f"{self.base_url}top-headlines"
        params = {
            'q': query if query else "breaking news",
            'language': 'en',
            'pageSize': max_results,
            'apiKey': self.api_key
        }
        
        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=self.timeout)
            if response.status_code == 200:
                data = response.json()
                return self._format_articles(data.get('articles', []))
            else:
                print(f"NewsAPI /top-headlines Error: {response.status_code} - {response.text}")
                return []
        except requests.RequestException as e:
            print(f"NewsAPI request failed: {e}")
            return []

    def fetch_article_content(self, url):
        """
        Use requests and BeautifulSoup to fetch raw article text directly 
        from the news source's URL. Returns up to 1000 characters.
        """
        if not url:
            return ""
            
        try:
            response = requests.get(url, headers=self.headers, timeout=self.timeout)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'lxml')
            
            # Extract only <p> tags
            paragraphs = soup.find_all('p')
            text_blocks = [p.get_text(separator=' ', strip=True) for p in paragraphs if p.get_text(strip=True)]
            
            full_text = " ".join(text_blocks)
            
            if not full_text:
                return ""
                
            return full_text[:1000]
            
        except Exception as e:
            print(f"Article fetch failed for {url}: {e}")
            return ""

    def is_api_available(self):
        """ Test API key with a 1-result dummy request """
        if not self.api_key or self.api_key == "your_newsapi_key_here":
            return False
            
        test_url = f"{self.base_url}top-headlines?language=en&pageSize=1&apiKey={self.api_key}"
        try:
            response = requests.get(test_url, headers=self.headers, timeout=self.timeout)
            return response.status_code == 200
        except:
            return False

    def get_combined_results(self, query, max_results=8):
        """
        Tries /everything first. If insufficient, falls back to /top-headlines.
        Deduplicates urls across the array.
        """
        articles = self.search_everything(query, max_results=max_results)
        
        if not articles or len(articles) < 3:
            print("NewsAPI /everything returned low results. Falling back to /top-headlines...")
            fallback_articles = self.search_top_headlines(query, max_results=max_results)
            articles.extend(fallback_articles)
            
        # Deduplicate using URLs
        seen_urls = set()
        deduped = []
        for a in articles:
            if a['url'] not in seen_urls:
                seen_urls.add(a['url'])
                deduped.append(a)
                
        return deduped[:max_results]

    def _format_articles(self, items):
        """ Helper mapping function across the API structures. """
        formatted = []
        for item in items:
            source_url = item.get('url', '')
            domain = item.get('source', {}).get('name', '').lower() or get_domain(source_url)
            
            formatted.append({
                "title": item.get('title', ''),
                "description": item.get('description', ''),
                "content": item.get('content', ''),
                "source_name": item.get('source', {}).get('name', ''),
                "source_url": domain,
                "domain": domain, # Appended to adhere to existing ecosystem bindings
                "published_at": item.get('publishedAt', ''),
                "url": source_url,
                "source_type": "newsapi"  # Appended for ecosystem requirements
            })
        return formatted


# --- Module-level backward compatibility bridges ---
# Allows existing files (e.g. evidence_analyzer.py) to transparently utilize NewsAPIFetcher

_fetcher = NewsAPIFetcher(config.NEWSAPI_KEY)

def fetch_from_newsapi(keywords):
    """ Legacy wrapper mapping list of keywords to the new fetcher mechanics """
    query = " OR ".join(keywords) if keywords else ""
    return _fetcher.get_combined_results(query, max_results=config.MAX_EVIDENCE_ARTICLES)
