import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import re

class WebFetcher:
    """
    URL article scraper module for FakeShield v2.
    Fetches, slices, and categorizes textual webpage content cleanly.
    """
    
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 FakeShield/2.0'
        }
        self.timeout = 10

    def is_valid_url(self, url):
        """
        Return True if URL starts with http/https and has valid domain format.
        """
        if not url or not isinstance(url, str):
            return False
            
        try:
            result = urlparse(url)
            return all([result.scheme in ['http', 'https'], result.netloc])
        except Exception:
            return False

    def extract_domain(self, url):
        """
        Return clean domain from URL (e.g. "bbc.com" from "https://www.bbc.com/news/...")
        """
        if not self.is_valid_url(url):
            return "unknown"
            
        try:
            domain = urlparse(url).netloc.lower()
            if domain.startswith('www.'):
                domain = domain[4:]
            return domain
        except Exception:
            return "unknown"

    def check_domain_credibility(self, url, credible_list, suspicious_list):
        """
        Compare domain against credible_list and suspicious_list from config.
        Return: "credible" / "suspicious" / "unknown"
        """
        domain = self.extract_domain(url)
        if domain == "unknown":
            return "unknown"
            
        for c in credible_list:
            if c in domain:
                return "credible"
                
        for s in suspicious_list:
            if s in domain:
                return "suspicious"
                
        return "unknown"

    def fetch_url_content(self, url):
        """
        GET request returning dictionary with titles, metadata, and core paragraphs parsed.
        """
        base_error_dict = {
            "title": "",
            "text": "",
            "description": "",
            "published_date": "",
            "url": url,
            "domain": "unknown",
            "fetch_success": False
        }
        
        if not self.is_valid_url(url):
            base_error_dict["error"] = "Invalid URL format provided."
            return base_error_dict
            
        domain = self.extract_domain(url)
        base_error_dict["domain"] = domain
            
        try:
            response = requests.get(url, headers=self.headers, timeout=self.timeout)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract Title: Try h1 first, fallback to title tag
            title_tag = soup.find('h1')
            if title_tag:
                title = title_tag.get_text(separator=' ', strip=True)
            else:
                title_tag = soup.find('title')
                title = title_tag.get_text(separator=' ', strip=True) if title_tag else ""
                
            # Extract Main Text: All <p> tags combined
            paragraphs = soup.find_all('p')
            text_blocks = [p.get_text(separator=' ', strip=True) for p in paragraphs if p.get_text(strip=True)]
            full_text = " ".join(text_blocks)
            
            # First 2000 chars as requested
            bounded_text = full_text[:2000] if full_text else ""
            
            # Extract Meta Description
            desc_tag = soup.find('meta', attrs={'name': 'description'}) or \
                       soup.find('meta', attrs={'property': 'og:description'})
            description = desc_tag['content'].strip() if desc_tag and desc_tag.has_attr('content') else ""
            
            # Extract Published Date
            pub_date = ""
            time_tag = soup.find('time')
            if time_tag and time_tag.has_attr('datetime'):
                pub_date = time_tag['datetime']
            else:
                meta_date = soup.find('meta', attrs={'property': 'article:published_time'})
                if meta_date and meta_date.has_attr('content'):
                    pub_date = meta_date['content']
            
            return {
                "title": title,
                "text": bounded_text,
                "description": description,
                "published_date": pub_date,
                "url": url,
                "domain": domain,
                "fetch_success": True
            }
            
        except requests.RequestException as e:
            base_error_dict["error"] = f"Network Error: {e}"
            return base_error_dict
        except Exception as e:
            base_error_dict["error"] = f"Extraction Error: {e}"
            return base_error_dict

    def fetch_multiple_urls(self, url_list):
        """
        Fetch up to 5 URLs in parallel using ThreadPoolExecutor.
        """
        if not url_list:
            return []
            
        # Ensure max 5 URLs
        limited_urls = url_list[:5]
        results = []
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_url = {executor.submit(self.fetch_url_content, url): url for url in limited_urls}
            for future in as_completed(future_to_url):
                try:
                    res = future.result()
                    results.append(res)
                except Exception as e:
                    url = future_to_url[future]
                    print(f"Parallel fetch generic failure on {url}: {e}")
                    
        return results

# --- Module-level backward compatibility bridges ---
# Provides compatibility mappings for `news_api.py`, `rss_fetcher.py` and `app.py` ecosystem logic.

_web_fetcher = WebFetcher()

def get_domain(url):
    """ Backwards compatibility for previously existing simple domain extractor routine. """
    return _web_fetcher.extract_domain(url)

def fetch_url_content(url):
    """ Backwards compatibility mapping for app.py calling fetch_url_content top-level. """
    result_dict = _web_fetcher.fetch_url_content(url)
    
    # In earlier implementations, application checked .get('success') instead of fetch_success
    result_dict['success'] = result_dict.get('fetch_success', False)
    return result_dict
