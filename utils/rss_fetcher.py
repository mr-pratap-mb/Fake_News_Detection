import requests
import feedparser
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import config
from utils.web_fetcher import get_domain

class RSSFetcher:
    """
    RSS Feed Fallback Fetcher for FakeShield v2.
    Utilizes concurrent requests and text-overlap scoring.
    """
    
    def __init__(self, feeds_dict):
        self.feeds = feeds_dict
        self.timeout = 8
        self._article_counts = {}

    def fetch_feed(self, source_name, url):
        """
        Fetches an individual RSS feed using requests (for strict timeout) 
        and parses it via feedparser. Limits output to 10 entries.
        """
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) FakeShield/2.0'
        }
        
        try:
            response = requests.get(url, headers=headers, timeout=self.timeout)
            response.raise_for_status()
            
            feed = feedparser.parse(response.content)
            
            if feed.bozo and hasattr(feed, 'bozo_exception'):
                # Not strictly a failure, but feed is malformed. If no entries, we return empty.
                if not getattr(feed, 'entries', []):
                    raise ValueError(f"Malformed feed: {feed.bozo_exception}")

            articles = []
            for entry in feed.entries[:10]:
                domain = get_domain(entry.get('link', '')) or get_domain(url)
                
                # feedparser falls back nicely, but let's grab summary as fallback for description
                description = entry.get('summary', entry.get('description', ''))
                # Clean up HTML tags potentially inside descriptions
                description_clean = re.sub(r'<[^>]+>', '', description)
                
                articles.append({
                    "title": entry.get('title', ''),
                    "description": description_clean,
                    "content": description_clean, # Using summary as fallback for content
                    "source_name": source_name,
                    "source_url": url,
                    "domain": domain, # Ecosystem adherence
                    "published_at": entry.get('published', ''),
                    "url": entry.get('link', ''),
                    "source_type": "rss" # Ecosystem adherence
                })
            
            self._article_counts[source_name] = len(articles)
            return articles
            
        except requests.RequestException as e:
            print(f"[RSS Fetcher] Network timeout/error for {source_name}: {e}")
            self._article_counts[source_name] = 0
            return []
        except Exception as e:
            print(f"[RSS Fetcher] Parsing failed for {source_name}: {e}")
            self._article_counts[source_name] = 0
            return []

    def fetch_all_feeds(self):
        """
        Fetches all feeds in parallel using ThreadPoolExecutor.
        Returns a flat list of all articles retrieved.
        """
        all_articles = []
        self._article_counts.clear()
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Map futures to source names
            future_to_source = {
                executor.submit(self.fetch_feed, name, url): name 
                for name, url in self.feeds.items()
            }
            
            for future in as_completed(future_to_source):
                try:
                    result = future.result()
                    all_articles.extend(result)
                except Exception as e:
                    source = future_to_source[future]
                    print(f"[RSS Fetcher] Unexpected error threading {source}: {e}")
                    
        return all_articles

    def search_feeds(self, query, max_results=8):
        """
        Scores retrieved parallel articles based on keyword overlap.
        Sorting by combined match score.
        """
        articles = self.fetch_all_feeds()
        
        if not query:
            # If no query, just return latest random assortment
            return articles[:max_results]
        
        # Tokenize query lightly to isolate matchable keywords
        query_words = [w.lower() for w in re.sub(r'[^a-zA-Z0-9\s]', '', str(query)).split() if len(w) > 2]
        
        scored_articles = []
        
        for article in articles:
            score = 0
            title_lower = article['title'].lower()
            desc_lower = article['description'].lower()
            
            for kw in query_words:
                if kw in title_lower:
                    score += 2
                if kw in desc_lower:
                    score += 1
            
            article['match_score'] = score
            scored_articles.append(article)
            
        # Filter purely unrelated items if query exists (score=0), fallback if needed
        # Or just sort by score descending.
        scored_articles.sort(key=lambda x: x.get('match_score', 0), reverse=True)
        scored_articles = [a for a in scored_articles if a.get('match_score', 0) > 0]
        
        # Return top matches, even if poorly scored.
        return scored_articles[:max_results]

    def get_feed_article_count(self):
        """ Returns dictionary map of source_name to items successfully extracted. """
        return self._article_counts


# --- Module-level backward compatibility bridges ---
# Maps to ecosystem requirements for evidence analyzer

_rss_fetcher = RSSFetcher(config.RSS_FEEDS)

def fetch_from_rss(keywords):
    """
    Adapter wrapper processing the list of keywords into a string 
    query and passing to the search_feeds engine.
    """
    query_str = " ".join(keywords) if keywords else ""
    return _rss_fetcher.search_feeds(query_str, max_results=config.MAX_EVIDENCE_ARTICLES)
