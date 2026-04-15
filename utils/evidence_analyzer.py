import random
import re
import nltk
from utils.text_processor import TextProcessor

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)

class EvidenceAnalyzer:
    """
    Core evidence analysis engine for FakeShield v2.
    Takes user claims and scraped evidence arrays to generate structured verdicts.
    """
    
    def __init__(self):
        # Instantiate TextProcessor internally for text similarity operations
        self.processor = TextProcessor()
        
        # Word lists used to detect contradiction within text payloads
        self.contradiction_words = [
            "not true", "false", "denied", "debunked", "no evidence",
            "fake", "hoax", "mislead", "untrue", "debunk", "refuted", "inaccurate"
        ]

    def extract_search_query(self, input_text, input_url_content=None):
        """
        Dynamically extract optimized search query keywords.
        Returns a space-separated query string capped at max 8 words.
        """
        target_text = input_url_content if input_url_content else input_text
        keywords_str = self.processor.extract_claim_keywords(target_text)
        
        # Split string, limit to 8 keywords, rejoin.
        words = keywords_str.split()[:8]
        return " ".join(words)

    def score_article_relevance(self, claim_text, article):
        """
        Computes TF-IDF text similarity between the original claim 
        and the article's title + description payload. Returns float 0.0-1.0
        """
        title = article.get('title', '')
        desc = article.get('description', '')
        
        # Combine contextual features
        combined_article_text = f"{title} {desc}".lower()
        claim_lower = claim_text.lower()
        
        if not combined_article_text.strip():
            return 0.0
            
        # 1. Numerical Consistency Check (Strict Constraint logic)
        numbers_in_claim = re.findall(r'\b\d+\b', claim_lower)
        if numbers_in_claim:
            # If numerical bounds exist in the prompt, they must exist in the corroborating evidence natively
            if not any(num in combined_article_text for num in numbers_in_claim):
                 return 0.0
                 
        # 2. Contextual Pruning (NLTK POS Tagging)
        claim_tokens = nltk.word_tokenize(claim_lower)
        claim_tags = nltk.pos_tag(claim_tokens)
        
        # Extract primarily declarative nouns handling the Core Subject
        primary_entities = set(word for word, tag in claim_tags if tag.startswith('NN') and len(word) > 2)
        
        if primary_entities:
            matches = sum(1 for entity in primary_entities if entity in combined_article_text)
            # If the evidence text doesn't possess at least 25% noun-overlap, strictly bleed out.
            if len(primary_entities) > 1 and (matches / len(primary_entities) < 0.25):
                return 0.0
            
        similarity = self.processor.compute_text_similarity(claim_text, combined_article_text)
        return similarity

    def check_source_credibility(self, article, credible_domains, suspicious_domains):
        """
        Checks the raw URL source domain against application configuration lists.
        Outputs bounded score limits.
        """
        domain = article.get('domain', '').lower()
        if not domain:
            domain = article.get('source_url', '').lower()

        # Iterate over configurations to locate matches
        for c_dom in credible_domains:
            if c_dom in domain:
                # Provide randomized distribution within the bounded class
                return {"score": random.randint(80, 100), "label": "Credible", "domain": domain}
                
        for s_dom in suspicious_domains:
            if s_dom in domain:
                return {"score": random.randint(0, 20), "label": "Suspicious", "domain": domain}
                
        # Unknown domain fallback
        return {"score": random.randint(40, 60), "label": "Unknown", "domain": domain}

    def analyze_evidence_support(self, claim_text, articles):
        """
        Iterates over the evidence array computing deep metric scores 
        including relevance, credibility, and mock sentiment alignments.
        Returns list of new annotated article dicts.
        """
        from config import CREDIBLE_DOMAINS, SUSPICIOUS_DOMAINS
        
        scored_articles = []
        for article in articles:
            relevance = self.score_article_relevance(claim_text, article)
            credibility_data = self.check_source_credibility(article, CREDIBLE_DOMAINS, SUSPICIOUS_DOMAINS)
            
            combined_text = (article.get('title', '') + " " + article.get('description', '')).lower()
            
            # Simple sentiment alignment rule-checking (Support vs Contradict)
            # Checks for contradiction keywords
            sentiment = "Supports"
            for cw in self.contradiction_words:
                if cw in combined_text:
                    sentiment = "Contradicts"
                    break
                    
            annotated_article = article.copy()
            annotated_article.update({
                "relevance_score": relevance,
                "credibility_score": credibility_data["score"],
                "credibility_label": credibility_data["label"],
                "sentiment_alignment": sentiment
            })
            scored_articles.append(annotated_article)
            
        return scored_articles

    def compute_corroboration_score(self, scored_articles):
        """
        Assess strength of story corroboration across highly credible sources.
        """
        if not scored_articles:
            return 0.0, "None"
            
        credible_coverage = 0
        total_relevance = 0.0
        
        for art in scored_articles:
            # We strictly evaluate highly correlated, credible articles
            if art.get("credibility_label") == "Credible" and art.get("relevance_score", 0.0) > 0.15:
                credible_coverage += 1
            total_relevance += art.get("relevance_score", 0.0)
            
        # Calculation threshold mappings
        if credible_coverage >= 3:
            score = 0.95
            label = "Strong"
        elif credible_coverage >= 1 or total_relevance > 0.8:
            score = 0.65
            label = "Moderate"
        elif total_relevance > 0.3:
            score = 0.35
            label = "Weak"
        else:
            score = 0.0
            label = "None"
            
        return score, label

    def detect_contradiction(self, claim_text, articles):
        """
        Strict pass looking specifically for structural negation of the claim.
        Utilizes predefined negation pattern string blocks.
        """
        contradicted = False
        contradicting_sources = []
        
        for art in articles:
            combined_text = (art.get('title', '') + " " + art.get('description', '')).lower()
            
            # Loop against negation words
            for cw in self.contradiction_words:
                if cw in combined_text:
                    contradicted = True
                    contradicting_sources.append(art)
                    break # Next article
                    
        return {
            "contradicted": contradicted,
            "contradicting_sources": contradicting_sources
        }

    def generate_evidence_summary(self, claim_text, articles):
        """
        Master orchestration routine processing the evidence and returning 
        the final mathematical JSON verdict.
        """
        # 1. Base Score Appends
        scored_articles = self.analyze_evidence_support(claim_text, articles)
        
        # Sort descending by relevance
        scored_articles.sort(key=lambda x: x.get('relevance_score', 0.0), reverse=True)
        top_relevant = scored_articles[:5]
        
        # 2. Corroboration Engine
        corrob_score, corrob_label = self.compute_corroboration_score(scored_articles)
        
        # 3. Contradiction Evaluator
        contradiction_data = self.detect_contradiction(claim_text, scored_articles)
        
        # 4. Global Credibility Tracker
        avg_credibility = 0.0
        top_credible_sources = []
        if scored_articles:
            avg_credibility = sum([a['credibility_score'] for a in scored_articles]) / len(scored_articles)
            top_credible_sources = [a.get('domain') for a in scored_articles if a.get('credibility_label') == "Credible"]
            
        # 5. Verdict Tree Path
        if contradiction_data["contradicted"] and avg_credibility > 40:
            verdict = "CONTRADICTED"
        elif corrob_score > 0.6:
            verdict = "SUPPORTED"
        elif corrob_label == "Weak":
            verdict = "UNVERIFIED"
        else:
            verdict = "INSUFFICIENT"
            
        return {
             "total_articles_found": len(articles),
             "relevant_articles": top_relevant,
             "corroboration_score": round(corrob_score, 2),
             "corroboration_label": corrob_label,
             "contradiction_detected": contradiction_data["contradicted"],
             "contradicting_sources": contradiction_data["contradicting_sources"],
             "average_source_credibility": round(avg_credibility, 2),
             "top_credible_sources": list(set(top_credible_sources)),
             "evidence_verdict": verdict
        }


# --- Module-level backward compatibility bridges ---
# Predictor relies on primitive `gather_evidence()` from earlier phase.

from utils.news_api import fetch_from_newsapi
from utils.rss_fetcher import fetch_from_rss

def gather_evidence(claim):
    """
    Legacy wrapper utilizing NewsAPI and RSS fetching routines.
    To be upgraded entirely natively downstream in predictor.py eventually!
    """
    analyzer = EvidenceAnalyzer()
    
    # Dynamic Query Generation
    query_str = analyzer.extract_search_query(claim)
    keywords_list = query_str.split()
    
    # 1. NewsAPI fetch
    evidence = fetch_from_newsapi(keywords_list)
    
    if not evidence:
         # 2. RSA Fallback
         evidence = fetch_from_rss(keywords_list)
         
    return evidence
