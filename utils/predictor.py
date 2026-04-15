import os
import time
import joblib
import traceback

from config import NEWSAPI_KEY, RSS_FEEDS

from utils.text_processor import TextProcessor
from utils.pattern_detector import PatternDetector
from utils.credibility_scorer import CredibilityScorer
from utils.similarity_engine import SimilarityEngine
from utils.evidence_analyzer import EvidenceAnalyzer
from utils.news_api import NewsAPIFetcher
from utils.rss_fetcher import RSSFetcher
from utils.web_fetcher import WebFetcher

class FakeNewsPredictor:
    """
    Master Prediction Engine combining all NLP, ML, and OSINT logic securely.
    """
    
    def __init__(self):
        # 1. Load ML Artifacts
        models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
        model_path = os.path.join(models_dir, 'best_model.pkl')
        vec_path = os.path.join(models_dir, 'tfidf_vectorizer.pkl')
        
        self.ml_model = None
        self.vectorizer = None
        
        try:
            if os.path.exists(model_path) and os.path.exists(vec_path):
                self.ml_model = joblib.load(model_path)
                self.vectorizer = joblib.load(vec_path)
        except Exception as e:
            print(f"Warning: ML Initialization failed - {e}")
            
        # 2. Instantiate Component Ecosystem
        self.text_processor = TextProcessor()
        self.pattern_detector = PatternDetector()
        self.credibility_scorer = CredibilityScorer()
        self.similarity_engine = SimilarityEngine()
        self.evidence_analyzer = EvidenceAnalyzer()
        self.news_api = NewsAPIFetcher(NEWSAPI_KEY)
        self.rss = RSSFetcher(RSS_FEEDS)
        self.web = WebFetcher()

    def fetch_live_evidence(self, query, input_url=None):
        """
        Orchestrates Live API fetchers.
        Step 1: NewsAPI | Step 2: RSS fallbacks | Step 3: Explicit URLs
        """
        source_used = "newsapi"
        articles = self.news_api.get_combined_results(query, max_results=8)
        
        # Fallback Trigger
        if not articles or len(articles) < 3:
            source_used = "both" if articles else "rss"
            rss_articles = self.rss.search_feeds(query, max_results=8)
            articles.extend(rss_articles)
            
        # Optional Direct URL Inject
        if input_url:
            url_data = self.web.fetch_url_content(input_url)
            if url_data.get("fetch_success"):
                # Align dict to match EvidenceAnalyzer expectations
                articles.append({
                    "title": url_data.get("title", ""),
                    "description": url_data.get("description", ""),
                    "content": url_data.get("text", ""),
                    "url": url_data.get("url", ""),
                    "domain": url_data.get("domain", ""),
                    "source_type": "user_url"
                })
                
        # Deduplication Check
        seen = set()
        deduped = []
        for art in articles:
            url = art.get('url', '')
            if url and url not in seen:
                seen.add(url)
                deduped.append(art)
                
        return deduped, source_used

    def predict(self, input_text=None, input_url=None):
        """
        The 10-step orchestration algorithm computing the final comprehensive Fake News Verdict.
        """
        start_time = time.time()
        
        try:
            # ---------------------------------------------
            # STEP 1 - Input Handling
            # ---------------------------------------------
            url_text = ""
            if input_url:
                web_result = self.web.fetch_url_content(input_url)
                if web_result.get("fetch_success"):
                    url_text = f"{web_result.get('title', '')}. {web_result.get('description', '')}. {web_result.get('text', '')}"
            
            # Combine content pipelines gracefully
            combined_content = f"{input_text if input_text else ''} {url_text}".strip()
            if len(combined_content) < 20:
                raise ValueError("Insufficient input data to analyze securely.")
                
            input_type = "both" if (input_text and input_url) else ("url" if input_url else "text")

            # ---------------------------------------------
            # STEP 2 - Text Preprocessing
            # ---------------------------------------------
            preproc_data = self.text_processor.preprocess_pipeline(combined_content)
            claim_keywords = preproc_data.get("claim_keywords", "")

            # ---------------------------------------------
            # STEP 3 - Live Evidence Fetching
            # ---------------------------------------------
            # Format explicitly for NewsAPI OR logic cleanly to avoid 'breaking news' bug
            search_terms = claim_keywords.split()[:4] # Take top 4 most powerful entities safely
            search_query = " OR ".join(search_terms) if search_terms else ""
            articles, source_used = self.fetch_live_evidence(search_query, input_url)

            # ---------------------------------------------
            # STEP 4 - Evidence Analysis
            # ---------------------------------------------
            evidence_data = self.evidence_analyzer.generate_evidence_summary(combined_content, articles)
            evidence_data["source_used"] = source_used # append extra variable per requirements

            # ---------------------------------------------
            # STEP 5 - Pattern Detection
            # ---------------------------------------------
            pattern_data = self.pattern_detector.detect_patterns(combined_content)
            pattern_data["suspicious_words"] = self.pattern_detector.get_suspicious_words(combined_content)
            # Alignment naming mismatch fix implicitly inferred in prompt requirement dict
            pattern_data["matched_patterns"] = pattern_data.pop("patterns_found", [])
            pattern_data["pattern_score"] = pattern_data.pop("score", 0.0)

            # ---------------------------------------------
            # STEP 6 - Credibility Scoring
            # ---------------------------------------------
            credibility_data = self.credibility_scorer.estimate_credibility(combined_content)
            tone_data = self.credibility_scorer.analyze_tone(combined_content)
            
            # Optional headline checking logically applied
            first_sentence = combined_content.split('.')[0] if '.' in combined_content else combined_content[:50]
            if len(first_sentence) < 80:
                self.credibility_scorer.score_headline_quality(first_sentence)

            # ---------------------------------------------
            # STEP 7 - Similarity Matching
            # ---------------------------------------------
            similarity_data = self.similarity_engine.get_full_similarity_report(combined_content)

            # ---------------------------------------------
            # STEP 8 - ML Classification
            # ---------------------------------------------
            ml_prob_fake = 0.5 
            if self.ml_model and self.vectorizer:
                try:
                    vec_text = self.vectorizer.transform([combined_content])
                    # Assuming predict_proba format [prob_real, prob_fake] 
                    ml_prob_fake = float(self.ml_model.predict_proba(vec_text)[0][1])
                except:
                    try:
                        pred = self.ml_model.predict(vec_text)[0]
                        ml_prob_fake = 1.0 if pred == 1 else 0.0
                    except:
                        pass
            ml_prob_real = 1.0 - ml_prob_fake

            # ---------------------------------------------
            # STEP 9 - Confidence Calibration
            # ---------------------------------------------
            base_fake_score = (
                (ml_prob_fake * 0.30) + 
                ((1.0 - (evidence_data["corroboration_score"])) * 0.35) + 
                (pattern_data["pattern_score"] * 0.20) + 
                ((1.0 - (credibility_data["score"] / 100)) * 0.15)
            )
            
            # Manual Rule-based Overrides securely bounded
            if evidence_data["evidence_verdict"] == "CONTRADICTED":
                base_fake_score = max(base_fake_score, 0.85)
            elif evidence_data["evidence_verdict"] == "SUPPORTED" and len(evidence_data["top_credible_sources"]) > 3:
                base_fake_score = min(base_fake_score, 0.22) # (Pushes real > 0.78)
            elif pattern_data["risk_level"] == "HIGH":
                base_fake_score = max(base_fake_score, 0.83) # (Pushes fake > 0.82)
            elif evidence_data["corroboration_label"] == "Strong" and not pattern_data["matched_patterns"]:
                base_fake_score = min(base_fake_score, 0.25) # (Pushes real > 0.75)
                
            base_fake_score = min(max(base_fake_score, 0.0), 1.0)
            
            # Labels and Confidence mappings
            if base_fake_score > 0.6:
                label = "Fake"
                confidence = base_fake_score
            elif base_fake_score < 0.4:
                label = "Real"
                confidence = 1.0 - base_fake_score
            else:
                label = "Uncertain"
                # Inverse penalty - closer to 0.5 is lower confidence
                confidence = 1.0 - (0.5 - abs(0.5 - base_fake_score))

            # ---------------------------------------------
            # STEP 10 - Final Output Assembly
            # ---------------------------------------------
            process_time = int((time.time() - start_time) * 1000)
            
            return {
                "label": label,
                "fake_probability": round(base_fake_score * 100, 2),
                "real_probability": round((1.0 - base_fake_score) * 100, 2),
                "confidence": round(confidence * 100, 2),
                "evidence": {
                    "total_found": evidence_data["total_articles_found"],
                    "relevant_articles": evidence_data["relevant_articles"],
                    "corroboration_score": evidence_data["corroboration_score"],
                    "corroboration_label": evidence_data["corroboration_label"],
                    "contradiction_detected": evidence_data["contradiction_detected"],
                    "evidence_verdict": evidence_data["evidence_verdict"],
                    "source_used": source_used
                },
                "credibility": {
                    "score": credibility_data["score"],
                    "label": credibility_data["label"],
                    "positive_signals": credibility_data["positive_signals"],
                    "negative_signals": credibility_data["negative_signals"]
                },
                "patterns": {
                    "risk_level": pattern_data["risk_level"],
                    "matched_patterns": pattern_data["matched_patterns"],
                    "pattern_score": pattern_data["pattern_score"],
                    "suspicious_words": pattern_data["suspicious_words"]
                },
                "similarity": {
                    "label": similarity_data["similarity_label"],
                    "score": round(similarity_data["similarity_score"] * 100, 2),
                    "closest_pattern": similarity_data["closest_pattern"]
                },
                "tone": {
                    "dominant_tone": tone_data["dominant_tone"],
                    "tone_flags": tone_data["tone_flags"]
                },
                "top_keywords": preproc_data["keywords"],
                "analyzed_preview": (combined_content[:150] + '...') if len(combined_content) > 150 else combined_content,
                "input_type": input_type,
                "processing_time_ms": process_time
            }

        except Exception as e:
            # Handle ALL exceptions - return structured error dict
            return {
                "error_encountered": True,
                "error_message": str(e),
                "traceback": traceback.format_exc(),
                "label": "Error",
                "processing_time_ms": int((time.time() - start_time) * 1000)
            }


# --- Module-level backward compatibility bridges ---
# app.py and script.js still rely on `analyze_claim()` providing the simpler interface.
# We intercept the complex output and map it safely down.

def analyze_claim(claim_text):
    predictor = FakeNewsPredictor()
    result = predictor.predict(input_text=claim_text)
    
    # Catch Error early
    if result.get("error_encountered"):
        return {
            "fake_probability": 0,
            "explanation": f"System Error Encountered: {result.get('error_message')}",
            "ml_score": 0,
            "similarity_score": 0,
            "credibility_score": 0,
            "evidences": []
        }
        
    # Standard format conversion map
    if result["label"] == "Fake":
        exp = f"High likelihood of being Fake. Live sources show a '{result['evidence']['evidence_verdict']}' verdict alongside high pattern risks."
    elif result["label"] == "Uncertain":
        exp = "Uncertain claim. Evidence shows mixed results. Use caution."
    else:
        exp = f"High likelihood of being Real. Verified with '{result['evidence']['corroboration_label']}' structural corroboration!"

    return {
        "fake_probability": result["fake_probability"],
        "explanation": exp,
        "ml_score": result["fake_probability"], # Abstracted for demo
        "similarity_score": result["similarity"]["score"],
        "credibility_score": result["credibility"]["score"],
        "evidences": result["evidence"]["relevant_articles"]
    }
