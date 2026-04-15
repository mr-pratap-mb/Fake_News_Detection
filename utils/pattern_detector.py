import re

class PatternDetector:
    """
    Misinformation pattern detector for FakeShield v2.
    Scans text against known heuristic fake-news manipulation vectors.
    """
    
    MISINFORMATION_PATTERNS = {
        "viral_claim": ["viral", "going viral", "share this before deleted", "forward this"],
        "old_media_reuse": ["old video", "old photo", "resurfaced", "from 2019", "from 2020", "years ago"],
        "fabricated_quote": ["never said", "falsely attributed", "fake statement", "misquoted", "fabricated"],
        "edited_media": ["edited image", "morphed", "doctored", "photoshopped", "manipulated photo"],
        "whatsapp_rumor": ["whatsapp forward", "received this message", "please share", "chain message"],
        "fake_authority": ["government hiding", "classified leak", "secret order", "pm declared secretly"],
        "clickbait": ["you won't believe", "shocking truth", "they don't want you to know", "exposed secretly"],
        "unverified_stats": ["according to anonymous sources", "insiders reveal", "secret report shows"],
        "urgency_trigger": ["share immediately", "before it's deleted", "act now", "urgent breaking"],
        "conspiracy": ["deep state", "new world order", "agenda exposed", "illuminati", "staged event"]
    }

    def detect_patterns(self, text):
        """
        Return full pattern report with risk level HIGH/MEDIUM/LOW.
        """
        if not text:
            return {"risk_level": "LOW", "patterns_found": [], "score": 0.0}
            
        patterns_found = self.get_pattern_summary(text)
        score = self.compute_pattern_score(text)
        
        if score > 0.6 or len(patterns_found) >= 3:
            risk = "HIGH"
        elif score > 0.3 or len(patterns_found) >= 1:
            risk = "MEDIUM"
        else:
            risk = "LOW"
            
        return {
            "risk_level": risk,
            "patterns_found": patterns_found,
            "score": score
        }

    def get_suspicious_words(self, text):
        """
        Return a list of matched trigger words natively discovered in the text.
        """
        if not text:
            return []
            
        text_lower = text.lower()
        matched = []
        
        for category, triggers in self.MISINFORMATION_PATTERNS.items():
            for trigger in triggers:
                # Basic string inclusion check. 
                if trigger in text_lower:
                    matched.append(trigger)
                    
        return list(set(matched))

    def compute_pattern_score(self, text):
        """
        Returns a float 0.0-1.0 mapping severity based on triggers found.
        """
        matched = self.get_suspicious_words(text)
        if not matched:
            return 0.0
            
        # Mathematical severity scaling: finding 4 or more trigger patterns caps at 1.0!
        score = min(len(matched) * 0.25, 1.0)
        return float(score)

    def get_pattern_summary(self, text):
        """
        Return human readable list of detected patterns (keys).
        """
        if not text:
            return []
            
        text_lower = text.lower()
        active_categories = []
        
        for category, triggers in self.MISINFORMATION_PATTERNS.items():
            for trigger in triggers:
                if trigger in text_lower:
                    active_categories.append(category)
                    break # Stop evaluating this category once triggered
                    
        return active_categories

# --- Module-level backward compatibility bridges ---
# Allows early phase predictor.py sequence to gracefully hit the mock ML probability pipeline.

import os
import joblib

def get_ml_fake_probability(text):
    """
    Legacy ML fetcher. Still utilizes the locally generated dummy TF-IDF 
    pipeline inside models/fake_news_model.pkl for standard ML scoring.
    """
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'fake_news_model.pkl')
    try:
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            try:
                prob = model.predict_proba([text])[0][1]
            except:
                pred = model.predict([text])[0]
                prob = 1.0 if pred == 1 else 0.0
            return float(prob)
        return 0.5
    except:
        return 0.5
