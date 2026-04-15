import re

class CredibilityScorer:
    """
    Semantic credibility scorer engine for FakeShield v2.
    Evaluates emotional manipulation and semantic tone markers alongside domain scoring.
    """
    
    SEMANTIC_SIGNALS = {
        "sensational": ["mind-blowing", "shocking", "destroyed", "slam", "obliterate", "you won't believe"],
        "fear_inducing": ["terrifying", "warning", "deadly", "imminent threat", "collapse", "danger"],
        "authority_misuse": ["top doctor", "secret expert", "leaked document", "banned video", "insider reveals"],
        "urgency": ["act now", "before it gets deleted", "urgent", "breaking", "asap", "hurry"],
        "emotional_bias": ["disgusting", "outrageous", "heroic", "evil", "corrupt", "genius"],
        "polarization": ["libtard", "snowflake", "fascist", "sheeple", "traitor", "enemy of the people"]
    }

    def analyze_tone(self, text):
        """
        Evaluate semantic signals to construct tone flags and a tone score.
        """
        if not text:
            return {"tone_flags": [], "dominant_tone": "neutral", "tone_score": 0}
            
        text_lower = str(text).lower()
        tone_flags = []
        
        category_counts = {k: 0 for k in self.SEMANTIC_SIGNALS.keys()}
        total_hits = 0
        
        for tone_category, signals in self.SEMANTIC_SIGNALS.items():
            for signal in signals:
                if signal in text_lower:
                    if tone_category not in tone_flags:
                        tone_flags.append(tone_category)
                    category_counts[tone_category] += 1
                    total_hits += 1
                    
        dominant_tone = "neutral"
        if total_hits > 0:
            dominant_tone = max(category_counts, key=category_counts.get)
            
        return {
            "tone_flags": tone_flags,
            "dominant_tone": dominant_tone,
            "tone_score": total_hits
        }

    def estimate_credibility(self, text):
        """
        Calculates score 0-100, label, and array of positive/negative signals.
        """
        tone_data = self.analyze_tone(text)
        
        # Base neutral score is 100. Penalties are deducted.
        score = 100
        negative_signals = []
        positive_signals = []
        
        if tone_data["tone_score"] > 0:
            score -= (tone_data["tone_score"] * 10)
            negative_signals.extend(tone_data["tone_flags"])
        else:
            positive_signals.append("neutral_objective_tone")
            
        # Basic bounds checking
        score = max(0, min(100, score))
        
        label = "High"
        if score < 40:
            label = "Low"
        elif score < 75:
            label = "Medium"
            
        return {
            "score": score,
            "label": label,
            "positive_signals": positive_signals,
            "negative_signals": negative_signals
        }

    def score_headline_quality(self, headline):
        """
        Analyzes the headline array for manipulative clickbait attributes.
        Returns: {"headline_score": int, "issues": [...]}
        """
        if not headline:
            return {"headline_score": 50, "issues": ["No headline provided"]}
            
        score = 100
        issues = []
        
        # 1. Detect ALL CAPS words (excluding valid acronyms usually max length 4)
        words = headline.split()
        all_caps = [w for w in words if w.isupper() and len(re.sub(r'[^A-Z]', '', w)) > 4]
        if all_caps:
            score -= 20
            issues.append("Excessive capitalization")
            
        # 2. Excessive punctuation
        if bool(re.search(r'([!]{2,}|[\?]{2,})', headline)):
            score -= 15
            issues.append("Excessive punctuation")
            
        # 3. Question-bait
        if headline.strip().endswith('?') and any(qw in headline.lower() for qw in ['is this', 'are they', 'could it']):
            score -= 10
            issues.append("Question-bait structure")
            
        # Bonus positive modifier
        if not issues:
            score = 100
            
        score = max(0, min(100, score))
        
        if score == 100:
            issues = ["Balanced neutral language"]
            
        return {
            "headline_score": score,
            "issues": issues
        }

    def compute_semantic_adjustment(self, text):
        """
        Provide float multiplier 0.8-1.2 based on detected tones.
        """
        tone_data = self.analyze_tone(text)
        hits = tone_data["tone_score"]
        
        # Too many red flag tones pushes the severity multiplier upwards (fake news multiplier -> high = bad)
        # So multiplier 1.2 makes it MORE likely to be flagged.
        if hits >= 3:
            return 1.2
        elif hits >= 1:
            return 1.1
        elif hits == 0:
            return 0.9 # Decreases risk factor gracefully
            
        return 1.0


# --- Module-level backward compatibility bridges ---
# Ensures ecosystem remains fully intact where expected.

import config

def get_domain_credibility(domain):
    """ Legacy bridge utilizing naive credible arrays """
    if not domain or domain == "unknown":
        return 0.0
    domain_lower = domain.lower()
    for c in config.CREDIBLE_DOMAINS:
        if c in domain_lower: return 1.0
    for s in config.SUSPICIOUS_DOMAINS:
        if s in domain_lower: return -1.0
    return 0.0

def assess_source_credibility(sources_list):
    """ Legacy bridge running naive assessment array loop """
    if not sources_list: return 0.5
    total = sum(get_domain_credibility(s.get('domain', '')) for s in sources_list)
    avg_score = (total / len(sources_list)) 
    normalized = 0.5 + (avg_score * 0.5) 
    return min(max(normalized, 0.0), 1.0)
