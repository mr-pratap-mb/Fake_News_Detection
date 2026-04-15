from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class SimilarityEngine:
    """
    Evaluates text against a known repository of structural misinformation archetypes.
    """
    
    # 30+ rigid examples of how fake news is often grammatically/structurally formatted
    KNOWN_MISINFO_PATTERNS = [
        "Government is hiding the truth about vaccines",
        "This video proves the incident was staged",
        "Share this before Facebook deletes it",
        "Anonymous insider leaks classified document",
        "This photo from 2019 is being shared as recent",
        "Famous personality never actually said this quote",
        "The elites don't want you to know this simple trick",
        "Breaking news the media is completely ignoring",
        "Forward this to everyone on your whatsapp before it's too late",
        "Scientists baffled by miracle cure that pharmaceutical companies hate",
        "Shocking truth revealed about the staged moon landing",
        "Secret society documents exposed in massive database hack",
        "If you don't share this your account will be deleted",
        "Doctors furious after hidden health remedy exposed",
        "Mainstream media is lying to you about the election results",
        "Classified military footage confirms aliens exist",
        "This natural herb destroys 99% of viruses in hours",
        "Local man discovers infinite free energy machine",
        "Unbelievable clip shows politician admitting to everything on hot mic",
        "A hidden camera caught what the government was doing underground",
        "Do not eat this food it contains tracking microchips",
        "New world order agenda document leaked online today",
        "The real reason they locked us down is finally revealed",
        "Secret operation under the airport exposed by brave whistleblowers",
        "They are spraying chemicals in the sky to control the weather",
        "This isn't a natural disaster it was artificially triggered",
        "Top scientist found dead after discovering the truth",
        "The earth is not what they taught you in school look at this proof",
        "Bill gates just announced a terrifying plan for the global population",
        "This ancient civilization had advanced technology that we suppressed",
        "Why is nobody talking about this massive scandal unfolding right now",
        "A secret tunnel network was just uncovered beneath the capital"
    ]

    def __init__(self):
        # We enforce TF-IDF initialization natively against the known archetype array
        self.vectorizer = TfidfVectorizer(stop_words='english')
        # Pre-fit and transform the knowledge base
        self.pattern_matrix = self.vectorizer.fit_transform(self.KNOWN_MISINFO_PATTERNS)

    def compute_similarity(self, input_text):
        """
        Transforms input using fitted vectorizer.
        Computes cosine similarity with all known patterns.
        Returns max similarity score and the matched pattern string.
        """
        if not input_text or not str(input_text).strip():
            return 0.0, ""
            
        try:
            input_vector = self.vectorizer.transform([str(input_text)])
            similarities = cosine_similarity(input_vector, self.pattern_matrix)[0]
            
            max_index = np.argmax(similarities)
            max_score = float(similarities[max_index])
            best_pattern = self.KNOWN_MISINFO_PATTERNS[max_index]
            
            return max_score, best_pattern
        except Exception:
            return 0.0, ""

    def get_similarity_label(self, score):
        """ Assesses risk label strictly based on structural boundaries. """
        if score > 0.45:
            return "HIGH"
        elif score >= 0.25:
            return "MEDIUM"
        else:
            return "LOW"

    def get_full_similarity_report(self, text):
        """
        Orchestration routine providing complete dictionary of similarity mappings.
        """
        if not text or not str(text).strip():
            return {
                "similarity_score": 0.0,
                "similarity_label": "LOW",
                "closest_pattern": "None",
                "top_3_matches": []
            }
            
        try:
            input_vector = self.vectorizer.transform([str(text)])
            similarities = cosine_similarity(input_vector, self.pattern_matrix)[0]
            
            # Extract top 3 matches using argsort logic
            top_3_indices = similarities.argsort()[-3:][::-1]
            top_3_matches = []
            
            for index in top_3_indices:
                score = float(similarities[index])
                # Only append if there's actual structural relevancy (avoids 0.0 fluff)
                if score > 0.05: 
                    top_3_matches.append({
                        "pattern": self.KNOWN_MISINFO_PATTERNS[index],
                        "score": score
                    })

            max_score = float(similarities[top_3_indices[0]]) if top_3_matches else 0.0
            best_pattern = self.KNOWN_MISINFO_PATTERNS[top_3_indices[0]] if top_3_matches else "None"
            
            return {
                "similarity_score": max_score,
                "similarity_label": self.get_similarity_label(max_score),
                "closest_pattern": best_pattern,
                "top_3_matches": top_3_matches
            }
        except Exception:
            return {
                "similarity_score": 0.0,
                "similarity_label": "LOW",
                "closest_pattern": "None",
                "top_3_matches": []
            }

# --- Module-level backward compatibility bridges ---
# Retained for existing `predictor.py` which hasn't been upgraded to this new spec yet!

from utils.text_processor import clean_text as fallback_clean_text

def calculate_similarity(claim, evidence_texts):
    """
    Legacy wrapper utilizing TfidfVectorizer dynamically between a claim and live evidence.
    """
    if not evidence_texts or not claim: return 0.0
    cleaned_claim = fallback_clean_text(claim)
    cleaned_evidences = [fallback_clean_text(e) for e in evidence_texts if e]
    corpus = [cleaned_claim] + cleaned_evidences
    try:
        vectorizer = TfidfVectorizer().fit_transform(corpus)
        vectors = vectorizer.toarray()
        similarities = cosine_similarity([vectors[0]], vectors[1:])[0]
        return float(max(similarities))
    except Exception:
        return 0.0
