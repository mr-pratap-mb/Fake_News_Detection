import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- NLTK Data Downloads at Module Load ---
# Set quiet=True to avoid dumping logs to console constantly
try:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
except Exception as e:
    print(f"Warning: NLTK Download failed: {e}")

class TextProcessor:
    """
    Complete NLP text preprocessing module for FakeShield v2.
    """
    
    def __init__(self):
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            self.stop_words = set()
        self.lemmatizer = WordNetLemmatizer()

    def clean_text(self, text):
        """
        Lowercase, remove URLs, remove HTML tags, keep only text and basic punctuation,
        and remove extra whitespace.
        """
        if not text:
            return ""
        
        # Lowercase
        text = str(text).lower()
        
        # Remove URLs (http/https/www)
        text = re.sub(r'https?://[^\s]+|www\.[^\s]+', '', text)
        
        # Remove HTML tags using simple regex
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove special characters except . , ! ? (Replace with space)
        text = re.sub(r'[^a-z0-9\.\,\!\?\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def tokenize(self, text):
        """ Word tokenize using NLTK """
        if not text:
            return []
        try:
            return word_tokenize(text)
        except Exception:
            # Fallback if punkt is missing entirely
            return text.split()

    def remove_stopwords(self, tokens):
        """ Filter NLTK English stopwords, keep words > 2 chars """
        if not tokens:
            return []
        return [word for word in tokens if word not in self.stop_words and len(word) > 2]

    def lemmatize(self, tokens):
        """ Apply WordNetLemmatizer to a list of tokens """
        if not tokens:
            return []
        return [self.lemmatizer.lemmatize(word) for word in tokens]

    def extract_ngrams(self, tokens, n=2):
        """ Return bigrams and trigrams (combined list of strings) """
        if not tokens:
            return []
        # Return tuples joined as strings
        bigrams = ngrams(tokens, 2)
        trigrams = ngrams(tokens, 3)
        combined = [" ".join(gram) for gram in bigrams] + [" ".join(gram) for gram in trigrams]
        return combined

    def get_keywords(self, text, top_n=10):
        """
        Remove stopwords, numbers, single chars
        Return top N by frequency as list of strings
        """
        if not text:
            return []
        
        clean_t = self.clean_text(text)
        # Remove punctuation for keyword frequency exactness and numbers
        clean_t = re.sub(r'[^a-z\s]', '', clean_t)
        
        tokens = self.tokenize(clean_t)
        filtered = [t for t in tokens if t not in self.stop_words and len(t) > 1 and not t.isnumeric()]
        
        counts = Counter(filtered)
        # Return words (first element of the Most Common tuple)
        return [word for word, count in counts.most_common(top_n)]

    def extract_claim_keywords(self, text):
        """
        Extract the most meaningful nouns/verbs (core claim words)
        Return top 5 words as space-separated string
        """
        if not text:
            return ""
            
        clean_t = self.clean_text(text)
        # Remove punctuation to ensure pure word tagging
        clean_t = re.sub(r'[^a-z\s]', '', clean_t)
        
        tokens = self.tokenize(clean_t)
        filtered_tokens = [t for t in tokens if t not in self.stop_words and len(t) > 2]
        
        try:
            pos_tags = nltk.pos_tag(filtered_tokens)
            # Nouns (NN, NNP, NNS, NNPS) and Verbs (VB, VBD, VBG, VBN, VBP, VBZ)
            core_words = [word for word, tag in pos_tags if tag.startswith('NN') or tag.startswith('VB')]
        except:
            # Fallback if pos_tagger fails
            core_words = filtered_tokens
            
        counts = Counter(core_words)
        top_5 = [word for word, count in counts.most_common(5)]
        
        return " ".join(top_5)

    def compute_text_similarity(self, text1, text2):
        """
        Use TF-IDF cosine similarity between two texts.
        Return float 0.0-1.0
        """
        if not text1 or not text2:
            return 0.0
            
        try:
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform([text1, text2])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(similarity)
        except Exception:
            return 0.0

    def preprocess_pipeline(self, text):
        """
        Run all steps, return a dictionary of processed NLP states.
        """
        if not text:
            return {
                "original": "",
                "cleaned": "",
                "tokens": [],
                "keywords": [],
                "claim_keywords": "",
                "ngrams": [],
                "word_count": 0,
                "sentence_count": 0
            }

        cleaned = self.clean_text(text)
        
        # Crude sentence count by punctuation
        # Re-using original text for accurate sentence splits
        sentence_count = max(1, len(re.split(r'[.!?]+', text)) - 1)
        
        # For tokens, ngrams, keywords, operate on a fully stripped version
        fully_stripped = re.sub(r'[^a-z\s]', '', cleaned)
        tokens = self.tokenize(fully_stripped)
        
        word_count = len(tokens)
        
        filtered = self.remove_stopwords(tokens)
        lemmas = self.lemmatize(filtered)
        
        ngrams_list = self.extract_ngrams(lemmas, n=2)
        keywords = self.get_keywords(text, top_n=10)
        claim_keywords = self.extract_claim_keywords(text)
        
        return {
            "original": text,
            "cleaned": cleaned,
            "tokens": lemmas,
            "keywords": keywords,
            "claim_keywords": claim_keywords,
            "ngrams": ngrams_list,
            "word_count": word_count,
            "sentence_count": sentence_count
        }

# --- Module-level backward compatibility bridges ---
# Because other files rely on `clean_text` and `extract_keywords` top-level functions

_processor = TextProcessor()

def clean_text(text):
    return _processor.clean_text(text)

def extract_keywords(text, num_keywords=5):
    # Using claim_keywords extraction
    words_str = _processor.extract_claim_keywords(text)
    words_list = words_str.split()
    return words_list[:num_keywords] if words_list else _processor.get_keywords(text, top_n=num_keywords)
