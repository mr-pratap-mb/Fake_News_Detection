import os
import json
import random
import time
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score, classification_report
import joblib

def generate_synthetic_dataset(num_samples=1200):
    """
    Generates an array of synthetic news texts dynamically alternating 
    between credible patterns and fake formatting structures.
    """
    print(f"Generating {num_samples} synthetic training samples...")
    data = []
    
    # Text Generation Blocks
    fake_subjects = ["The government", "Deep state elites", "Secret societies", "Scientists", "Doctors", "Mainstream media", "Politicians", "Corporate billionaires"]
    fake_actions = ["are hiding the truth about", "lied to you regarding", "have covered up", "are secretly spreading", "just passed a secret order concerning", "dont want you to know about"]
    fake_targets = ["miracle vaccines", "tracking microchips", "weather machines", "alien encounters", "staged global events", "infinite free energy", "chemtrail poisoning"]
    fake_urgency = ["Share this immediately", "Wake up sheeple", "Before it's deleted", "Shocking leak", "Viral truth exposed", "Censored everywhere"]
    
    real_subjects = ["The local council", "A recent study", "International observers", "The health department", "Economic reports", "Global organizations", "University researchers", "Tech analysts"]
    real_actions = ["has announced a new initiative regarding", "published findings on", "reported an increase in", "approved funding for", "detailed the impact of", "concluded their investigation into"]
    real_targets = ["regional infrastructure changes", "quarterly market fluctuations", "seasonal healthcare trends", "educational performance metrics", "renewable energy statistics", "public safety protocols"]
    real_closing = ["Read the full report online", "Available in the latest journal issue", "Press release distributed locally", "Data compiled by official agencies", "Scheduled for community feedback"]

    half = num_samples // 2
    
    # 1. Generate Fake Scenarios (Label 1)
    for _ in range(half):
        text = f"{random.choice(fake_urgency)}! {random.choice(fake_subjects)} {random.choice(fake_actions)} {random.choice(fake_targets)}!"
        # Add slight variations
        if random.random() > 0.5:
            text += f" Evidence inside video!"
        data.append({"text": text, "label": 1})
        
    # 2. Generate Real Scenarios (Label 0)
    for _ in range(half):
        text = f"{random.choice(real_subjects)} {random.choice(real_actions)} {random.choice(real_targets)}. {random.choice(real_closing)}."
        data.append({"text": text, "label": 0})
        
    # Shuffle Dataset
    random.shuffle(data)
    return pd.DataFrame(data)


def train_and_evaluate():
    print("--- FakeShield v2 ML Training Pipeline ---")
    start_time = time.time()
    
    # 1. Gather Data
    df = generate_synthetic_dataset(1200)
    X = df['text']
    y = df['label']
    
    # 2. Extract Features using specified parameters
    print("\nVectorizing textual datasets...")
    vectorizer = TfidfVectorizer(max_features=15000, ngram_range=(1, 3))
    X_vec = vectorizer.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)
    
    # 3. Model Declarations
    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=500),
        "Linear SVC": LinearSVC(max_iter=1000, dual=False),
        "Multinomial Naive Bayes": MultinomialNB()
    }
    
    best_model_name = ""
    best_f1_score = 0.0
    best_model_instance = None
    
    print("\nTraining and Evaluating Models:")
    print("-" * 40)
    
    for name, clf in classifiers.items():
        # Fit
        clf.fit(X_train, y_train)
        # Predict
        preds = clf.predict(X_test)
        # Score via F1
        f1 = f1_score(y_test, preds)
        
        print(f"[{name}] F1-Score: {f1:.4f}")
        
        # We capture the best iteration
        if f1 > best_f1_score:
            best_f1_score = f1
            best_model_name = name
            best_model_instance = clf
            
    print("-" * 40)
    print(f"Optimal Model Selected: {best_model_name} (F1: {best_f1_score:.4f})")
    
    # Print Full Report for Best Model
    print("\nBest Model Classification Report:")
    best_preds = best_model_instance.predict(X_test)
    print(classification_report(y_test, best_preds, target_names=["Real News (0)", "Fake News (1)"]))
    
    # 4. Save Models & Artifacts natively
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    model_path = os.path.join(models_dir, 'best_model.pkl')
    vec_path = os.path.join(models_dir, 'tfidf_vectorizer.pkl')
    meta_path = os.path.join(models_dir, 'model_metadata.json')
    
    joblib.dump(best_model_instance, model_path)
    joblib.dump(vectorizer, vec_path)
    
    metadata = {
        "model_name": best_model_name,
        "f1_score": best_f1_score,
        "training_samples": 1200,
        "vectorizer_config": {
            "max_features": 15000,
            "ngram_range": [1, 3]
        },
        "timestamp": time.time()
    }
    
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=4)
        
    print(f"\nArtifacts Saved in [models/]:")
    print(f"- {os.path.basename(model_path)}")
    print(f"- {os.path.basename(vec_path)}")
    print(f"- {os.path.basename(meta_path)}")
    
    print(f"\nTraining completed in {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    train_and_evaluate()
