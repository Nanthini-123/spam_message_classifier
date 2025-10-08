import os, re, joblib, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load dataset
df = pd.read_csv("/Users/nanthinik/Desktop/spam_message_classifier/scripts/dataset/balanced_urls.csv")
df = df[~df['URL'].str.contains("lifepitlok.com", case=False, na=False)]
if df.empty: exit("⚠️ Dataset empty after filtering!")

# Preprocessing
def preprocess_url(url): return re.sub(r"https?://|www\.|[^\w\s]", " ", url.lower())

# Feature extraction
vectorizer, model = TfidfVectorizer(ngram_range=(1, 2)), LogisticRegression(max_iter=500)
X, y = vectorizer.fit_transform(df["URL"]), df["Label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train & evaluate model
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f"\n🎯 Accuracy: {accuracy_score(y_test, y_pred):.4f} | Precision: {precision_score(y_test, y_pred, pos_label='bad'):.4f} | Recall: {recall_score(y_test, y_pred, pos_label='bad'):.4f} | F1: {f1_score(y_test, y_pred, pos_label='bad'):.4f}")

# Save model & vectorizer
models_dir = "/Users/nanthinik/Desktop/spam_message_classifier/scripts/models"
os.makedirs(models_dir, exist_ok=True)
joblib.dump(model, f"{models_dir}/url_classifier.pkl")
joblib.dump(vectorizer, f"{models_dir}/tfidf_vectorizer.pkl")

# Predict URL
def predict_url(url):
    return "✅ Safe" if any(d in url for d in ["google.com", "youtube.com", "facebook.com"]) else "🚨 Phishing" if model.predict(vectorizer.transform([preprocess_url(url)]))[0] == "bad" else "✅ Safe"

# Interactive classification
while (url := input("\n🔎 Enter URL ('exit' to quit): ").strip().lower()) != "exit":
    print(f"🔍 Prediction: {predict_url(url)}")
