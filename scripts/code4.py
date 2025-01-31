import os
import re
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import whois

# ✅ Load dataset
dataset_path = "/Users/nanthinik/Desktop/spam_message_classifier/scripts/dataset/balanced_urls.csv"
if not os.path.exists(dataset_path):
    print(f"Error: File not found - {dataset_path}")
else:
    df = pd.read_csv(dataset_path)
    print("✅ Dataset loaded successfully!")
    print(df.head())  # Print first 5 rows to verify

# ✅ Function to clean URLs
def preprocess_url(url):
    url = url.lower()  # Convert URL to lowercase
    url = re.sub(r"https?://", "", url)  # Remove http or https
    url = re.sub(r"www\.", "", url)  # Remove www.
    url = re.sub(r"[^\w\s]", " ", url)  # Remove special characters
    return url    

# ✅ Filter out rows with the problematic domain
df = df[~df['URL'].str.contains("lifepitlok.com", case=False, na=False)]

# Print the updated dataset to verify
print("\n✅ Dataset after filtering problematic domain:")
print(df.head())  # Verify the dataset is filtered correctly

# ✅ Check if the dataset is empty after filtering
if df.empty:
    print("⚠️ The dataset is empty after filtering. Exiting the script.")
    exit()  # Exit if the dataset is empty

# ✅ Trusted domains (You can expand this list based on your criteria)
trusted_domains = ["google.com", "youtube.com", "facebook.com"]

def is_trusted(url):
    return any(domain in url for domain in trusted_domains)

# ✅ Function to extract domain name from URL
def extract_domain(url):
    url = url.lower()  # Convert to lowercase
    url = re.sub(r'https?://', '', url)  # Remove protocol
    url = re.sub(r'www\.', '', url)  # Remove www
    domain = url.split('/')[0]  # Get the domain name
    return domain


# ✅ Feature extraction using TF-IDF
vectorizer = TfidfVectorizer(ngram_range=(1, 2))  # Bi-grams improve detection
X = vectorizer.fit_transform(df["URL"])  # Use "URL" column directly
y = df["Label"]

# ✅ Splitting dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Train model (using Logistic Regression for faster training)
print("Training model... (Using Logistic Regression, faster)")
model = LogisticRegression(max_iter=500)  # Increase iterations for stability
model.fit(X_train, y_train)
print("✅ Model training completed!")

# ✅ Model evaluation
y_pred = model.predict(X_test)

print("\n🎯 Model Performance:")
print(f"✅ Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"✅ Precision: {precision_score(y_test, y_pred, pos_label='bad'):.4f}")
print(f"✅ Recall: {recall_score(y_test, y_pred, pos_label='bad'):.4f}")
print(f"✅ F1-score: {f1_score(y_test, y_pred, pos_label='bad'):.4f}")

# ✅ Save model and vectorizer
models_dir = "/Users/nanthinik/Desktop/spam_message_classifier/scripts/models"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

model_path = os.path.join(models_dir, "url_classifier.pkl")
vectorizer_path = os.path.join(models_dir, "tfidf_vectorizer.pkl")

joblib.dump(model, model_path)
joblib.dump(vectorizer, vectorizer_path)

print("\n✅ Model and vectorizer saved successfully!")

# ✅ Load model and vectorizer
model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

print("✅ Model and vectorizer loaded successfully!")

# ✅ Interactive URL classification
def predict_url(url):
    if is_trusted(url):  # Check if URL is trusted before prediction
        return "✅ Safe (good)"  # Return safe for trusted domains
    processed_url = preprocess_url(url)  # Clean the URL
    features = vectorizer.transform([processed_url])  # Transform the cleaned URL
    prediction = model.predict(features)[0]  # Predict the class
    return "🚨 Phishing (bad)" if prediction == "bad" else "✅ Safe (good)"

while True:
    new_url = input("\n🔎 Enter a URL to classify (or type 'exit' to stop): ").strip()
    if new_url.lower() == "exit":
        print("🔴 Exiting... Goodbye!")
        break
    prediction = predict_url(new_url)
    print(f"🔍 Prediction: {prediction}")
