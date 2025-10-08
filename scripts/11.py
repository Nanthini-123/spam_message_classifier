import os, pickle, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Load data
def load_data(path='./dataset/messages.csv'):
    if not os.path.exists(path): exit(f"Error: {path} not found!")
    df = pd.read_csv(path, encoding="ISO-8859-1")[['v1', 'v2']].rename(columns={'v1': 'label', 'v2': 'message'})
    return df.dropna().assign(label=df['label'].map({'ham': 0, 'spam': 1}), message=df['message'].str.lower())

# Train model
def train():
    df = load_data()
    X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)
    vec, model = TfidfVectorizer(max_features=5000), LogisticRegression()
    X_train, X_test = vec.fit_transform(X_train), vec.transform(X_test)
    model.fit(X_train, y_train)
    print("\n📊 Model Metrics:\n", classification_report(y_test, model.predict(X_test)))
    return model, vec

# Save model
def save(model, vec, model_path='./models/spam_model.pkl', vec_path='./models/tfidf_vectorizer.pkl'):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    pickle.dump(model, open(model_path, 'wb'))
    pickle.dump(vec, open(vec_path, 'wb'))
    print("✅ Model & Vectorizer Saved!")

# Load model for message classification
def classify_message(model, vec, message):
    return "📩 Spam" if model.predict(vec.transform([message.lower()]))[0] else "📨 Ham"

# Run
if __name__ == "__main__":
    model, vec = train()
    save(model, vec)
    
    while True:
        msg = input("\n💬 Enter message (or 'exit' to quit): ").strip()
        if msg.lower() == "exit": break
        print(f"🔍 Classified as: {classify_message(model, vec, msg)}")
