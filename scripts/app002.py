from flask import Flask, request, jsonify, render_template
import joblib
import os

# Initialize Flask app
app = Flask(__name__)

# Define model and vectorizer paths
message_model_path = "/Users/nanthinik/Desktop/spam_message_classifier/scripts/models/spam_classifier_model.joblib"
url_model_path = "/Users/nanthinik/Desktop/spam_message_classifier/scripts/models/url_classifier.pkl"
message_vectorizer_path = "/Users/nanthinik/Desktop/spam_message_classifier/scripts/models/tfidf_vectorizer.joblib"
url_vectorizer_path = "/Users/nanthinik/Desktop/spam_message_classifier/scripts/models/tfidf_vectorizer.pkl"

# Load the models and vectorizers
try:
    message_model = joblib.load(message_model_path)
    message_vectorizer = joblib.load(message_vectorizer_path)
    url_model = joblib.load(url_model_path)
    url_vectorizer = joblib.load(url_vectorizer_path)
    print("✅ Models loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")

# Preprocessing functions
def preprocess_message(message):
    return message.lower().strip()  # Example preprocessing

def preprocess_url(url):
    return url.lower().strip()  # Example preprocessing

# Serve the main HTML page
@app.route("/")
def index():
    return render_template("index.html")
# Simple keyword-based spam detection
SPAM_KEYWORDS = {"free", "money", "cash", "win", "offer", "prize", "click", "claim", "iphone", "now", "guaranteed"}

# Keyword-based spam detection
SPAM_KEYWORDS = {"free", "money", "cash", "win", "winner", "prize", "offer", "claim", "click", "congratulations"}

@app.route("/classify_message", methods=["POST"])
def classify_message():
    try:
        data = request.get_json()
        message = data.get("message", "").lower().strip()

        if not message:
            return jsonify({"error": "Message is required!"}), 400

        # Check for spam keywords
        if any(word in message for word in SPAM_KEYWORDS):
            result = "Spam"
        else:
            result = "Not Spam"

        return jsonify({"prediction": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Keyword-based phishing detection
PHISHING_KEYWORDS = {"login", "verify", "update", "bank", "secure", "account", "confirm", "free", "gift", "claim"}

@app.route("/classify_url", methods=["POST"])
def classify_url():
    try:
        data = request.get_json()
        url = data.get("url", "").lower().strip()

        if not url:
            return jsonify({"error": "URL is required!"}), 400

        # Check for phishing keywords
        if any(word in url for word in PHISHING_KEYWORDS):
            result = "Phishing"
        else:
            result = "Safe"

        return jsonify({"prediction": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
      

# Handle favicon.ico error
@app.route('/favicon.ico')
def favicon():
    return '', 204

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True, port=8000)
