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
    print("✅ Models and vectorizers loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")

# Preprocessing functions
def preprocess_message(message):
    return message.lower().strip()  # Basic text preprocessing

def preprocess_url(url):
    return url.lower().strip()  # Basic URL preprocessing

# Serve the main HTML page
@app.route("/")
def index():
    return render_template("index.html")
@app.route("/classify_message", methods=["POST"])
def classify_message():
    try:
        data = request.get_json()
        message = data.get("message", "").lower().strip()

        if not message:
            return jsonify({"error": "Message is required!"}), 400

        # Transform the message using the vectorizer
        features = message_vectorizer.transform([message])

        # Predict using the model
        prediction = message_model.predict(features)[0]  # Returns 0 or 1

        # Convert numerical prediction to text
        result = "Spam" if prediction == 1 else "Not Spam"

        return jsonify({"prediction": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Keyword-based phishing detection
PHISHING_KEYWORDS = {"free", "money", "cash", "win", "winner", "prize", "offer", "claim", "click", "congratulations"}
# Simple keyword-based phishing detection
PHISHING_KEYWORDS = {"free", "money", "cash", "win", "offer", "prize", "click", "claim", "iphone", "now", "guaranteed"}
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
      


    except Exception as e:
        return jsonify({"error": str(e)}), 500
# Handle favicon.ico error
@app.route('/favicon.ico')
def favicon():
    return '', 204
if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=8000)
# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True, port=8000)
