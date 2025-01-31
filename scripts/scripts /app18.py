from flask import Flask, request, jsonify, render_template
import joblib
import os
import re

app = Flask(__name__, static_folder='static', template_folder='templates')

# Define the base path for the models directory
base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')

# Construct the full paths for message and URL models and vectorizers
message_model_path = os.path.join(base_path, 'spam_classifier_model.joblib')
message_vectorizer_path = os.path.join(base_path, 'tfidf_vectorizer.joblib')
url_model_path = os.path.join(base_path, 'url_classifier.pkl')
url_vectorizer_path = os.path.join(base_path, 'vectorizer.pkl')

# Load models and vectorizer

try:
    message_model = joblib.load(message_model_path)
    message_vectorizer = joblib.load(message_vectorizer_path)
    url_model = joblib.load(url_model_path)
    url_vectorizer = joblib.load(url_vectorizer_path)
    print("Models and vectorizers loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading models or vectorizers: {e}")
    raise

# Serve the index.html page using Flask's render_template
@app.route('/')
def index():
    return render_template('index.html')

# Preprocessing function for URLs
def preprocess_url(url):
    url = url.lower()
    url = re.sub(r"https?://", "", url)
    url = re.sub(r"www\.", "", url)
    url = re.sub(r"[^\w\s]", " ", url)
    return url

# Predict for message classification
@app.route("/classify-message", methods=["POST"])
def classify_message():
    message = request.form.get("message")
    if message:
        # Preprocess and vectorize the message
        transformed_message = message_vectorizer.transform([message])  
        prediction = message_model.predict(transformed_message)[0]
        result = "Spam" if prediction == 1 else "Not Spam"
        return jsonify({"message": message, "classification": result})
    return jsonify({"error": "No message received"}), 400

# Predict for URL classification
@app.route("/classify-url", methods=["POST"])
def classify_url():
    url = request.form.get("url")
    if url:
        processed_url = preprocess_url(url)
        # Vectorize the URL and predict
        transformed_url = url_vectorizer.transform([processed_url])
        prediction = url_model.predict(transformed_url)[0]
        
        result = "Phishing" if prediction == 'bad' else "Safe"
        return jsonify({"url": url, "classification": result})
    return jsonify({"error": "No URL received"}), 400

# Predict for message classification (JSON endpoint)
@app.route('/predict-message', methods=['POST'])
def predict_message():
    data = request.get_json()
    message = data.get('message')
    if not message:
        return jsonify({'error': 'Message is required'}), 400
    
    vector = message_vectorizer.transform([message])
    prediction = message_model.predict(vector)
    prediction = int(prediction[0])

    return jsonify({'prediction': prediction})

# Predict for URL classification (JSON endpoint)
@app.route('/predict-url', methods=['POST'])
def predict_url():
    data = request.get_json()
    url = data.get('url')

    if not url:
        return jsonify({'error': 'URL is required'}), 400

    # Preprocess and vectorize the URL
    processed_url = preprocess_url(url)
    vectorized_url = url_vectorizer.transform([processed_url])

    # Get the model's prediction
    prediction = url_model.predict(vectorized_url)

    # Handle possible 'bad' or 'good' outcomes
    prediction = prediction[0]

    if prediction == 'bad':
        prediction = 1
    elif prediction == 'good':
        prediction = 0
    else:
        return jsonify({'error': 'Unexpected prediction result'}), 500

    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8000)))
