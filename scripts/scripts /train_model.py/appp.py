from flask import Flask, request, jsonify
import joblib
import os

app = Flask(__name__)

# Define the base path for the models directory
base_path = '/Users/nanthinik/Desktop/spam_message_classifier/scripts/models/'

# Construct the full paths for message and URL models and vectorizers
message_model_path = os.path.join(base_path, 'spam_classifier_model.joblib')
message_vectorizer_path = os.path.join(base_path, 'tfidf_vectorizer.joblib')
url_model_path = os.path.join(base_path, 'model.pkl')
url_vectorizer_path = os.path.join(base_path, 'vectorizer.pkl')

# Load models and vectorizers
try:
    message_model = joblib.load(message_model_path)
    message_vectorizer = joblib.load(message_vectorizer_path)
    url_model = joblib.load(url_model_path)
    url_vectorizer = joblib.load(url_vectorizer_path)
    print("Models and vectorizers loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading models or vectorizers: {e}")
    raise

# Predict for message classification
@app.route('/predict-message', methods=['POST'])
def predict_message():
    data = request.json
    message = data.get('message')
    if not message:
        return jsonify({'error': 'Message is required'}), 400
    vector = message_vectorizer.transform([message])
    prediction = message_model.predict(vector)
    return jsonify({'prediction': prediction[0]})

# Predict for URL classification
@app.route('/predict-url', methods=['POST'])
def predict_url():
    data = request.json
    url = data.get('url')
    if not url:
        return jsonify({'error': 'URL is required'}), 400
    vector = url_vectorizer.transform([url])
    prediction = url_model.predict(vector)
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
