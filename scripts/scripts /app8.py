from flask import Flask, request, jsonify, render_template
import joblib
import os

app = Flask(__name__, static_folder='static', template_folder='templates')

# Define the base path for the models directory
base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')

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

# Serve the index.html page using Flask's render_template
@app.route('/')
def index():
    return render_template('index.html')  # Flask automatically looks in the 'templates' folder

@app.route("/classify", methods=["POST"])
def classify_message():
    message = request.form.get("message")
    if message:
        # Dummy classification logic
        transformed_message = message_vectorizer.transform([message])  
        prediction = message_model.predict(transformed_message)[0]
        result = "Spam" if prediction == 1 else "Not Spam"
        return jsonify({"message": message, "classification": result})
    return jsonify({"error": "No message received"}), 400



@app.route("/classify-url", methods=["POST"])
def classify_url():
    url = request.form.get("url")
    if url:
        # Dummy URL classification logic
        transformed_url = url_vectorizer.transform([url])  # Use correct vectorizer
        prediction = url_model.predict(transformed_url)[0]
        result = "Spam URL" if prediction == 0 else "Spam URL"
        return jsonify({"url": url, "classification": result})
    return jsonify({"error": "No URL received"}), 400

# Predict for message classification
@app.route('/predict-message', methods=['POST'])
def predict_message():
    data = request.get_json()
    message = data.get('message')
    if not message:
        return jsonify({'error': 'Message is required'}), 400
    
    vector = message_vectorizer.transform([message])
    print("vectorized Input:",vector)
    prediction = message_model.predict(vector)
    prediction = int(prediction[0])

    return jsonify({'prediction': prediction})

@app.route('/predict-url', methods=['POST'])
def predict_url():
    # Extract URL from the request
    data = request.get_json()
    url = data.get('url')

    if not url:
        return jsonify({'error': 'URL is required'}), 400

    # Vectorize the URL (you already have this part)
    vectorized_url = url_vectorizer.transform([url])

    # Get the model's prediction
    prediction = url_model.predict(vectorized_url)

    # If the model outputs strings like 'bad' or 'good', convert them to integers
    prediction = prediction[0]  # Assuming prediction is a list/array with one element

    if prediction == 'bad':
        prediction = 1
    elif prediction == 'good':
        prediction = 0
    else:
        # In case there are other outputs, handle them here
        return jsonify({'error': 'Unexpected prediction result'}), 500
    print("Vectorized URL:", vectorized_url.toarray())  # Print vectorized form of URL
    print("Prediction:", prediction)  # Check model prediction 

    # Return the prediction as an integer
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8000)))
