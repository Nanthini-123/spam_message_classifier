import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import os
import joblib

def load_and_preprocess_data_for_urls():
    # Load the URL dataset
    urls_file = '/Users/nanthinik/Desktop/spam_message_classifier/scripts/dataset/urls.csv'
    try:
        urls = pd.read_csv(urls_file)
        print(f"Columns in the dataset: {urls.columns}")
        print(urls.head())  # Display the first few rows to verify the structure

        # Check if the expected columns 'URL' and 'Label' are in the dataset
        if 'URL' not in urls.columns or 'Label' not in urls.columns:
            print("The dataset does not have the expected columns.")
            print(f"Available columns: {urls.columns}")
            return None

        # Preprocess: Create a new DataFrame with the relevant columns
        urls = urls[['URL', 'Label']]  # Adjust based on the actual column names in your dataset
        print("Data loaded and processed successfully")
        return urls

    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    
def load_and_preprocess_data_for_urls():
    # Load the URL dataset
    urls_file = './dataset/urls.csv'
    try:
        urls = pd.read_csv(urls_file)
        print(f"Columns in the dataset: {urls.columns}")
        print(urls.head())  # Display the first few rows to verify the structure

        # Check if the expected columns 'URL' and 'Label' are in the dataset
        if 'URL' not in urls.columns or 'Label' not in urls.columns:
            print("The dataset does not have the expected columns.")
            print(f"Available columns: {urls.columns}")
            return None

        # Check for missing values
        print("Checking for missing values...")
        print(urls.isnull().sum())  # Display count of missing values for each column

        # Drop rows with missing values
        urls.dropna(inplace=True)
        print("Missing values removed.")

        # Preprocess: Keep only relevant columns
        urls = urls[['URL', 'Label']]  # Adjust based on the actual column names in your dataset
        print("Data loaded and processed successfully.")
        return urls

    except Exception as e:
        print(f"Error loading data: {e}")
        return None
def load_and_preprocess_data_for_urls():
    urls_file = './dataset/urls.csv'
    try:
        urls = pd.read_csv(urls_file)
        print(f"Columns in the dataset: {urls.columns}")

        # Check for expected columns
        if 'URL' not in urls.columns or 'Label' not in urls.columns:
            raise ValueError("Dataset must have 'URL' and 'Label' columns.")

        # Drop missing values
        urls.dropna(inplace=True)

        # Map labels to integers: "good" -> 1, "bad" -> 0
        label_mapping = {'good': 1, 'bad': 0}
        urls['Label'] = urls['Label'].map(label_mapping)

        if urls['Label'].isnull().any():
            raise ValueError("Some labels could not be mapped. Check dataset for invalid values.")

        print("Data loaded and processed successfully.")
        return urls
    except Exception as e:
        print(f"Error loading data: {e}")
        return None    
    
def train_model_on_urls(urls_data):
    # Vectorize the URLs
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(urls_data['URL'])  # Using the 'URL' column as features
    y = urls_data['Label']  # Using the 'Label' column as target labels
    # Replace placeholder labels with actual labels
    y = data['label_column'].astype(int)  # Ensure labels are integers

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a logistic regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Print classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    data['Label'].value_counts()
    prediction_prob = model.predict_proba(vectorized_input)
    print("Prediction Probability:", prediction_prob) 

def main():
    # Load and preprocess data
    urls_data = load_and_preprocess_data_for_urls()

    # Ensure the data was loaded properly before proceeding
    if urls_data is not None:
        train_model_on_urls(urls_data)
    else:
        print("Data processing failed. Please check the dataset.")

from sklearn.metrics import classification_report

# After training and making predictions on test data:
y_pred = model.predict(X_test_vectorized)

# Print the classification report to evaluate performance
print(classification_report(y_test, y_pred))

# Print the class distribution to check for imbalance
print("Class distribution:", data['Label'].value_counts())

# Check if the model is giving probabilities
prediction_probs = model.predict_proba(X_test_vectorized)
print("Prediction probabilities:", prediction_probs[:5])  # Show first 5 predictions

        


def train_model_on_urls(urls_data):
    # Vectorize the URLs
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(urls_data['URL'])  # Using the 'URL' column as features
    y = urls_data['Label']  # Using the 'Label' column as target labels

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a logistic regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Print classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Interactive Testing
    while True:
        test_url = input("\nEnter a URL to classify (or type 'exit' to quit): ").strip()
        print(f"Input URL: '{test_url}'")  # Debug: Print the entered URL to confirm it's captured

        if test_url.lower() == 'exit':
            print("Exiting the program.")
            break
        
        if not test_url:
            print("Please enter a valid URL or text.")
            continue
        
        try:
            # Vectorize the input URL
            test_vector = vectorizer.transform([test_url])  # Vectorize the input URL
            print(f"Vectorized Input: {test_vector.toarray()}")  # Debug: Print vectorized input

            # Make prediction
            prediction = model.predict(test_vector)  # Predict the label
            print(f"The URL '{test_url}' is classified as: {prediction[0]}")
        except Exception as e:
            print(f"Error during prediction: {e}")

    # Interactive Testing
    while True:
        test_url = input("\nEnter a URL to classify (or type 'exit' to quit): ")
        if test_url.lower() == 'exit':
            break
        test_vector = vectorizer.transform([test_url])

def main():
    """
    Main function to load data, train model, and save model and vectorizer.
    """
    # Load and preprocess data
    urls_data = load_and_preprocess_data_for_urls()

    # Ensure the data was loaded properly before proceeding
    if urls_data is not None:
        # Train the model and vectorizer
        url_model, url_vectorizer = train_model_on_urls(urls_data)

        # Specify paths for saving
        url_model_path = "./scripts/models/url_classifier_model.joblib"
        url_vectorizer_path = "./scripts/models/url_tfidf_vectorizer.joblib"

        # Save the trained model and vectorizer
        save_url_model_and_vectorizer(url_model, url_vectorizer, url_model_path, url_vectorizer_path)
    else:
        print("Data processing failed. Please check the dataset.")

from sklearn.linear_model import LogisticRegression

# Train the model
def train_model_on_urls(urls_data):
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(urls_data['URL'])  # Vectorize the 'URL' column
    y = urls_data['Label']  # Label column

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Now the model is trained, you can use it for predictions
    y_pred = model.predict(X_test)  # Prediction with the trained model

    # Print the classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    return model, vectorizer

def main():
    # Load and preprocess data
    urls_data = load_and_preprocess_data_for_urls()

    if urls_data is not None:
        # Train the model and get the trained model and vectorizer
        model, vectorizer = train_model_on_urls(urls_data)

        # Now you can use the trained model for predictions
        y_pred = model.predict(X_test_vectorized)  # Ensure you have X_test_vectorized ready for this

        # You can also save the model here if needed
        save_model_and_vectorizer(model, vectorizer, 'model.pkl', 'vectorizer.pkl')

    else:
        print("Data processing failed. Please check the dataset.")



import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

# Define the function to save the model and vectorizer
def save_model_and_vectorizer(model, vectorizer, model_filename='model.pkl', vectorizer_filename='vectorizer.pkl'):
    """
    Save the trained model and vectorizer to disk.

    Parameters:
    - model: Trained machine learning model.
    - vectorizer: The vectorizer used for feature extraction.
    - model_filename: Filename for saving the model.
    - vectorizer_filename: Filename for saving the vectorizer.
    """
    try:
        # Save the model
        with open(model_filename, 'wb') as model_file:
            pickle.dump(model, model_file)
        print(f"Model saved as {model_filename}")

        # Save the vectorizer
        with open(vectorizer_filename, 'wb') as vectorizer_file:
            pickle.dump(vectorizer, vectorizer_file)
        print(f"Vectorizer saved as {vectorizer_filename}")
    except Exception as e:
        print(f"Error saving model and vectorizer: {e}")

import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import joblib

# Define the function to save the model and vectorizer
def save_model_and_vectorizer(model, vectorizer, model_filename='model.pkl', vectorizer_filename='vectorizer.pkl'):
    """
    Save the trained model and vectorizer to disk.

    Parameters:
    - model: Trained machine learning model.
    - vectorizer: The vectorizer used for feature extraction.
    - model_filename: Filename for saving the model.
    - vectorizer_filename: Filename for saving the vectorizer.
    """
    # Save the model
    with open(model_filename, 'wb') as model_file:
        pickle.dump(model, model_file)
    print(f"Model saved as {model_filename}")

    # Save the vectorizer
    with open(vectorizer_filename, 'wb') as vectorizer_file:
        pickle.dump(vectorizer, vectorizer_file)
    print(f"Vectorizer saved as {vectorizer_filename}")


# Example data for fitting the vectorizer and model (replace this with your actual training data)
X_train = ["your training text data here"]
y_train = ["your target labels here"]

# Create a TfidfVectorizer instance and fit it to the training data
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)

# Create a model (e.g., RandomForestClassifier) and fit it
model = RandomForestClassifier()
model.fit(X_train_tfidf, y_train)

# Now you can save the trained model and vectorizer
save_model_and_vectorizer(model, vectorizer, 'model.pkl', 'vectorizer.pkl')

# Example of defining and training the model and vectorizer

# Create a TfidfVectorizer instance
vectorizer = TfidfVectorizer()

# Example data for fitting the vectorizer and model (replace this with your actual training data)
X_train = ["your training text data here", "another example text", "and more training data"]
y_train = [0, 1, 0]  # Replace with your actual labels

# Fit the vectorizer to the training data
X_train_tfidf = vectorizer.fit_transform(X_train)

# Create a model (e.g., RandomForestClassifier) and fit it
model = RandomForestClassifier()
model.fit(X_train_tfidf, y_train)

# Now you can save the trained model and vectorizer
save_model_and_vectorizer(model, vectorizer)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

# Define the function to save the model and vectorizer
def save_model_and_vectorizer(model, vectorizer, model_filename='model.pkl', vectorizer_filename='vectorizer.pkl'):
    """
    Save the trained model and vectorizer to disk.

    Parameters:
    - model: Trained machine learning model.
    - vectorizer: The vectorizer used for feature extraction.
    - model_filename: Filename for saving the model.
    - vectorizer_filename: Filename for saving the vectorizer.
    """
    # Save the model
    with open(model_filename, 'wb') as model_file:
        pickle.dump(model, model_file)
    print(f"Model saved as {model_filename}")

    # Save the vectorizer
    with open(vectorizer_filename, 'wb') as vectorizer_file:
        pickle.dump(vectorizer, vectorizer_file)
    print(f"Vectorizer saved as {vectorizer_filename}")

# Example of defining and training the model and vectorizer

# Create a TfidfVectorizer instance
vectorizer = TfidfVectorizer()

# Example data for fitting the vectorizer and model (replace this with your actual training data)
X_train = ["your training text data here"]
y_train = ["your target labels here"]

# Fit the vectorizer to the training data
X_train_tfidf = vectorizer.fit_transform(X_train)

# Create a model (e.g., RandomForestClassifier) and fit it
model = RandomForestClassifier()
model.fit(X_train_tfidf, y_train)

# Now you can save the trained model and vectorizer
save_model_and_vectorizer(model, vectorizer)



# Run the main function
if __name__ == "__main__":
    main()
