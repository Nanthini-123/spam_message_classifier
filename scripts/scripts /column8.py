import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib
import pickle


def load_data():
    """Load and preprocess the dataset."""
    dataset_path = './dataset/messages.csv'  # Update the path if necessary

    # Check if the file exists
    if not os.path.exists(dataset_path):
        print(f"Dataset file not found at {dataset_path}. Please check the path.")
        exit()

    try:
        print("Loading dataset...")
        messages = pd.read_csv(dataset_path, encoding="ISO-8859-1")
        print("Dataset loaded successfully.")

        # Keep only relevant columns
        messages = messages[['v1', 'v2']]
        print(f"Columns kept for processing: {messages.columns.tolist()}")

        # Rename columns for clarity
        messages.rename(columns={'v1': 'label', 'v2': 'message'}, inplace=True)

        # Convert labels to binary (ham=0, spam=1)
        messages['label'] = messages['label'].map({'ham': 0, 'spam': 1})

        # Convert messages to lowercase
        messages['message'] = messages['message'].str.lower()

        # Drop rows with missing values
        messages.dropna(inplace=True)
        print("Data preprocessing completed successfully.")
        print(messages.head())  # Display the first few rows for verification

        return messages

    except Exception as e:
        print(f"Error loading or preprocessing dataset: {e}")
        exit()

def train_model():
    """Train and evaluate a spam classification model."""
    # Load preprocessed data
    data = load_data()

    # Separate features and labels
    X = data['message']
    y = data['label']

    # Split data into training and testing sets
    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set size: {len(X_train)}, Testing set size: {len(X_test)}")

    # Transform text data into numerical features using TF-IDF
    print("Transforming text data using TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=5000)  # Limit to top 5000 features
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    print("TF-IDF transformation completed.")

    # Train a Logistic Regression model
    print("Training the Logistic Regression model...")
    model = LogisticRegression(random_state=42)
    model.fit(X_train_tfidf, y_train)
    print("Model training completed.")

    # Evaluate the model
    print("Evaluating the model...")
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\nModel Evaluation Metrics:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return model, vectorizer
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

def clean_message_dataset(file_path):
    """
    Load and preprocess the message dataset for training.

    Args:
        file_path (str): Path to the message dataset file.

    Returns:
        pd.DataFrame: Cleaned dataset ready for training.
    """
    try:
        # Load dataset
        print(f"Loading dataset from {file_path}...")
        data = pd.read_csv(file_path, encoding="ISO-8859-1")
        print(f"Dataset loaded. Columns: {list(data.columns)}")

        # Keep only necessary columns
        if 'v1' not in data.columns or 'v2' not in data.columns:
            raise ValueError("Dataset must have 'v1' for labels and 'v2' for messages.")
        data = data[['v1', 'v2']]
        print(f"Columns kept for processing: {list(data.columns)}")

        # Rename columns for clarity
        data.rename(columns={'v1': 'label', 'v2': 'message'}, inplace=True)

        # Map labels to binary (ham -> 0, spam -> 1)
        data['label'] = data['label'].map({'ham': 0, 'spam': 1})
        if data['label'].isnull().any():
            raise ValueError("Unexpected values in 'label' column. Check your dataset.")

        # Drop rows with missing or invalid data
        print("Checking for missing values...")
        data.dropna(subset=['label', 'message'], inplace=True)

        # Convert messages to lowercase
        print("Converting messages to lowercase...")
        data['message'] = data['message'].str.lower()

        # Remove extra spaces from messages
        print("Removing extra spaces...")
        data['message'] = data['message'].str.strip()

        # Drop duplicate messages
        print("Dropping duplicate messages...")
        data.drop_duplicates(subset=['message'], inplace=True)

        print(f"Data cleaning completed. Remaining rows: {len(data)}")
        print(data.head())  # Display first few rows of the cleaned dataset

        return data

    except Exception as e:
        print(f"Error while cleaning the dataset: {e}")
        return None

# Example usage
if __name__ == "__main__":
    cleaned_data = clean_message_dataset('./dataset/messages.csv')
    if cleaned_data is not None:
        # Save the cleaned dataset for future use
        cleaned_data.to_csv('./dataset/cleaned_messages.csv', index=False)
        print("Cleaned dataset saved to './dataset/cleaned_messages.csv'.")

def main():
    print("Starting the spam message classifier training...")
    
    # Train the model and vectorizer
    model, vectorizer = train_model()
    print("\nModel and vectorizer are ready for use.")

    # Save the model and vectorizer
    save_model_and_vectorizer(
        model,
        vectorizer,
        "./models/spam_classifier_model.joblib",
        "./models/tfidf_vectorizer.joblib"
    )
    print("Model and vectorizer have been saved successfully.")

    # Interactive message classification
    while True:
        user_input = input("Enter a message to classify (or type 'exit' to quit): ").strip()
        if user_input.lower() == "exit":
            print("Exiting the classifier. Goodbye!")
            break

        if user_input:
            result = classify_message(model, vectorizer, user_input)
            print(f"The message '{user_input}' is classified as: {result}")


if __name__ == "__main__":
    main()        

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

# Example of how to call this function
def main():
    # Assuming 'model' and 'vectorizer' are already trained and initialized
    model = ...  # your trained model here (e.g., a classifier)
    vectorizer = ...  # your vectorizer here (e.g., CountVectorizer or TfidfVectorizer)

    # Save both the model and the vectorizer
    save_model_and_vectorizer(model, vectorizer)

# Run the main function
if __name__ == "__main__":
    main()

def main():
    print("Starting the spam message classifier training...")
    
    # Train the model and vectorizer
    model, vectorizer = train_model()
    print("\nModel and vectorizer are ready for use.")

    # Save the model and vectorizer
    save_model_and_vectorizer(
        model,
        vectorizer,
        "./models/spam_classifier_model.joblib",
        "./models/tfidf_vectorizer.joblib"
    )
    print("Model and vectorizer have been saved successfully.")

    # Interactive message classification
    while True:
        user_input = input("Enter a message to classify (or type 'exit' to quit): ").strip()
        if user_input.lower() == "exit":
            print("Exiting the classifier. Goodbye!")
            break

        if user_input:
            result = classify_message(model, vectorizer, user_input)
            print(f"The message '{user_input}' is classified as: {result}")
def classify_message(model, vectorizer, message):
    """Classify a single message as spam or ham."""
    transformed_message = vectorizer.transform([message.lower()])
    prediction = model.predict(transformed_message)[0]
    return "spam" if prediction == 1 else "ham"


def main():
    print("Starting the spam message classifier training...")
    model, vectorizer = train_model()
    print("\nModel and vectorizer are ready for use.")

    while True:
        user_input = input("Enter a message to classify (or type 'exit' to quit): ").strip()
        if user_input.lower() == "exit":
            print("Exiting the classifier. Goodbye!")
            break

        if user_input:
            result = classify_message(model, vectorizer, user_input)
            print(f"The message '{user_input}' is classified as: {result}")


    # Transform the input message and make a prediction
    message_tfidf = vectorizer.transform([message.lower()])
    prediction = model.predict(message_tfidf)[0]
    result = "Spam" if prediction == 1 else "Ham"
    print(f"The message '{message}' is classified as: {result}")

if __name__ == "__main__":
    main()
