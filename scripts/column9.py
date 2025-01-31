import os
import pandas as pd
import pickle
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report


def load_data():
    """Load and preprocess the dataset."""
    dataset_path = './dataset/messages.csv'  # Update the path if necessary

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


def save_model_and_vectorizer(model, vectorizer, model_filename, vectorizer_filename):
    """Save the trained model and vectorizer to disk."""
    joblib.dump(model, model_filename)
    print(f"Model saved as {model_filename}")

    joblib.dump(vectorizer, vectorizer_filename)
    print(f"Vectorizer saved as {vectorizer_filename}")


def classify_message(model, vectorizer, message):
    """Classify a single message as spam or ham."""
    transformed_message = vectorizer.transform([message.lower()])
    prediction = model.predict(transformed_message)[0]
    return "Spam" if prediction == 1 else "Ham"


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
