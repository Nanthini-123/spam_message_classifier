import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

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

        # Preprocess: Create a new DataFrame with the relevant columns
        urls = urls[['URL', 'Label']]  # Adjust based on the actual column names in your dataset
        print("Data loaded and processed successfully")
        return urls

    except Exception as e:
        print(f"Error loading data: {e}")
        return None

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

    while True:
        test_url = input("\nEnter a URL to classify (or type 'exit' to quit): ").strip()
        print(f"Input URL: {test_url}")  # Debug: Print the entered URL

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

def main():
    # Load and preprocess data
    urls_data = load_and_preprocess_data_for_urls()

    # Ensure the data was loaded properly before proceeding
    if urls_data is not None:
        train_model_on_urls(urls_data)
    else:
        print("Data processing failed. Please check the dataset.")

# Run the main function
if __name__ == "__main__":
    main()
