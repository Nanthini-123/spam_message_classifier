import pandas as pd

file_path = "/Users/nanthinik/Desktop/spam_message_classifier/scripts/dataset/cleaned_messages.csv"
df = pd.read_csv(file_path)

if df.empty:
    print("⚠️ Error: The dataset is empty!")
else:
    print(f"✅ Dataset loaded successfully! {df.shape[0]} rows, {df.shape[1]} columns")
    print(df.head())
