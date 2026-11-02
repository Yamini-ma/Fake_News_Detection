
import pandas as pd
import re
import os

# Step 1: Define dataset paths
fake_path = "data/Fake.csv"
true_path = "data/True.csv"

# Step 2: Load datasets
print("Loading datasets...")

fake_df = pd.read_csv(fake_path)
true_df = pd.read_csv(true_path)

# Step 3: Display dataset information
print("\nFAKE NEWS DATASET")
print("Shape:", fake_df.shape)
print(fake_df.head())

print("\nREAL NEWS DATASET")
print("Shape:", true_df.shape)
print(true_df.head())

# Step 4: Add labels
fake_df["label"] = 0
true_df["label"] = 1

# Step 5: Combine datasets
news_df = pd.concat(
    [fake_df, true_df],
    ignore_index=True
)

print("\nCOMBINED DATASET")
print("Shape:", news_df.shape)

# Step 6: Check missing values
print("\nMissing values:")
print(news_df.isnull().sum())

# Step 7: Remove rows without news text
news_df.dropna(subset=["text"], inplace=True)

# Replace missing titles with empty strings
news_df["title"] = news_df["title"].fillna("")

# Step 8: Remove duplicate news articles
news_df.drop_duplicates(
    subset=["text"],
    inplace=True
)

# Step 9: Combine title and article text
news_df["content"] = (
    news_df["title"].astype(str)
    + " "
    + news_df["text"].astype(str)
)

# Step 10: Clean text
def clean_text(text):
    text = str(text)
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

news_df["content"] = news_df["content"].apply(clean_text)

# Step 11: Remove empty content
news_df = news_df[
    news_df["content"].str.strip() != ""
]

# Step 12: Keep required columns
news_df = news_df[["content", "label"]]

# Step 13: Shuffle dataset
news_df = news_df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

# Step 14: Display final dataset information
print("\nFINAL DATASET")
print("Shape:", news_df.shape)

print("\nLabel distribution:")
print(news_df["label"].value_counts())

print("\nSample records:")
print(news_df.head())

# Step 15: Save processed dataset
os.makedirs("data", exist_ok=True)

news_df.to_csv(
    "data/processed_news.csv",
    index=False
)

print("\nPreprocessing completed successfully!")
print("Saved file: data/processed_news.csv")
