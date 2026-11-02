
import pandas as pd
import numpy as np
import os
import json
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# Step 1: Load processed dataset
print("Loading processed dataset...")

df = pd.read_csv("data/processed_news.csv")

df = df.dropna(subset=["content", "label"])

X = df["content"].astype(str)
y = df["label"].astype(int)

print("Dataset shape:", df.shape)
print("\nLabel distribution:")
print(y.value_counts())

# Step 2: Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# Step 3: Define TF-IDF and machine learning models
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ),

    "Passive Aggressive": PassiveAggressiveClassifier(
        max_iter=1000,
        random_state=42,
        class_weight="balanced",
        tol=1e-3
    )
}

results = []
trained_models = {}

# Step 4: Train and evaluate both models
for name, classifier in models.items():

    print("\nTraining:", name)

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=50000,
            ngram_range=(1, 2),
            stop_words="english",
            min_df=2,
            max_df=0.95,
            sublinear_tf=True
        )),

        ("classifier", classifier)
    ])

    pipeline.fit(X_train, y_train)

    # Predictions
    predictions = pipeline.predict(X_test)

    # Evaluation metrics
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, zero_division=0
    )
    recall = recall_score(
        y_test, predictions, zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, zero_division=0
    )

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        target_names=["FAKE", "REAL"],
        zero_division=0
    ))

    print("Accuracy:", round(accuracy * 100, 2), "%")
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1 Score:", round(f1, 4))

    results.append({
        "Model": name,
        "Accuracy": round(accuracy, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1 Score": round(f1, 4)
    })

    trained_models[name] = {
        "pipeline": pipeline,
        "predictions": predictions
    }

# Step 5: Compare model performance
results_df = pd.DataFrame(results)

print("\nMODEL COMPARISON")
print(results_df.to_string(index=False))

# Step 6: Select the best model using F1 Score
best_model_name = results_df.sort_values(
    by=["F1 Score", "Accuracy"],
    ascending=False
).iloc[0]["Model"]

best_pipeline = trained_models[best_model_name]["pipeline"]
best_predictions = trained_models[best_model_name]["predictions"]

print("\nBest Model:", best_model_name)

# Step 7: Create output folder
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# Step 8: Save best trained pipeline
joblib.dump(best_pipeline, "models/best_model.joblib")

# Save comparison results
results_df.to_csv(
    "results/model_comparison.csv",
    index=False
)

with open("results/model_metrics.json", "w") as file:
    json.dump(results, file, indent=4)

# Step 9: Generate confusion matrix
cm = confusion_matrix(y_test, best_predictions)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["FAKE", "REAL"],
    yticklabels=["FAKE", "REAL"]
)

plt.title("Confusion Matrix - " + best_model_name)
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.tight_layout()

plt.savefig(
    "results/confusion_matrix.png",
    dpi=300
)

plt.close()

# Step 10: Display final result
print("\nModel training completed successfully!")
print("Best model:", best_model_name)
print("Saved model: models/best_model.joblib")
print("Saved comparison: results/model_comparison.csv")
print("Saved graph: results/confusion_matrix.png")
