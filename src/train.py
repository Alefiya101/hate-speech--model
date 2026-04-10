"""
train.py
--------
Full pipeline for hate speech detection:
  1. Download dataset
  2. Clean text
  3. Split data (80/20 stratified)
  4. TF-IDF vectorization
  5. Train Logistic Regression (class_weight='balanced')
  6. Cross-validation (5-fold, macro F1)
  7. Evaluate on test set + save plots
  8. Save model + vectorizer to pickle

Run:
    python src/train.py
"""

import os
import sys
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import pickle

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Allow importing from src/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess import clean_text

# ── Config ──────────────────────────────────────────────────────────────────
DATASET_URL = (
    "https://raw.githubusercontent.com/t-davidson/hate-speech-and-offensive-language"
    "/master/data/labeled_data.csv"
)
DATA_PATH   = os.path.join("data", "hate_speech.csv")
OUTPUT_DIR  = "outputs"

LABEL_NAMES = {0: "Hate Speech", 1: "Offensive", 2: "Neither"}

# ── Helpers ──────────────────────────────────────────────────────────────────
def download_dataset(url: str, save_path: str) -> pd.DataFrame:
    """Download the dataset CSV if not already present."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if os.path.exists(save_path):
        print(f"[INFO] Dataset already exists at '{save_path}'. Loading...")
    else:
        print(f"[INFO] Downloading dataset from GitHub...")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        with open(save_path, "wb") as f:
            f.write(response.content)
        print(f"[INFO] Dataset saved to '{save_path}'.")

    df = pd.read_csv(save_path)
    return df


def plot_class_distribution(df: pd.DataFrame, label_col: str, output_dir: str):
    """Bar chart of class distribution."""
    counts = df[label_col].value_counts().sort_index()
    labels = [LABEL_NAMES[i] for i in counts.index]

    plt.figure(figsize=(7, 4))
    sns.barplot(x=labels, y=counts.values, palette="viridis")
    plt.title("Class Distribution in Dataset")
    plt.xlabel("Class")
    plt.ylabel("Number of Samples")
    plt.tight_layout()

    path = os.path.join(output_dir, "class_distribution.png")
    plt.savefig(path)
    plt.close()
    print(f"[INFO] Class distribution chart saved to '{path}'.")


def plot_confusion_matrix(cm, output_dir: str):
    """Heatmap of the confusion matrix."""
    labels = list(LABEL_NAMES.values())

    plt.figure(figsize=(7, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()

    path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(path)
    plt.close()
    print(f"[INFO] Confusion matrix saved to '{path}'.")


# ── Main Pipeline ─────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load dataset ──────────────────────────────────────────────────────────
    print("\n-- Step 1: Loading Dataset ---------------------------------")
    df = download_dataset(DATASET_URL, DATA_PATH)
    print(f"[INFO] Total samples: {len(df)}")
    print(f"[INFO] Columns: {list(df.columns)}")

    # The dataset column for labels is named 'class'
    df = df[["tweet", "class"]].rename(columns={"class": "label"})
    df.dropna(inplace=True)

    print("\n[INFO] Raw class distribution:")
    for label, count in df["label"].value_counts().sort_index().items():
        print(f"       {LABEL_NAMES[label]}: {count} samples")

    plot_class_distribution(df, "label", OUTPUT_DIR)

    # 2. Preprocess text ───────────────────────────────────────────────────────
    print("\n-- Step 2: Cleaning Text -----------------------------------")
    df["clean_tweet"] = df["tweet"].apply(clean_text)
    print(f"[INFO] Sample before cleaning: {df['tweet'].iloc[0]}")
    print(f"[INFO] Sample after  cleaning: {df['clean_tweet'].iloc[0]}")

    # 3. Train/Test split ──────────────────────────────────────────────────────
    print("\n-- Step 3: Splitting Data (80/20 stratified) ---------------")
    X = df["clean_tweet"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[INFO] Train samples: {len(X_train)}")
    print(f"[INFO] Test  samples: {len(X_test)}")

    # 4. TF-IDF Vectorization ──────────────────────────────────────────────────
    print("\n-- Step 4: TF-IDF Vectorization ----------------------------")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),   # unigrams + bigrams
        max_features=15000,
        sublinear_tf=True,    # apply log normalization
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec  = vectorizer.transform(X_test)
    print(f"[INFO] Vocabulary size: {len(vectorizer.vocabulary_)}")

    # 5. Train Logistic Regression (balanced weights) ─────────────────────────
    print("\n-- Step 5: Training Logistic Regression (balanced) ---------")
    model = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    model.fit(X_train_vec, y_train)
    print("[INFO] Training complete.")

    # 6. Cross-Validation (5-fold, macro F1) ───────────────────────────────────
    print("\n-- Step 6: Cross-Validation (5-fold) -----------------------")
    # Build a pipeline so CV handles vectorization per fold correctly
    cv_pipeline = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), max_features=15000, sublinear_tf=True),
        LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
    )
    X_all = df["clean_tweet"]
    y_all = df["label"]
    cv_scores = cross_val_score(cv_pipeline, X_all, y_all, cv=5, scoring="f1_macro", n_jobs=-1)
    print(f"[RESULT] CV Macro F1 scores : {[round(s, 3) for s in cv_scores]}")
    print(f"[RESULT] CV Mean Macro F1   : {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

    # 7. Evaluate on held-out test set ────────────────────────────────────────
    print("\n-- Step 7: Evaluation on Test Set --------------------------")
    y_pred = model.predict(X_test_vec)

    acc = accuracy_score(y_test, y_pred)
    print(f"\n[RESULT] Accuracy: {acc * 100:.2f}%\n")

    print("[RESULT] Classification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=list(LABEL_NAMES.values()),
        )
    )

    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, OUTPUT_DIR)

    # 8. Save model + vectorizer to pickle ────────────────────────────────────
    print("\n-- Step 8: Saving Model ------------------------------------")
    model_path = os.path.join(OUTPUT_DIR, "model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump({"model": model, "vectorizer": vectorizer}, f)
    print(f"[INFO] Model and vectorizer saved to '{model_path}'.")
    print("[INFO] Load with: pickle.load(open('outputs/model.pkl', 'rb'))")

    print("\n-- Done! ---------------------------------------------------")
    print("[INFO] Outputs saved in the 'outputs/' folder.")
    print("[INFO] Run 'python src/predict.py' to classify your own text.\n")


if __name__ == "__main__":
    main()
