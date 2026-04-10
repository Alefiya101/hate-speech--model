"""
predict.py
----------
Interactive hate speech classifier.
Loads the saved model and lets you type sentences to classify.

Run:
    python src/predict.py

Requirements:
    - Run train.py first to generate outputs/model.pkl
"""

import os
import sys
import pickle

# Allow importing from src/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess import clean_text

MODEL_PATH  = os.path.join("outputs", "model.pkl")
LABEL_NAMES = {0: "Hate Speech", 1: "Offensive", 2: "Neither"}
LABEL_EMOJI = {0: "[!]", 1: "[~]", 2: "[OK]"}


def load_model(path: str):
    """Load model and vectorizer from pickle file."""
    if not os.path.exists(path):
        print(f"[ERROR] Model not found at '{path}'.")
        print("[ERROR] Please run 'python src/train.py' first to train and save the model.")
        sys.exit(1)
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["model"], data["vectorizer"]


def predict(text: str, model, vectorizer) -> tuple:
    """Clean text, vectorize, and return (label_int, label_name)."""
    cleaned = clean_text(text)
    vec     = vectorizer.transform([cleaned])
    label   = model.predict(vec)[0]
    proba   = model.predict_proba(vec)[0]
    return label, LABEL_NAMES[label], proba


def main():
    print("\n================================================")
    print("   Hate Speech Detector — Interactive Mode")
    print("================================================")
    print("Loading model...")

    model, vectorizer = load_model(MODEL_PATH)
    print("Model loaded successfully!")
    print("\nType a sentence to classify it.")
    print("Type 'quit' or 'exit' to stop.\n")
    print("Labels:")
    print("  [!]  -> Hate Speech")
    print("  [~]  -> Offensive Language")
    print("  [OK] -> Neither (normal speech)")
    print("------------------------------------------------\n")

    while True:
        try:
            text = input("Enter text: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[INFO] Exiting. Goodbye!")
            break

        if not text:
            print("[WARN] Please enter some text.\n")
            continue

        if text.lower() in ("quit", "exit"):
            print("[INFO] Exiting. Goodbye!")
            break

        label_int, label_name, proba = predict(text, model, vectorizer)
        emoji = LABEL_EMOJI[label_int]

        print(f"\n  Prediction  : {emoji} {label_name}")
        print(f"  Confidence  :")
        for i, name in LABEL_NAMES.items():
            bar = "#" * int(proba[i] * 30)
            print(f"    {name:<20} {proba[i]*100:5.1f}%  {bar}")
        print()


if __name__ == "__main__":
    main()
