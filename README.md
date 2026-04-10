
# Hate Speech Detection System

A machine learning pipeline to detect hate speech in text, built as a college NLP project.
The system classifies text into three categories: **Hate Speech**, **Offensive Language**, and **Neither**.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Libraries Used](#libraries-used)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [How to Run](#how-to-run)
- [How It Works](#how-it-works)
- [Results](#results)
- [Limitations](#limitations)

---

## Project Overview

Hate speech detection is a Natural Language Processing (NLP) task where the goal is to automatically
identify whether a piece of text contains hateful or offensive content.

This project:
1. Uses a real-world labelled Twitter dataset (~24,000 tweets)
2. Preprocesses and cleans raw tweet text
3. Converts text into numerical features using TF-IDF
4. Trains a Logistic Regression classifier
5. Evaluates using accuracy, precision, recall, F1-score, and cross-validation
6. Saves the trained model for reuse
7. Provides an interactive command-line tool to classify any text in real time

---

## Dataset

**Davidson et al. (2017) — Hate Speech and Offensive Language**

| Property | Details |
|---|---|
| Source | Twitter (collected via Twitter API) |
| Size | ~24,783 tweets |
| Labels | `0` = Hate Speech, `1` = Offensive Language, `2` = Neither |
| Class Distribution | Hate: 1,430 · Offensive: 19,190 · Neither: 4,163 |
| Paper | [Thomas Davidson et al., 2017](https://arxiv.org/abs/1703.04009) |
| GitHub | [t-davidson/hate-speech-and-offensive-language](https://github.com/t-davidson/hate-speech-and-offensive-language) |

> The dataset is **downloaded automatically** when you run `train.py` for the first time.
> No manual download needed.

**Note on Class Imbalance:**
The dataset is heavily skewed — 77% of samples are "Offensive". To handle this,
we use `class_weight='balanced'` which makes the model pay more attention to
under-represented classes (especially "Hate Speech").

---

## Libraries Used

### `pandas`
**What it is:** A data manipulation and analysis library.  
**Why we use it:** To load the CSV dataset, inspect columns, filter rows, rename columns,
drop null values, and prepare the text and label columns for training.  
```python
import pandas as pd
df = pd.read_csv("data/hate_speech.csv")
```

---

### `scikit-learn`
**What it is:** The most widely used Python library for classical machine learning.  
**Why we use it:** We use multiple components from scikit-learn:

| Component | Purpose |
|---|---|
| `TfidfVectorizer` | Converts cleaned text into numerical TF-IDF feature vectors |
| `LogisticRegression` | The classification model that predicts hate/offensive/neither |
| `train_test_split` | Splits the dataset into training (80%) and test (20%) subsets |
| `cross_val_score` | Runs 5-fold cross-validation for a more reliable F1 estimate |
| `make_pipeline` | Chains the vectorizer + model together for clean cross-validation |
| `classification_report` | Prints per-class precision, recall, and F1-score |
| `confusion_matrix` | Shows how many predictions were correct vs misclassified per class |
| `accuracy_score` | Computes overall accuracy on the test set |

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
```

---

### `nltk` (Natural Language Toolkit)
**What it is:** A classic NLP library for text processing in Python.  
**Why we use it:** Specifically used to load the **English stopwords list** — common words
like "the", "is", "and", "a" that carry no meaningful signal for hate speech detection.
Removing them reduces noise and makes the TF-IDF features more informative.  
```python
from nltk.corpus import stopwords
STOP_WORDS = set(stopwords.words("english"))
```

---

### `matplotlib`
**What it is:** The foundational Python library for creating static plots and charts.  
**Why we use it:** Used as the backend rendering engine for all visualizations.
`seaborn` builds on top of `matplotlib`, so it must be installed alongside it.  
```python
import matplotlib.pyplot as plt
plt.savefig("outputs/confusion_matrix.png")
```

---

### `seaborn`
**What it is:** A high-level statistical data visualization library built on top of `matplotlib`.  
**Why we use it:** Used to create two key charts:
- **Bar chart** — visualizes the class distribution (how many hate/offensive/neither samples exist)
- **Heatmap** — visualizes the confusion matrix to see where the model makes mistakes

```python
import seaborn as sns
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
```

---

### `requests`
**What it is:** A simple HTTP library for making web requests in Python.  
**Why we use it:** To download the dataset CSV file directly from GitHub programmatically,
so users don't need to manually download and place any files.  
```python
import requests
response = requests.get(DATASET_URL, timeout=30)
```

---

### `pickle` (built-in)
**What it is:** Python's built-in object serialization module (part of the standard library, no install needed).  
**Why we use it:** To save the trained model and TF-IDF vectorizer to a `.pkl` file after training.
This means `predict.py` can load the saved model instantly without retraining from scratch every time.  
```python
import pickle
pickle.dump({"model": model, "vectorizer": vectorizer}, f)
```

---

### `re` (built-in)
**What it is:** Python's built-in regular expressions module.  
**Why we use it:** In `preprocess.py`, `re` is used to clean raw tweet text — remove URLs,
`@mentions`, `#hashtags`, punctuation, numbers, and extra whitespace using pattern matching.  
```python
import re
text = re.sub(r"http\S+|www\S+", "", text)  # Remove URLs
text = re.sub(r"@\w+", "", text)            # Remove @mentions
```

---

## Project Structure

```
nlp-pbl/
├── data/
│   └── hate_speech.csv          # Auto-downloaded on first run (~24K tweets)
├── src/
│   ├── preprocess.py            # Text cleaning functions (URLs, mentions, stopwords)
│   ├── train.py                 # Full ML pipeline: load → clean → train → evaluate → save
│   └── predict.py               # Interactive classifier: type text, get a label
├── outputs/
│   ├── class_distribution.png   # Bar chart of label distribution
│   ├── confusion_matrix.png     # Heatmap of model predictions vs true labels
│   └── model.pkl                # Saved model + vectorizer (generated after training)
├── requirements.txt             # All pip dependencies
└── README.md                    # This file
```

---

## Setup & Installation

**Prerequisites:** Python 3.10 or higher

**Step 1 — Clone or navigate to the project folder:**
```bash
cd c:\Users\NABI\nlp-pbl
```

**Step 2 — Install all dependencies:**
```bash
pip install -r requirements.txt
```

This installs: `pandas`, `scikit-learn`, `nltk`, `matplotlib`, `seaborn`, `requests`

---

## How to Run

### Train the Model
```bash
python src/train.py
```

This will:
1. Download the dataset automatically (first run only)
2. Clean and preprocess all tweet text
3. Split data: 80% train / 20% test (stratified)
4. Vectorize text with TF-IDF (unigrams + bigrams, 15,000 features)
5. Train Logistic Regression with balanced class weights
6. Run 5-fold cross-validation and print macro F1 scores
7. Evaluate on test set — print accuracy + classification report
8. Save confusion matrix and class distribution charts to `outputs/`
9. Save trained model to `outputs/model.pkl`

---

### Interactive Predictor
```bash
python src/predict.py
```

Type any sentence and the model will predict:
- `[!]  Hate Speech`
- `[~]  Offensive Language`
- `[OK] Neither`

Along with a confidence score for each class. Type `quit` to exit.

**Example:**
```
Enter text: I hope everyone has a wonderful day

  Prediction  : [OK] Neither
  Confidence  :
    Hate Speech           3.1%  #
    Offensive            11.4%  ###
    Neither              85.5%  #########################
```

---

## How It Works

### 1. Text Preprocessing (`preprocess.py`)
Raw tweets contain noise that hurts model accuracy:
- URLs like `http://example.com` → removed
- @mentions like `@john` → removed
- Hashtags like `#hate` → the `#` is removed, word kept
- Punctuation and numbers → removed
- Common stopwords ("the", "is", "a") → removed
- Everything lowercased

### 2. TF-IDF Vectorization
TF-IDF (Term Frequency–Inverse Document Frequency) converts text into numbers.
- **TF** — how often a word appears in the current tweet
- **IDF** — penalizes words that appear in almost every tweet (common words)
- Result: each tweet becomes a vector of 15,000 numbers

We use **bigrams (1,2)** meaning the model also considers two-word combinations
like "hate speech", "you idiot" as single features — this captures context better
than individual words alone.

### 3. Logistic Regression
A simple but powerful linear classifier that works very well for text classification.
- `class_weight='balanced'` — automatically adjusts weights so rare classes
  (Hate Speech: 1,430 samples) are not overwhelmed by dominant classes (Offensive: 19,190 samples)
- `max_iter=1000` — enough iterations for the solver to converge on this dataset

### 4. Cross-Validation
Instead of evaluating on just one 80/20 split, we use **5-fold cross-validation**:
the data is split into 5 equal parts, the model is trained 5 times (each time
using a different part as the test set), and the results are averaged.
This gives a more reliable and honest estimate of real-world performance.

---

## Results

| Metric | Value |
|---|---|
| Overall Accuracy | ~89% |
| Macro F1 (CV mean) | ~0.72–0.75 |
| Hate Speech F1 | ~0.55–0.65 (improved with balanced weights) |
| Offensive F1 | ~0.94 |
| Neither F1 | ~0.81 |

> **Why is Hate Speech F1 lower?**  
> The dataset has only 1,430 hate speech samples out of 24,783 total (~5.8%).
> This class imbalance makes it harder for any model to learn hate speech patterns.
> Using `class_weight='balanced'` significantly improves recall for this class.

---

## Limitations

1. **Dataset bias** — The dataset only contains English tweets from 2017. It may not generalize well to modern slang, other platforms, or other languages.
2. **Context blindness** — TF-IDF looks at individual words, not full sentence context. Sarcasm and coded language can fool it.
3. **Label ambiguity** — The boundary between "hate speech" and "offensive language" is often subjective.
4. **No deep learning** — A BERT-based model would likely improve Hate Speech F1 to 0.80+ but requires significantly more compute.
>>>>>>> ae2eca8 (Initial Commit)
