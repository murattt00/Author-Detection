# ✍️ Author Detection & Stylometry System (NLP & ML)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![NLTK](https://img.shields.io/badge/NLP-NLTK-green.svg)](https://www.nltk.org/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-F37626.svg?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

An end-to-end Natural Language Processing (NLP) and Machine Learning system that identifies authorship of unstructured texts by analyzing linguistic patterns, stylometry, and vocabulary distributions across the **C50 (Reuters-50-50)** benchmark dataset.

---

## 📌 Project Overview

Authorship attribution is the task of identifying the author of a given document among a set of candidate authors. This project implements a comprehensive stylometric analysis and text classification pipeline:
- Ingests 50 different authors with thousands of news articles.
- Performs rigorous linguistic preprocessing (lowercasing, stopword removal, lemmatization/stemming, punct filtering).
- Extracts stylistic and semantic features via character/word n-grams and TF-IDF representations.
- Trains and benchmarks multiple supervised classifiers (Support Vector Machines, Naive Bayes, Random Forest, Logistic Regression).
- Evaluates classification performance via Confusion Matrices, Precision, Recall, and F1-Scores.

---

## ✨ Key Features

- **🧹 Deep Text Preprocessing Pipeline:** Tokenization via `nltk.tokenize`, punctuation removal, custom stopword handling, and linguistic normalization.
- **📊 Stylometric Feature Extraction:**
  - Word-level TF-IDF (Term Frequency - Inverse Document Frequency).
  - Sub-word and Character n-grams (capturing subconscious punctuation and spelling habits).
  - Lexical diversity and sentence length distribution metrics.
- **🤖 Multi-Model Benchmark:** Comparative evaluation of:
  - Linear Support Vector Machines (SVM / LinearSVC)
  - Multinomial Naive Bayes (MultinomialNB)
  - Logistic Regression (with L2 Regularization)
  - Random Forest & Ensemble trees
- **📈 Comprehensive Evaluation Metrics:** Detailed scikit-learn `classification_report`, macro/weighted F1 metrics, and Seaborn-rendered Confusion Matrices illustrating per-author attribution accuracy.

---

## 🏗️ Architecture & Pipeline

```
Raw Text Articles (C50 Dataset)
            │
            ▼
┌───────────────────────────────────────┐
│        Text Preprocessing             │
│  - Tokenization (NLTK punkt)          │
│  - Stopword Filtering                 │
│  - WordNet Lemmatization              │
└───────────────────────────────────────┘
            │
            ▼
┌───────────────────────────────────────┐
│        Feature Engineering            │
│  - TF-IDF Vectorizer (Word & Char)    │
│  - N-gram Range (1-2 words, 2-4 chars)│
│  - Document Frequency Truncation      │
└───────────────────────────────────────┘
            │
            ▼
┌───────────────────────────────────────┐
│        Model Training & Tuning        │
│  - Linear SVM / Logistic Regression   │
│  - Multinomial Naive Bayes            │
│  - Cross-Validation & Grid Search     │
└───────────────────────────────────────┘
            │
            ▼
┌───────────────────────────────────────┐
│        Evaluation & Attribution       │
│  - Confusion Matrix Visualizations    │
│  - Precision, Recall, F1 Scores       │
│  - Author Attribution Inference       │
└───────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Category | Technology | Purpose |
|---|---|---|
| **Language** | Python 3.10+ | Core language |
| **NLP** | NLTK (punkt, stopwords, wordnet) | Tokenization & text normalization |
| **Machine Learning** | Scikit-Learn | Vectorization, classifiers, cross-validation, metrics |
| **Data Processing** | NumPy, Pandas | Matrix operations, structured dataset curation |
| **Visualization** | Matplotlib, Seaborn | Confusion matrices, feature importance graphs |
| **Environment** | Jupyter Notebook | Interactive experimentation & workflow |

---

## 📂 Project Structure

```
Author-Detection/
├── author_detection.ipynb   # Main end-to-end Jupyter Notebook
├── content/                 # Dataset folder (C50 train & test articles)
│   ├── C50train/            # Training corpus partitioned by author
│   └── C50test/             # Testing corpus partitioned by author
├── .gitattributes
└── README.md                # Project documentation
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Recommended: Virtual environment (`venv` or `conda`)

### 1. Clone the Repository

```bash
git clone https://github.com/murattt00/Author-Detection.git
cd Author-Detection
```

### 2. Install Dependencies

```bash
pip install numpy pandas scikit-learn nltk matplotlib seaborn jupyter
```

### 3. Download Required NLTK Corpora

Inside a Python shell or at the top of the notebook:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

### 4. Run the Notebook

```bash
jupyter notebook author_detection.ipynb
```

---

## 📊 Results & Observations

- **Character vs. Word n-grams:** Combining word-level TF-IDF with character n-grams significantly boosts attribution accuracy for shorter texts, as character patterns capture subtle author punctuation choices.
- **Top Classifier:** Linear SVM / Logistic Regression consistently outperforms Naive Bayes on high-dimensional sparse TF-IDF feature spaces due to clear hyperplane separation across 50 distinct authors.

---