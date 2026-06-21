# BZU Student Intelligence Platform

A **bilingual (English / Arabic)** Streamlit NLP application built for **ENCS5342 — Natural Language Processing**.

The platform analyzes BZU student and community content and demonstrates a full NLP pipeline, from raw text preprocessing through classical ML and deep learning models to information retrieval and evaluation.

> **Status:** Project skeleton only. Datasets, models, and Streamlit pages are **not** implemented yet — they will be added step by step.

---

## Features (planned)

1. Course Feedback Analysis
2. Student Movements & University Decisions
3. Professor & Course Insights
4. Upload & Analyze Dataset
5. Search Engine
6. Analytics Dashboard
7. AI Student Assistant
8. Model Comparison
9. About NLP Pipeline

The UI includes a **language switcher** in the sidebar to toggle between English and Arabic.

---

## Tech stack

- **Python**
- **Streamlit** — web interface
- **scikit-learn** — TF-IDF, Logistic Regression, metrics
- **PyTorch** — Neural Network, LSTM
- **HuggingFace Transformers** — AraBERT
- **SQLite** — storage
- **TF-IDF + Cosine Similarity** — information retrieval

---

## NLP pipeline (planned)

- Dataset generation (bilingual AR/EN)
- Preprocessing & normalization
- Tokenization
- N-grams
- TF-IDF
- Word/sentence embeddings
- Logistic Regression
- Feed-forward Neural Network
- LSTM
- AraBERT
- Information retrieval (search)
- Evaluation metrics & confusion matrices

---

## Project structure

```
bzu_student_intelligence/
├── app.py                  # Streamlit entry point (placeholder)
├── requirements.txt
├── README.md
├── config/                 # Paths, constants, settings
├── data/                   # raw / processed / db (generated later)
├── src/
│   ├── data/               # Dataset generation
│   ├── preprocessing/      # Cleaning, tokenization, n-grams
│   ├── features/           # TF-IDF, embeddings
│   ├── models/             # Logistic Regression, NN, LSTM, AraBERT
│   ├── retrieval/          # TF-IDF + cosine search engine
│   ├── evaluation/         # Metrics, confusion matrices
│   ├── database/           # SQLite helpers
│   └── i18n/               # English / Arabic translations
├── pages/                  # Streamlit multipage UI (added later)
├── notebooks/              # Experiments / EDA
├── models_store/           # Trained model artifacts (generated later)
└── tests/
```

---

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt
```

## Running the app

```bash
streamlit run app.py
```

> The app is currently a placeholder and will be implemented in a later step.

---

## Course

ENCS5342 — Natural Language Processing, Birzeit University (BZU).
