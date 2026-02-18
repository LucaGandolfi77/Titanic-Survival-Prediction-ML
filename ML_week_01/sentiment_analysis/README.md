# 💬 Sentiment Analysis — From TF-IDF to DistilBERT

A complete NLP pipeline comparing classical ML, word embeddings, and
transformer fine-tuning on Amazon product reviews.

## Architecture

```
Raw Text → Clean → TF-IDF → LogReg       → ~93% acc  (Stage 1)
                 → GloVe  → XGBoost      → ~90% acc  (Stage 2)
         → Tokenize → DistilBERT → Fine-tune → ~95% acc  (Stage 3)
```

## Dataset

**Amazon Polarity** (HuggingFace `datasets`):
- Binary: 0 = negative, 1 = positive
- Train: 20,000 (stratified 50/50)
- Test:   4,000 (stratified 50/50)

## Quick Start

```bash
pip install -r requirements.txt
jupyter notebook sentiment_analysis.ipynb
# Run All Cells — first run downloads data (~600 MB, cached thereafter)
```

## Stage Comparison (approximate)

| Model                         | Stage | Accuracy | F1    | Train Time |
|-------------------------------|-------|----------|-------|------------|
| Logistic Regression (TF-IDF) | 1     | ~93%     | ~0.93 | < 5 s      |
| Naive Bayes (BoW)            | 1     | ~91%     | ~0.91 | < 1 s      |
| SVM Linear (TF-IDF)          | 1     | ~93%     | ~0.93 | < 10 s     |
| SGD Classifier (TF-IDF)      | 1     | ~92%     | ~0.92 | < 2 s      |
| Logistic Regression (GloVe)  | 2     | ~88%     | ~0.88 | < 5 s      |
| XGBoost (GloVe)              | 2     | ~90%     | ~0.90 | ~30 s      |
| DistilBERT fine-tuned        | 3     | ~95%     | ~0.95 | ~15 min    |

## Project Structure

```
sentiment_analysis/
├── sentiment_analysis.ipynb        # Main notebook (Run All Cells)
├── data/
│   ├── amazon_train.csv            # 20k reviews (saved at runtime)
│   └── amazon_test.csv             # 4k reviews
├── embeddings/
│   └── glove.6B.100d.txt           # Downloaded at runtime (~850 MB)
├── outputs/
│   ├── figures/                    # All EDA and evaluation plots
│   ├── models/
│   │   ├── tfidf_vectorizer.pkl
│   │   ├── logreg_model.pkl
│   │   └── distilbert_finetuned/   # HuggingFace save_pretrained
│   └── results/
│       ├── stage_comparison.csv
│       └── inference_demo.html
├── requirements.txt
└── README.md
```

## Hardware

Optimised for **Apple Silicon M1** (MPS backend).
Falls back to CPU gracefully if MPS is unavailable.
