# Iris Classification Lab — kNN vs Decision Tree

University lab exercise on **Scikit-Learn classification**.  
Two classifiers (k-Nearest Neighbors and Decision Tree) are trained on
the classic Iris dataset and their performance is compared side by side.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

The script will:

1. Load the Iris dataset (150 samples, 4 features, 3 classes).
2. Split into **train (70 %)**, **validation (15 %)**, **test (15 %)**.
3. Select the best *k* for kNN by evaluating k ∈ {3, 5, 7} on the validation set.
4. Train a Decision Tree classifier.
5. Evaluate both models on the test set (accuracy, classification report, confusion matrix).
6. Save plots to `outputs/`:
   - `confusion_matrices.png` — side-by-side confusion matrices
   - `accuracy_comparison.png` — bar chart of validation vs test accuracy

## Project Structure

```
iris_classification/
├── main.py            # Orchestrates the full pipeline
├── pipeline.py        # Reusable functions (load, train, evaluate, plot)
├── requirements.txt   # Python dependencies
├── README.md
└── outputs/           # Generated plots (created at runtime)
```
