import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# === 1. Load the Iris dataset ===
iris = load_iris()
X = iris.data      # Feature matrix (150 samples, 4 features)
y = iris.target    # Target vector (0: Setosa, 1: Versicolor, 2: Virginica)

# === 2. Split into train, validation, and test sets
# First split into train (70%) and temp (30%)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Split temp into validation (15%) and test (15%)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"Training set: {X_train.shape}")
print(f"Validation set: {X_val.shape}")
print(f"Test set: {X_test.shape}\n")

# === 3. Initialize & Train Models ===
dt_classifier = DecisionTreeClassifier(random_state=42)
knn_classifier = KNeighborsClassifier(n_neighbors=3)

dt_classifier.fit(X_train, y_train)
knn_classifier.fit(X_train, y_train)

# === 4. Evaluation Function (Reusable for clarity) ===
def evaluate_model(y_true, y_pred, model_name):
    acc = accuracy_score(y_true, y_pred)
    print(f"📊 {model_name} - Accuracy: {acc:.4f}")
    print("Classification Report:\n", classification_report(y_true, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print("-" * 40)

# === 5. Evaluate on Validation Set ===
print("=== VALIDATION SET EVALUATION ===")
evaluate_model(y_val, dt_classifier.predict(X_val), "Decision Tree (Validation)")
evaluate_model(y_val, knn_classifier.predict(X_val), "KNN (Validation)")

# === 6. Evaluate on Test Set ===
print("\n=== TEST SET EVALUATION ===")
evaluate_model(y_test, dt_classifier.predict(X_test), "Decision Tree (Test)")
evaluate_model(y_test, knn_classifier.predict(X_test), "KNN (Test)")

# === 7. Quick Comparison Summary ===
dt_test_acc = accuracy_score(y_test, dt_classifier.predict(X_test))
knn_test_acc = accuracy_score(y_test, knn_classifier.predict(X_test))

print("\n=== MODEL COMPARISON ===")
if dt_test_acc > knn_test_acc:
    print(f"Decision Tree outperformed KNN ({dt_test_acc:.4f} vs {knn_test_acc:.4f})")
elif knn_test_acc > dt_test_acc:
    print(f"KNN outperformed Decision Tree ({knn_test_acc:.4f} vs {dt_test_acc:.4f})")
else:
    print("Both models identical accuracy on the test set.")
