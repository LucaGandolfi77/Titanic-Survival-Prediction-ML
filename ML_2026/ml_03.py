# ml_pipeline_example.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# === 1. Load data from CSV ===
# Assume your CSV file has columns: feature1, feature2, ..., featureN, target
data = pd.read_csv("iris.csv")
# Separate features (X) and target (y)
X = data.drop("Species", axis=1)
y = data["Species"]
# === 2. Split into train, validation, and test sets ===
# First split into train (70%) and temp (30%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42,
stratify=y)
# Split temp into validation (15%) and test (15%)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42,
stratify=y_temp)
print(f"Training set: {X_train.shape}")
print(f"Validation set: {X_val.shape}")
print(f"Test set: {X_test.shape}")
# === 3. Train a Decision Tree model ===
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)
# === 4. Evaluate on validation set (optional tuning step) ===
val_pred = clf.predict(X_val)
val_acc = accuracy_score(y_val, val_pred)
print("\nValidation Accuracy:", val_acc)
# === 5. Evaluate on test set ===
test_pred = clf.predict(X_test)
print("\n=== Test Set Evaluation ===")
print("Accuracy:", accuracy_score(y_test, test_pred))
print("\nClassification Report:\n", classification_report(y_test, test_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, test_pred))