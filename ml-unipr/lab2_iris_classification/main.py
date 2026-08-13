"""main.py — Iris Classification Lab: kNN vs Decision Tree."""

from __future__ import annotations

import os

from pipeline import (
    load_and_split,
    select_best_knn,
    train_decision_tree,
    evaluate_model,
    plot_confusion_matrices,
    plot_accuracy_comparison,
)


def main() -> None:
    os.makedirs("outputs", exist_ok=True)

    # ==============================================================
    # 1. Load & Split
    # ==============================================================
    print("=" * 60)
    print("1. Loading Iris Dataset & Splitting")
    print("=" * 60)

    data = load_and_split()
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]
    target_names = data["target_names"]

    print(f"  Features : {data['feature_names']}")
    print(f"  Classes  : {target_names}")
    print(f"  Train    : {X_train.shape}")
    print(f"  Validation: {X_val.shape}")
    print(f"  Test     : {X_test.shape}")

    # ==============================================================
    # 2. kNN — Hyperparameter Selection
    # ==============================================================
    print("\n" + "=" * 60)
    print("2. k-Nearest Neighbors — Validation")
    print("=" * 60)

    best_knn, knn_val_results = select_best_knn(
        X_train, y_train, X_val, y_val, k_values=[3, 5, 7],
    )

    for k, acc in knn_val_results.items():
        print(f"  k={k}  val accuracy = {acc:.2%}")

    best_k = best_knn.n_neighbors
    print(f"\n  ★ Best k = {best_k} "
          f"(val accuracy = {knn_val_results[best_k]:.2%})")

    # ==============================================================
    # 3. Decision Tree
    # ==============================================================
    print("\n" + "=" * 60)
    print("3. Decision Tree — Training")
    print("=" * 60)

    dt = train_decision_tree(X_train, y_train)
    dt_val_acc = evaluate_model(dt, X_val, y_val, target_names, "Validation")[0]
    print(f"  Validation accuracy = {dt_val_acc:.2%}")

    # ==============================================================
    # 4. Test-Set Evaluation
    # ==============================================================
    print("\n" + "=" * 60)
    print("4. Test-Set Evaluation")
    print("=" * 60)

    # kNN
    knn_test_acc, knn_report, knn_cm = evaluate_model(
        best_knn, X_test, y_test, target_names,
    )
    print(f"\n  --- kNN (k={best_k}) ---")
    print(f"  Accuracy: {knn_test_acc:.2%}")
    print(f"\n{knn_report}")

    # Decision Tree
    dt_test_acc, dt_report, dt_cm = evaluate_model(
        dt, X_test, y_test, target_names,
    )
    print(f"\n  --- Decision Tree ---")
    print(f"  Accuracy: {dt_test_acc:.2%}")
    print(f"\n{dt_report}")

    # ==============================================================
    # 5. Visualisation
    # ==============================================================
    print("=" * 60)
    print("5. Plots")
    print("=" * 60)

    plot_confusion_matrices(knn_cm, dt_cm, target_names)

    knn_val_acc = knn_val_results[best_k]
    plot_accuracy_comparison(knn_val_acc, knn_test_acc,
                             dt_val_acc, dt_test_acc)

    # ==============================================================
    # Summary
    # ==============================================================
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    print(f"  {'Model':<18} {'Val Acc':>10} {'Test Acc':>10}")
    print("  " + "-" * 40)
    print(f"  {'kNN (k=' + str(best_k) + ')':<18} {knn_val_acc:>10.2%} "
          f"{knn_test_acc:>10.2%}")
    print(f"  {'Decision Tree':<18} {dt_val_acc:>10.2%} "
          f"{dt_test_acc:>10.2%}")

    print("\nDone. See outputs/ for saved plots.")


if __name__ == "__main__":
    main()
