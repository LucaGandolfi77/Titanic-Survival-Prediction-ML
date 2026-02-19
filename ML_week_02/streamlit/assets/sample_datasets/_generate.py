"""Generate sample datasets for the ML Playground."""
from pathlib import Path
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
import pandas as pd

OUT = Path(__file__).resolve().parent

def save(name, bunch):
    df = pd.DataFrame(bunch.data, columns=bunch.feature_names)
    df["target"] = bunch.target
    df.to_csv(OUT / name, index=False)
    print(f"  ✓ {name} ({df.shape})")

if __name__ == "__main__":
    save("iris.csv", load_iris())
    save("wine.csv", load_wine())
    save("breast_cancer.csv", load_breast_cancer())
    print("Done.")
