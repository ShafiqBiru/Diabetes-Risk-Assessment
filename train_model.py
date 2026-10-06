"""
train_model.py — Diabetes Prediction Pipeline Training
========================================================
Rocheston RCAI Classroom Project

Trains an end-to-end scikit-learn Pipeline on the Pima Indians Diabetes
dataset using all 8 clinical features.

Improvements over the initial notebook:
  1. Splits data BEFORE imputation/scaling to prevent data leakage.
  2. Uses sklearn.pipeline.Pipeline to encapsulate preprocessing.
  3. Benchmarks multiple classifiers and selects the best one.
  4. Saves the full pipeline (imputer + scaler + model) into Diabetesmodel.pkl
     so that raw clinical inputs are automatically transformed during inference.
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

DATA_PATH  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diabetes.csv")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Diabetesmodel.pkl")

ALL_FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
]
TARGET = "Outcome"


def load_and_clean(path: str) -> pd.DataFrame:
    """Load CSV and replace biologically impossible zeros with NaN."""
    df = pd.read_csv(path)
    zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    df[zero_cols] = df[zero_cols].replace(0, np.nan)
    print(f"[+] Loaded {len(df)} records from {os.path.basename(path)}")
    print(f"    Class distribution: Non-diabetic={int((df[TARGET]==0).sum())}, "
          f"Diabetic={int((df[TARGET]==1).sum())}")
    return df


def train():
    df = load_and_clean(DATA_PATH)

    X = df[ALL_FEATURES]
    y = df[TARGET]

    # Split FIRST to prevent data leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"[+] Split: {len(X_train)} train / {len(X_test)} test (stratified)")

    models = {
        "Random Forest":       RandomForestClassifier(n_estimators=100, criterion="entropy", random_state=42),
        "Logistic Regression":  LogisticRegression(max_iter=500, random_state=42),
        "KNN (k=24)":          KNeighborsClassifier(n_neighbors=24, metric="minkowski", p=2),
        "Decision Tree":       DecisionTreeClassifier(criterion="entropy", random_state=42),
    }

    print(f"\n{'='*72}")
    print(f"{'Model':<24} | {'Accuracy':>8} | {'Precision':>9} | {'Recall':>6} | {'F1':>6}")
    print(f"{'='*72}")

    best_name, best_acc, best_pipe = None, -1, None

    for name, clf in models.items():
        pipe = Pipeline([
            ("imputer",    SimpleImputer(strategy="mean")),
            ("scaler",     MinMaxScaler()),
            ("classifier", clf),
        ])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec  = recall_score(y_test, y_pred, zero_division=0)
        f1v  = f1_score(y_test, y_pred, zero_division=0)

        marker = ""
        if acc > best_acc:
            best_acc  = acc
            best_name = name
            best_pipe = pipe
            marker = "  ★"

        print(f"{name:<24} | {acc*100:7.2f}% | {prec:9.3f} | {rec:6.3f} | {f1v:6.3f}{marker}")

    print(f"{'='*72}")

    # Save best pipeline
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(best_pipe, f)
    print(f"\n[+] Best model: {best_name} ({best_acc*100:.2f}% accuracy)")
    print(f"[+] Saved pipeline to {MODEL_PATH}")

    # Verify inference with raw unscaled values
    print(f"\n[+] Inference test (raw unscaled input):")
    sample = pd.DataFrame(
        [[3, 137, 72, 30, 138, 43.0, 0.50, 33]],
        columns=ALL_FEATURES,
    )
    pred  = best_pipe.predict(sample)[0]
    probs = best_pipe.predict_proba(sample)[0]
    label = "Diabetic" if pred == 1 else "Non-Diabetic"
    print(f"    Input: {dict(zip(ALL_FEATURES, sample.values[0]))}")
    print(f"    Prediction: {label}")
    print(f"    Probabilities: Non-diabetic={probs[0]*100:.1f}%, Diabetic={probs[1]*100:.1f}%")
    print(f"    [✓] Inference scaling handled automatically by pipeline")


if __name__ == "__main__":
    train()
