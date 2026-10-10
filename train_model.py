"""
train_model.py — Diabetes Prediction Pipeline Training
========================================================
Rocheston RCAI Classroom Project

Trains end-to-end scikit-learn Pipelines on the Pima Indians Diabetes
dataset:
  1. Full 8-feature clinical pipeline (Diabetesmodel.pkl).
  2. Dedicated 4-core feature pipeline for non-invasive screening (Diabetesmodel_4.pkl).
  3. Precomputes population medians for laboratory biomarker imputation.
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

DATA_PATH    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diabetes.csv")
MODEL_PATH_8 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Diabetesmodel.pkl")
MODEL_PATH_4 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Diabetesmodel_4.pkl")

ALL_FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
]
PRIMARY_FEATURES = ["Glucose", "BloodPressure", "BMI", "Age"]
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

    # ─────────────────────────────────────────────
    #  1. TRAIN 8-FEATURE PIPELINE
    # ─────────────────────────────────────────────
    X8 = df[ALL_FEATURES]
    y = df[TARGET]

    X8_train, X8_test, y_train, y_test = train_test_split(
        X8, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"\n[+] 8-Feature Train/Test Split: {len(X8_train)} train / {len(X8_test)} test")

    models8 = {
        "Random Forest":       RandomForestClassifier(n_estimators=100, criterion="entropy", random_state=42),
        "Logistic Regression":  LogisticRegression(max_iter=500, random_state=42),
        "KNN (k=24)":          KNeighborsClassifier(n_neighbors=24, metric="minkowski", p=2),
        "Decision Tree":       DecisionTreeClassifier(criterion="entropy", random_state=42),
    }

    print(f"\n{'='*72}")
    print(f"8-Feature Models: {'Model':<20} | {'Accuracy':>8} | {'Precision':>9} | {'Recall':>6} | {'F1':>6}")
    print(f"{'='*72}")

    best_name8, best_acc8, best_pipe8 = None, -1, None

    for name, clf in models8.items():
        pipe = Pipeline([
            ("imputer",    SimpleImputer(strategy="mean")),
            ("scaler",     MinMaxScaler()),
            ("classifier", clf),
        ])
        pipe.fit(X8_train, y_train)
        y_pred = pipe.predict(X8_test)

        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec  = recall_score(y_test, y_pred, zero_division=0)
        f1v  = f1_score(y_test, y_pred, zero_division=0)

        marker = ""
        if acc > best_acc8:
            best_acc8  = acc
            best_name8 = name
            best_pipe8 = pipe
            marker = "  ★"

        print(f"{name:<20} | {acc*100:7.2f}% | {prec:9.3f} | {rec:6.3f} | {f1v:6.3f}{marker}")

    with open(MODEL_PATH_8, "wb") as f:
        pickle.dump(best_pipe8, f)
    print(f"[+] Saved 8-feature pipeline to {MODEL_PATH_8} ({best_name8}: {best_acc8*100:.2f}%)")

    # ─────────────────────────────────────────────
    #  2. TRAIN DEDICATED 4-CORE FEATURE PIPELINE
    # ─────────────────────────────────────────────
    X4 = df[PRIMARY_FEATURES]
    X4_train, X4_test, _, _ = train_test_split(
        X4, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"\n[+] 4-Feature Train/Test Split: {len(X4_train)} train / {len(X4_test)} test")

    pipe4 = Pipeline([
        ("imputer",    SimpleImputer(strategy="mean")),
        ("scaler",     MinMaxScaler()),
        ("classifier", LogisticRegression(max_iter=500, random_state=42)),
    ])
    pipe4.fit(X4_train, y_train)
    y4_pred = pipe4.predict(X4_test)
    acc4 = accuracy_score(y_test, y4_pred)

    with open(MODEL_PATH_4, "wb") as f:
        pickle.dump(pipe4, f)
    print(f"[+] Saved 4-feature pipeline to {MODEL_PATH_4} (Logistic Regression: {acc4*100:.2f}%)")

    # ─────────────────────────────────────────────
    #  3. POPULATION MEDIANS REPORT
    # ─────────────────────────────────────────────
    print(f"\n[+] Population Medians for Imputation:")
    for feat in ["Insulin", "SkinThickness", "Pregnancies", "DiabetesPedigreeFunction"]:
        med = df[feat].median()
        print(f"    - {feat:<24}: {med}")


if __name__ == "__main__":
    train()
