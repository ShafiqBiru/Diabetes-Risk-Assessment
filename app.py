"""
app.py — AI Diabetes Risk Assessment
=====================================
Rocheston RCAI Classroom Project
A production-ready Streamlit web application for diabetes risk prediction
using machine learning trained on the Pima Indians Diabetes dataset.
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from io import BytesIO

import streamlit as st
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
    roc_curve,
    auc,
)

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
DATA_PATH   = os.path.join(os.path.dirname(__file__), "diabetes.csv")
MODEL_PATH  = os.path.join(os.path.dirname(__file__), "Diabetesmodel.pkl")

ALL_FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
]
TARGET = "Outcome"

# Clinical reference ranges  (for display & validation)
FEATURE_META = {
    "Pregnancies":              {"label": "Pregnancies",                     "min": 0,    "max": 20,   "default": 1,     "step": 1,   "unit": "",       "type": int,   "primary": False},
    "Glucose":                  {"label": "Fasting Glucose",                "min": 0,    "max": 300,  "default": 120,   "step": 1,   "unit": "mg/dL",  "type": int,   "primary": True},
    "BloodPressure":            {"label": "Blood Pressure (Diastolic)",     "min": 0,    "max": 200,  "default": 70,    "step": 1,   "unit": "mm Hg",  "type": int,   "primary": True},
    "SkinThickness":            {"label": "Skin Thickness",                 "min": 0,    "max": 100,  "default": 20,    "step": 1,   "unit": "mm",     "type": int,   "primary": False},
    "Insulin":                  {"label": "2-Hour Serum Insulin",           "min": 0,    "max": 900,  "default": 80,    "step": 1,   "unit": "µU/mL",  "type": int,   "primary": False},
    "BMI":                      {"label": "Body Mass Index (BMI)",          "min": 10.0, "max": 60.0, "default": 25.0,  "step": 0.1, "unit": "kg/m²",  "type": float, "primary": True},
    "DiabetesPedigreeFunction": {"label": "Diabetes Pedigree Function",     "min": 0.0,  "max": 2.5,  "default": 0.47,  "step": 0.01,"unit": "",       "type": float, "primary": False},
    "Age":                      {"label": "Age",                            "min": 1,    "max": 120,  "default": 30,    "step": 1,   "unit": "years",  "type": int,   "primary": True},
}

PRESETS = {
    "Custom": None,
    "Normal Clinical Profile": {
        "Pregnancies": 1, "Glucose": 88, "BloodPressure": 68, "SkinThickness": 22,
        "Insulin": 45, "BMI": 22.5, "DiabetesPedigreeFunction": 0.25, "Age": 28,
    },
    "Borderline / Pre-diabetic": {
        "Pregnancies": 2, "Glucose": 115, "BloodPressure": 78, "SkinThickness": 29,
        "Insulin": 125, "BMI": 29.5, "DiabetesPedigreeFunction": 0.45, "Age": 42,
    },
    "High-Risk Profile": {
        "Pregnancies": 5, "Glucose": 175, "BloodPressure": 88, "SkinThickness": 38,
        "Insulin": 220, "BMI": 38.2, "DiabetesPedigreeFunction": 1.20, "Age": 55,
    },
    "Notebook Sample (Glucose=137)": {
        "Pregnancies": 3, "Glucose": 137, "BloodPressure": 72, "SkinThickness": 30,
        "Insulin": 138, "BMI": 43.0, "DiabetesPedigreeFunction": 0.50, "Age": 33,
    },
}

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AI Diabetes Risk Assessment",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Global ──────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.block-container { padding-top: 1.5rem; }

/* ── Hero banner ─────────────────────────── */
.hero-banner {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0e7490 100%);
    border-radius: 16px;
    padding: 2.2rem 2.5rem;
    margin-bottom: 1.5rem;
    color: #f0f9ff;
    box-shadow: 0 8px 32px rgba(14, 116, 144, .25);
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -40%;
    right: -10%;
    width: 420px;
    height: 420px;
    background: radial-gradient(circle, rgba(56,189,248,.12) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-banner h1 {
    font-size: 2rem;
    font-weight: 800;
    letter-spacing: -0.5px;
    margin: 0 0 .35rem 0;
}
.hero-banner p {
    font-size: 1.02rem;
    opacity: .88;
    max-width: 620px;
    line-height: 1.55;
    margin: 0;
}
.hero-badge {
    display: inline-block;
    background: rgba(56,189,248,.18);
    border: 1px solid rgba(56,189,248,.35);
    border-radius: 999px;
    padding: .25rem .85rem;
    font-size: .78rem;
    font-weight: 600;
    letter-spacing: .4px;
    color: #7dd3fc;
    margin-bottom: .8rem;
}

/* ── Risk cards ──────────────────────────── */
.risk-card {
    border-radius: 14px;
    padding: 1.6rem 1.8rem;
    margin: .8rem 0;
    position: relative;
    overflow: hidden;
}
.risk-high {
    background: linear-gradient(135deg, #451a1a 0%, #7f1d1d 100%);
    border: 1px solid #dc2626;
    color: #fecaca;
}
.risk-high h3 { color: #fca5a5; margin-top: 0; }
.risk-low {
    background: linear-gradient(135deg, #052e16 0%, #14532d 100%);
    border: 1px solid #16a34a;
    color: #bbf7d0;
}
.risk-low h3 { color: #86efac; margin-top: 0; }
.risk-medium {
    background: linear-gradient(135deg, #451a03 0%, #78350f 100%);
    border: 1px solid #d97706;
    color: #fef3c7;
}
.risk-medium h3 { color: #fde68a; margin-top: 0; }

/* ── Metric tiles ────────────────────────── */
.metric-tile {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    text-align: center;
    color: #e2e8f0;
}
.metric-tile .value {
    font-size: 1.9rem;
    font-weight: 700;
    color: #38bdf8;
}
.metric-tile .label {
    font-size: .82rem;
    opacity: .7;
    margin-top: .2rem;
}

/* ── Disclaimer ──────────────────────────── */
.disclaimer {
    background: #172032;
    border: 1px solid #1e3a5f;
    border-left: 4px solid #0ea5e9;
    border-radius: 8px;
    padding: 1rem 1.4rem;
    font-size: .88rem;
    color: #94a3b8;
    line-height: 1.55;
    margin-top: 1.5rem;
}

/* ── Tab customisation ───────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0;
    padding: 10px 20px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  DATA & MODEL LOADING
# ─────────────────────────────────────────────
@st.cache_data
def load_dataset():
    """Load the Pima Indians Diabetes dataset."""
    df = pd.read_csv(DATA_PATH)
    return df


def _clean_df(df):
    """Replace biologically impossible zeros with NaN."""
    zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    df = df.copy()
    df[zero_cols] = df[zero_cols].replace(0, np.nan)
    return df


def _train_fresh_pipeline(df):
    """Train a new pipeline on the full 8-feature set and return it with metrics."""
    df = _clean_df(df)
    X = df[ALL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Build models
    models = {
        "Random Forest":       RandomForestClassifier(n_estimators=100, criterion="entropy", random_state=42),
        "Logistic Regression":  LogisticRegression(max_iter=500, random_state=42),
        "KNN (k=24)":          KNeighborsClassifier(n_neighbors=24, metric="minkowski", p=2),
        "Decision Tree":       DecisionTreeClassifier(criterion="entropy", random_state=42),
    }

    best_name, best_score, best_pipe = None, -1, None
    metrics_list = []

    for name, clf in models.items():
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler",  MinMaxScaler()),
            ("classifier", clf),
        ])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec  = recall_score(y_test, y_pred, zero_division=0)
        f1v  = f1_score(y_test, y_pred, zero_division=0)
        metrics_list.append({"Model": name, "Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1v})
        if acc > best_score:
            best_score = acc
            best_name  = name
            best_pipe  = pipe

    # Save best pipeline
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(best_pipe, f)

    return best_pipe, best_name, ALL_FEATURES, pd.DataFrame(metrics_list)


@st.cache_resource
def load_model():
    """
    Try to load Diabetesmodel.pkl.
    If it fails or is missing, retrain from diabetes.csv.
    Returns (pipeline, feature_names, training_metrics_df_or_None).
    """
    df = load_dataset()

    # Attempt to load existing model
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, "rb") as f:
                pipeline = pickle.load(f)

            # Determine what features it expects
            n_feats = pipeline.named_steps["scaler"].n_features_in_
            if n_feats == len(ALL_FEATURES):
                return pipeline, ALL_FEATURES, None
            elif n_feats == 4:
                return pipeline, ["Glucose", "Insulin", "BMI", "Age"], None
            else:
                raise ValueError(f"Unexpected feature count ({n_feats})")
        except Exception as e:
            st.warning(f"⚠️ Could not load existing model (`{e}`). Retraining from dataset…")

    # Fallback: retrain
    pipe, best_name, feat_names, metrics_df = _train_fresh_pipeline(df)
    st.success(f"✅ Trained fresh model — best algorithm: **{best_name}** ({metrics_df[metrics_df['Model']==best_name]['Accuracy'].iloc[0]*100:.1f}% accuracy)")
    return pipe, feat_names, metrics_df


def compute_benchmark_metrics(df, feature_names):
    """
    Run all benchmark models on dataset and return a DataFrame of metrics
    plus per-model predictions for confusion matrices.
    """
    df = _clean_df(df)
    X = df[feature_names]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    models = {
        "Random Forest":       RandomForestClassifier(n_estimators=100, criterion="entropy", random_state=42),
        "Logistic Regression":  LogisticRegression(max_iter=500, random_state=42),
        "KNN (k=24)":          KNeighborsClassifier(n_neighbors=24, metric="minkowski", p=2),
        "Decision Tree":       DecisionTreeClassifier(criterion="entropy", random_state=42),
    }

    rows = []
    predictions = {}
    for name, clf in models.items():
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler",  MinMaxScaler()),
            ("classifier", clf),
        ])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec  = recall_score(y_test, y_pred, zero_division=0)
        f1v  = f1_score(y_test, y_pred, zero_division=0)
        rows.append({"Model": name, "Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1v})
        predictions[name] = (y_test, y_pred)

    return pd.DataFrame(rows), predictions


# ─────────────────────────────────────────────
#  PLOTTING HELPERS
# ─────────────────────────────────────────────
def _fig_to_buf(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    buf.seek(0)
    plt.close(fig)
    return buf


DARK_BG   = "#0f172a"
DARK_FG   = "#e2e8f0"
GRID_CLR  = "#1e293b"
ACCENT    = "#38bdf8"
PALETTE   = ["#38bdf8", "#818cf8", "#34d399", "#fbbf24", "#f87171", "#c084fc"]


def plot_correlation_heatmap(df):
    fig, ax = plt.subplots(figsize=(9, 7))
    fig.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)
    numeric = df.select_dtypes(include=[np.number])
    mask = np.triu(np.ones_like(numeric.corr(), dtype=bool))
    sns.heatmap(
        numeric.corr(), mask=mask, annot=True, fmt=".2f",
        cmap="coolwarm", center=0, linewidths=.5,
        ax=ax, cbar_kws={"shrink": .8},
        annot_kws={"size": 9, "color": DARK_FG},
    )
    ax.tick_params(colors=DARK_FG, labelsize=9)
    ax.set_title("Feature Correlation Matrix", color=DARK_FG, fontsize=14, fontweight="bold", pad=14)
    return _fig_to_buf(fig)


def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Non-Diabetic", "Diabetic"],
        yticklabels=["Non-Diabetic", "Diabetic"],
        ax=ax, annot_kws={"size": 14},
    )
    ax.set_xlabel("Predicted", color=DARK_FG, fontsize=11)
    ax.set_ylabel("Actual", color=DARK_FG, fontsize=11)
    ax.set_title(title, color=DARK_FG, fontsize=13, fontweight="bold", pad=10)
    ax.tick_params(colors=DARK_FG)
    return _fig_to_buf(fig)


def plot_feature_distributions(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.drop(TARGET, errors="ignore")
    n = len(numeric_cols)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 3.2))
    fig.set_facecolor(DARK_BG)
    axes = axes.flatten()
    for i, col in enumerate(numeric_cols):
        ax = axes[i]
        ax.set_facecolor(DARK_BG)
        ax.hist(df[col].dropna(), bins=25, color=PALETTE[i % len(PALETTE)], alpha=.85, edgecolor="#1e293b")
        ax.set_title(col, color=DARK_FG, fontsize=11, fontweight="600")
        ax.tick_params(colors=DARK_FG, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(GRID_CLR)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Feature Distributions", color=DARK_FG, fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout()
    return _fig_to_buf(fig)


def plot_outcome_distribution(df):
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)
    counts = df[TARGET].value_counts().sort_index()
    bars = ax.bar(["Non-Diabetic (0)", "Diabetic (1)"], counts.values,
                  color=[PALETTE[2], PALETTE[4]], edgecolor=DARK_BG, width=.55)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 8, str(int(h)),
                ha="center", va="bottom", color=DARK_FG, fontweight="bold", fontsize=13)
    ax.set_ylabel("Count", color=DARK_FG)
    ax.set_title("Outcome Class Distribution", color=DARK_FG, fontsize=13, fontweight="bold", pad=10)
    ax.tick_params(colors=DARK_FG)
    for spine in ax.spines.values():
        spine.set_color(GRID_CLR)
    return _fig_to_buf(fig)


def plot_benchmark_bars(metrics_df):
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)
    x = np.arange(len(metrics_df))
    w = 0.2
    for i, col in enumerate(["Accuracy", "Precision", "Recall", "F1"]):
        bars = ax.bar(x + i * w, metrics_df[col], w, label=col, color=PALETTE[i], alpha=.9)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.008,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=7.5, color=DARK_FG)
    ax.set_xticks(x + 1.5 * w)
    ax.set_xticklabels(metrics_df["Model"], color=DARK_FG, fontsize=10)
    ax.set_ylabel("Score", color=DARK_FG)
    ax.set_ylim(0, 1.12)
    ax.set_title("Model Performance Benchmark", color=DARK_FG, fontsize=14, fontweight="bold", pad=12)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.3)
    ax.tick_params(colors=DARK_FG)
    for spine in ax.spines.values():
        spine.set_color(GRID_CLR)
    return _fig_to_buf(fig)


def plot_risk_gauge(probability):
    """Semicircular gauge showing risk probability."""
    fig, ax = plt.subplots(figsize=(4.5, 2.8), subplot_kw={"projection": "polar"})
    fig.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)

    # Gauge goes from π to 0  (left to right)
    theta_bg = np.linspace(np.pi, 0, 200)
    ax.fill_between(theta_bg, 0.6, 1.0, color="#1e293b", alpha=.6)

    # Colour zones
    green  = np.linspace(np.pi, np.pi * 0.6, 80)
    yellow = np.linspace(np.pi * 0.6, np.pi * 0.35, 50)
    red    = np.linspace(np.pi * 0.35, 0, 70)
    ax.fill_between(green,  0.62, 0.98, color="#22c55e", alpha=.35)
    ax.fill_between(yellow, 0.62, 0.98, color="#eab308", alpha=.35)
    ax.fill_between(red,    0.62, 0.98, color="#ef4444", alpha=.35)

    # Needle
    angle = np.pi * (1 - probability)
    ax.plot([angle, angle], [0, 0.95], color="#f8fafc", linewidth=2.5, zorder=10)
    ax.scatter([angle], [0.95], color="#f8fafc", s=30, zorder=11)
    ax.scatter([0], [0], color="#64748b", s=60, zorder=11)

    ax.set_ylim(0, 1.15)
    ax.set_thetamin(0)
    ax.set_thetamax(180)
    ax.set_axis_off()

    # Labels
    fig.text(0.5, 0.18, f"{probability*100:.1f}%", ha="center", va="center",
             fontsize=26, fontweight="800", color=DARK_FG, family="Inter")
    fig.text(0.5, 0.04, "Diabetes Risk Score", ha="center", va="center",
             fontsize=10, color="#94a3b8", family="Inter")
    fig.text(0.10, 0.28, "Low", ha="center", fontsize=8, color="#22c55e")
    fig.text(0.90, 0.28, "High", ha="center", fontsize=8, color="#ef4444")

    return _fig_to_buf(fig)


# ─────────────────────────────────────────────
#  MAIN APPLICATION
# ─────────────────────────────────────────────
def main():
    # Hero banner
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-badge">RCAI CLASSROOM PROJECT</div>
        <h1>🩺 AI Diabetes Risk Assessment</h1>
        <p>Machine-learning powered diabetes screening using the Pima Indians Diabetes dataset.
        Enter clinical parameters to receive an AI-assisted risk estimate in real time.</p>
    </div>
    """, unsafe_allow_html=True)

    # Load data & model
    df_raw = load_dataset()
    pipeline, feature_names, initial_metrics = load_model()

    # ── TABS ─────────────────────────────────
    tab_assess, tab_explore, tab_analytics = st.tabs([
        "🔍  Risk Assessment",
        "📊  Data Explorer",
        "🧪  Model Analytics",
    ])

    # ═══════════════════════════════════════════
    #  TAB 1 — RISK ASSESSMENT
    # ═══════════════════════════════════════════
    with tab_assess:
        # Sidebar: preset selector
        with st.sidebar:
            st.markdown("### ⚡ Quick Presets")
            preset_name = st.selectbox(
                "Load a sample patient profile",
                list(PRESETS.keys()),
                index=0,
                help="Pre-fill the input fields with a representative clinical profile.",
            )
            preset = PRESETS[preset_name]

            st.markdown("---")
            st.markdown("### 📖 Clinical Ranges")
            st.markdown("""
| Metric | Normal | Concern |
|--------|--------|---------|
| Fasting Glucose | 70–99 | ≥ 126 mg/dL |
| Blood Pressure | 60–80 | ≥ 90 mm Hg |
| BMI | 18.5–24.9 | ≥ 30 |
| Insulin (2h) | 16–166 | > 166 µU/mL |
            """)
            st.markdown("---")
            st.caption("Model features: " + ", ".join(feature_names))

        # ── Input panel ──
        st.subheader("Patient Clinical Input")

        # Determine primary vs secondary features
        primary_feats   = [f for f in feature_names if FEATURE_META[f]["primary"]]
        secondary_feats = [f for f in feature_names if not FEATURE_META[f]["primary"]]

        col_a, col_b = st.columns(2)
        input_values = {}

        # Primary features
        for idx, feat in enumerate(primary_feats):
            meta = FEATURE_META[feat]
            default = preset[feat] if preset else meta["default"]
            container = col_a if idx % 2 == 0 else col_b
            with container:
                label_str = f"{meta['label']}"
                if meta["unit"]:
                    label_str += f"  ({meta['unit']})"
                val = st.slider(
                    label_str,
                    min_value=meta["type"](meta["min"]),
                    max_value=meta["type"](meta["max"]),
                    value=meta["type"](default),
                    step=meta["type"](meta["step"]) if meta["type"] == float else meta["step"],
                    key=f"input_{feat}",
                )
                input_values[feat] = val

        # Secondary features (collapsible)
        if secondary_feats:
            with st.expander("🔧  Advanced Parameters (click to expand)", expanded=False):
                sec_cols = st.columns(min(len(secondary_feats), 4))
                for idx, feat in enumerate(secondary_feats):
                    meta = FEATURE_META[feat]
                    default = preset[feat] if preset else meta["default"]
                    with sec_cols[idx % len(sec_cols)]:
                        label_str = f"{meta['label']}"
                        if meta["unit"]:
                            label_str += f"  ({meta['unit']})"
                        val = st.slider(
                            label_str,
                            min_value=meta["type"](meta["min"]),
                            max_value=meta["type"](meta["max"]),
                            value=meta["type"](default),
                            step=meta["type"](meta["step"]) if meta["type"] == float else meta["step"],
                            key=f"input_{feat}",
                        )
                        input_values[feat] = val

        st.markdown("")

        # ── Input validation ──
        warnings_list = []
        if input_values.get("Glucose", 999) == 0:
            warnings_list.append("⚠️ Glucose of 0 mg/dL is physiologically impossible. This will be treated as a missing value.")
        if input_values.get("BloodPressure", 999) == 0:
            warnings_list.append("⚠️ Blood Pressure of 0 mm Hg is physiologically impossible. This will be treated as a missing value.")
        if input_values.get("BMI", 999) <= 0:
            warnings_list.append("⚠️ BMI must be a positive number.")
        if input_values.get("Age", 1) < 1:
            warnings_list.append("⚠️ Age must be at least 1 year.")

        for w in warnings_list:
            st.warning(w)

        # ── Predict button ──
        predict_btn = st.button("🚀  Run Assessment", type="primary", use_container_width=True)

        if predict_btn:
            # Build input DataFrame
            row = {feat: input_values.get(feat, FEATURE_META[feat]["default"]) for feat in feature_names}
            # Replace physiological zeros with NaN so imputer handles them
            for zf in ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]:
                if zf in row and row[zf] == 0:
                    row[zf] = np.nan
            patient_df = pd.DataFrame([row], columns=feature_names)

            prediction = pipeline.predict(patient_df)[0]

            # Probability
            if hasattr(pipeline, "predict_proba"):
                probs = pipeline.predict_proba(patient_df)[0]
            else:
                probs = np.array([1 - prediction, prediction], dtype=float)

            prob_diabetic     = float(probs[1])
            prob_non_diabetic = float(probs[0])

            st.markdown("---")
            st.subheader("📋 Assessment Results")

            res_left, res_right = st.columns([3, 2])

            with res_left:
                # Risk level
                if prob_diabetic >= 0.65:
                    card_class = "risk-high"
                    icon = "🔴"
                    label = "High Risk — Likely Diabetic"
                elif prob_diabetic >= 0.40:
                    card_class = "risk-medium"
                    icon = "🟡"
                    label = "Moderate Risk — Borderline"
                else:
                    card_class = "risk-low"
                    icon = "🟢"
                    label = "Low Risk — Unlikely Diabetic"

                st.markdown(f"""
                <div class="risk-card {card_class}">
                    <h3>{icon} {label}</h3>
                    <p style="font-size:1.05rem; margin-bottom:.5rem;">
                        The model predicts a <strong>{prob_diabetic*100:.1f}%</strong> probability of diabetes
                        based on the provided clinical parameters.
                    </p>
                    <p style="opacity:.8; font-size:.9rem;">
                        Prediction: <strong>{'Diabetic (Class 1)' if prediction == 1 else 'Non-Diabetic (Class 0)'}</strong>
                    </p>
                </div>
                """, unsafe_allow_html=True)

                # Risk factor flags
                st.markdown("##### Key Risk Factors")
                flags = []
                g = input_values.get("Glucose", 0)
                if g >= 126:
                    flags.append("🔴 **Glucose ≥ 126 mg/dL** — Diabetic fasting threshold")
                elif g >= 100:
                    flags.append("🟡 **Glucose 100–125 mg/dL** — Impaired fasting glucose (Prediabetes)")

                bp = input_values.get("BloodPressure", 0)
                if bp >= 90:
                    flags.append("🔴 **Blood Pressure ≥ 90 mm Hg** — Hypertension stage")
                elif bp >= 80:
                    flags.append("🟡 **Blood Pressure 80–89 mm Hg** — Elevated")

                bmi_val = input_values.get("BMI", 0)
                if bmi_val >= 30:
                    flags.append("🔴 **BMI ≥ 30** — Obesity (significant insulin resistance risk)")
                elif bmi_val >= 25:
                    flags.append("🟡 **BMI 25–29.9** — Overweight")

                age_val = input_values.get("Age", 0)
                if age_val >= 45:
                    flags.append("🟡 **Age ≥ 45** — Elevated age-related risk")

                ins = input_values.get("Insulin", 0)
                if ins > 166:
                    flags.append("🟡 **Insulin > 166 µU/mL** — Hyperinsulinemia")

                if flags:
                    for f in flags:
                        st.markdown(f)
                else:
                    st.success("All parameters within healthy clinical ranges.")

            with res_right:
                # Gauge
                gauge_buf = plot_risk_gauge(prob_diabetic)
                st.image(gauge_buf, use_container_width=True)

                # Probability bars
                mcol1, mcol2 = st.columns(2)
                with mcol1:
                    st.markdown(f"""
                    <div class="metric-tile">
                        <div class="value" style="color:#22c55e">{prob_non_diabetic*100:.1f}%</div>
                        <div class="label">Non-Diabetic</div>
                    </div>
                    """, unsafe_allow_html=True)
                with mcol2:
                    st.markdown(f"""
                    <div class="metric-tile">
                        <div class="value" style="color:#ef4444">{prob_diabetic*100:.1f}%</div>
                        <div class="label">Diabetic</div>
                    </div>
                    """, unsafe_allow_html=True)

            # Disclaimer
            st.markdown("""
            <div class="disclaimer">
                <strong>⚕️ Medical Disclaimer:</strong> This tool is an AI-assisted screening estimate
                developed for academic purposes using the Pima Indians Diabetes Dataset. It is
                <strong>not a medical diagnosis</strong>. Always consult a qualified healthcare
                professional for clinical evaluation, diagnostic testing, and treatment decisions.
            </div>
            """, unsafe_allow_html=True)


    # ═══════════════════════════════════════════
    #  TAB 2 — DATA EXPLORER
    # ═══════════════════════════════════════════
    with tab_explore:
        st.subheader("📊 Dataset Overview")

        info_col1, info_col2, info_col3, info_col4 = st.columns(4)
        with info_col1:
            st.metric("Total Records", f"{len(df_raw):,}")
        with info_col2:
            st.metric("Features", len(df_raw.columns) - 1)
        with info_col3:
            st.metric("Diabetic (1)", int(df_raw[TARGET].sum()))
        with info_col4:
            st.metric("Non-Diabetic (0)", int((df_raw[TARGET] == 0).sum()))

        st.markdown("##### First 10 Records")
        st.dataframe(df_raw.head(10), use_container_width=True, hide_index=True)

        st.markdown("##### Statistical Summary")
        st.dataframe(df_raw.describe().T.round(2), use_container_width=True)

        st.markdown("---")
        dist_col1, dist_col2 = st.columns([2, 1])
        with dist_col1:
            st.markdown("##### Feature Distributions")
            st.image(plot_feature_distributions(df_raw), use_container_width=True)
        with dist_col2:
            st.markdown("##### Outcome Balance")
            st.image(plot_outcome_distribution(df_raw), use_container_width=True)

        st.markdown("---")
        st.markdown("##### Correlation Heatmap")
        st.image(plot_correlation_heatmap(df_raw), use_container_width=True)


    # ═══════════════════════════════════════════
    #  TAB 3 — MODEL ANALYTICS
    # ═══════════════════════════════════════════
    with tab_analytics:
        st.subheader("🧪 Model Performance Benchmarks")
        st.caption(f"Evaluated on 20% held-out test set · Features: {', '.join(feature_names)}")

        with st.spinner("Running model benchmarks…"):
            metrics_df, preds = compute_benchmark_metrics(df_raw, feature_names)

        # Metrics table
        fmt_df = metrics_df.copy()
        for col in ["Accuracy", "Precision", "Recall", "F1"]:
            fmt_df[col] = fmt_df[col].apply(lambda v: f"{v*100:.2f}%")
        st.dataframe(fmt_df, use_container_width=True, hide_index=True)

        # Bar chart
        st.image(plot_benchmark_bars(metrics_df), use_container_width=True)

        # Confusion matrices
        st.markdown("---")
        st.markdown("##### Confusion Matrices")
        cm_cols = st.columns(2)
        for idx, (name, (y_true, y_pred)) in enumerate(preds.items()):
            with cm_cols[idx % 2]:
                buf = plot_confusion_matrix(y_true, y_pred, title=name)
                st.image(buf, use_container_width=True)


if __name__ == "__main__":
    main()
