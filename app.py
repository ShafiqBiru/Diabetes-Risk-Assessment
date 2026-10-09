"""
app.py — AI Diabetes Risk Assessment
=====================================
Rocheston RCAI Classroom Project
A production-ready Streamlit web application for diabetes risk prediction
using machine learning trained on the Pima Indians Diabetes dataset.

Design: Apple Health UI Design System (iOS/macOS Human Interface Guidelines)
"""

import os
import pickle
import warnings
from io import BytesIO

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import streamlit as st
import streamlit.components.v1 as components

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
    confusion_matrix,
)

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  CONFIGURATION & FEATURE METADATA
# ─────────────────────────────────────────────
DATA_PATH  = os.path.join(os.path.dirname(__file__), "diabetes.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "Diabetesmodel.pkl")

ALL_FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
]
TARGET = "Outcome"

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

CLINICAL_RANGES = [
    {
        "metric": "Fasting Glucose",
        "icon": "💧",
        "unit": "mg/dL",
        "normal": "70 – 99",
        "concern": "≥ 126",
    },
    {
        "metric": "Blood Pressure",
        "icon": "❤️",
        "unit": "mm Hg",
        "normal": "< 80",
        "concern": "≥ 90",
    },
    {
        "metric": "Body Mass Index",
        "icon": "⚖️",
        "unit": "kg/m²",
        "normal": "18.5 – 24.9",
        "concern": "≥ 30.0",
    },
    {
        "metric": "Insulin (2-hr)",
        "icon": "💉",
        "unit": "µU/mL",
        "normal": "16 – 166",
        "concern": "> 166",
    },
]

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AI Diabetes Risk Assessment",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
#  APPLE HEALTH DESIGN SYSTEM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── Apple System Font Stack ────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=SF+Pro+Display:wght@400;500;600;700&family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"], .stApp {
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text", "Helvetica Neue", "Inter", sans-serif !important;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
    background-color: #F2F2F7 !important;
    color: #1C1C1E !important;
}

/* ── Fixed Ambient Color Orbs for High Optical Transparency Refraction ── */
.ambient-orb-blue {
    position: fixed;
    top: 4%;
    left: 6%;
    width: 560px;
    height: 560px;
    background: radial-gradient(circle, rgba(0, 122, 255, 0.28) 0%, rgba(88, 86, 214, 0.16) 45%, transparent 70%);
    filter: blur(85px);
    -webkit-filter: blur(85px);
    border-radius: 50%;
    z-index: 0;
    pointer-events: none;
    transform: translateZ(0);
}

.ambient-orb-cyan {
    position: fixed;
    bottom: 6%;
    right: 5%;
    width: 600px;
    height: 600px;
    background: radial-gradient(circle, rgba(50, 199, 235, 0.30) 0%, rgba(52, 199, 89, 0.16) 50%, transparent 70%);
    filter: blur(95px);
    -webkit-filter: blur(95px);
    border-radius: 50%;
    z-index: 0;
    pointer-events: none;
    transform: translateZ(0);
}

/* ── Container Layout ───────────────────────────────────── */
.block-container {
    position: relative !important;
    z-index: 1 !important;
    padding-top: 1.5rem !important;
    padding-bottom: 3.5rem !important;
    max-width: 1100px !important;
}

/* ── Header Chrome Clean Up (Liquid Frosted Glass) ──────── */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header[data-testid="stHeader"] {
    background: rgba(242, 242, 247, 0.45) !important;
    backdrop-filter: blur(28px) saturate(210%) !important;
    -webkit-backdrop-filter: blur(28px) saturate(210%) !important;
    border-bottom: 1px solid rgba(255, 255, 255, 0.6) !important;
}

/* ── Global Typography Hierarchy & High Contrast ────────── */
h1, h2, h3, h4, h5, h6 {
    color: #1C1C1E !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em !important;
}
p, span, label, div {
    color: #1C1C1E;
}
.stMarkdown p {
    color: #1C1C1E !important;
    line-height: 1.5;
}

/* ── Hero Banner (Apple-style vibrant gradient) ─────────── */
.hero-banner {
    background: linear-gradient(135deg, #007AFF 0%, #5856D6 55%, #AF52DE 100%);
    border-radius: 22px;
    padding: 2.2rem 2.6rem;
    margin-bottom: 1.8rem;
    color: #FFFFFF !important;
    box-shadow: 0 10px 32px rgba(0, 122, 255, 0.24);
    position: relative;
    overflow: hidden;
    border: 1px solid rgba(255, 255, 255, 0.45);
    border-top: 1px solid rgba(255, 255, 255, 0.85);
}
.hero-banner::after {
    content: '';
    position: absolute;
    top: -60%;
    right: -20%;
    width: 480px;
    height: 480px;
    background: radial-gradient(circle, rgba(255,255,255,0.14) 0%, transparent 70%);
    border-radius: 50%;
    pointer-events: none;
}
.hero-banner h1 {
    font-size: 2.1rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.03em !important;
    margin: 0 0 0.4rem 0 !important;
    color: #FFFFFF !important;
}
.hero-banner p {
    font-size: 1.05rem !important;
    opacity: 0.95;
    max-width: 650px;
    line-height: 1.55;
    margin: 0 !important;
    color: #FFFFFF !important;
}
.hero-badge {
    display: inline-block;
    background: rgba(255, 255, 255, 0.2);
    border: 1px solid rgba(255, 255, 255, 0.4);
    border-radius: 999px;
    padding: 0.25rem 0.85rem;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.6px;
    color: #FFFFFF !important;
    margin-bottom: 0.8rem;
    text-transform: uppercase;
}

/* ── Section Category Labels (Apple Health Small Caps) ──── */
.section-label {
    font-size: 0.75rem !important;
    font-weight: 700 !important;
    color: #8E8E93 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.8px !important;
    margin-top: 1rem !important;
    margin-bottom: 0.6rem !important;
    padding-left: 2px !important;
}

/* ── True Apple Liquid Glass Shader (60%–70% Optical Transparency) ── */
.apple-glass-card {
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.38) 0%, rgba(255, 255, 255, 0.22) 100%) !important;
    backdrop-filter: blur(28px) saturate(210%) !important;
    -webkit-backdrop-filter: blur(28px) saturate(210%) !important;
    border-radius: 20px !important;
    border: 1px solid rgba(255, 255, 255, 0.45) !important;
    border-top: 1px solid rgba(255, 255, 255, 0.75) !important;
    box-shadow: 0 10px 32px 0 rgba(0, 0, 0, 0.06), inset 0 1px 1px 0 rgba(255, 255, 255, 0.7) !important;
    padding: 1.4rem 1.6rem !important;
    margin-bottom: 1rem !important;
    transition: transform 0.2s cubic-bezier(0.16, 1, 0.3, 1), box-shadow 0.2s cubic-bezier(0.16, 1, 0.3, 1);
}
.apple-glass-card:hover {
    box-shadow: 0 14px 44px 0 rgba(0, 0, 0, 0.08), inset 0 1px 1px 0 rgba(255, 255, 255, 0.9) !important;
    transform: translateY(-1px);
}

/* ── Input Parameter Glass Container (60%–70% Transparency) ── */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.38) 0%, rgba(255, 255, 255, 0.22) 100%) !important;
    backdrop-filter: blur(28px) saturate(210%) !important;
    -webkit-backdrop-filter: blur(28px) saturate(210%) !important;
    border-radius: 20px !important;
    border: 1px solid rgba(255, 255, 255, 0.45) !important;
    border-top: 1px solid rgba(255, 255, 255, 0.75) !important;
    box-shadow: 0 10px 32px 0 rgba(0, 0, 0, 0.06), inset 0 1px 1px 0 rgba(255, 255, 255, 0.7) !important;
    padding: 1.3rem 1.5rem !important;
    margin-bottom: 1.1rem !important;
}

/* ── Metric Dual-Input Header Row ── */
.metric-row-header {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 0.15rem;
    padding-left: 2px;
    padding-right: 2px;
}
.metric-row-title {
    font-weight: 700;
    font-size: 0.92rem;
    color: #1C1C1E;
    letter-spacing: -0.01em;
}
.metric-row-unit {
    font-size: 0.72rem;
    font-weight: 600;
    color: #8E8E93;
    letter-spacing: 0.3px;
}

/* ── Apple-Style Liquid Glass Interactive Slider Thumb (Dragging Head) ── */
/* BaseWeb Streamlit Slider & Native Range Input Thumbs */
div[data-testid="stSlider"] div[role="slider"],
input[type="range"]::-webkit-slider-thumb,
input[type="range"]::-moz-range-thumb {
    width: 24px !important;
    height: 32px !important;
    border-radius: 8px !important;
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.45) 0%, rgba(255, 255, 255, 0.25) 100%) !important;
    backdrop-filter: blur(12px) saturate(180%) !important;
    -webkit-backdrop-filter: blur(12px) saturate(180%) !important;
    border: 1px solid rgba(255, 255, 255, 0.8) !important;
    border-top: 1px solid rgba(255, 255, 255, 0.95) !important;
    box-shadow: 0 3px 12px rgba(0, 0, 0, 0.15), inset 0 1px 1px rgba(255, 255, 255, 0.9) !important;
    cursor: grab !important;
    transition: transform 0.15s cubic-bezier(0.16, 1, 0.3, 1), box-shadow 0.15s ease, background 0.15s ease !important;
    outline: none !important;
}

/* Hover: Subtle scale-up with heightened top specular reflection */
div[data-testid="stSlider"] div[role="slider"]:hover,
input[type="range"]::-webkit-slider-thumb:hover,
input[type="range"]::-moz-range-thumb:hover {
    transform: scale(1.06) !important;
    border-top: 1px solid rgba(255, 255, 255, 1) !important;
    box-shadow: 0 4px 16px rgba(0, 122, 255, 0.22), inset 0 1px 1.5px rgba(255, 255, 255, 1) !important;
}

/* Active / Dragging: Elastic tactile feedback + semi-transparent glass fill */
div[data-testid="stSlider"] div[role="slider"]:active,
div[data-testid="stSlider"] div[role="slider"]:focus,
input[type="range"]::-webkit-slider-thumb:active,
input[type="range"]::-moz-range-thumb:active {
    transform: scale(1.12) !important;
    cursor: grabbing !important;
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.50) 0%, rgba(255, 255, 255, 0.30) 100%) !important;
    border: 1px solid rgba(255, 255, 255, 0.95) !important;
    box-shadow: 0 6px 20px rgba(0, 122, 255, 0.25), inset 0 1px 2px rgba(255, 255, 255, 1) !important;
}

/* Slider Track styling: Semi-translucent with Apple Blue active tint */
div[data-testid="stSlider"] {
    padding-top: 6px !important;
}
div[data-testid="stSlider"] div[data-baseweb="slider"] {
    margin-top: -2px !important;
}
div[data-testid="stSlider"] div[data-baseweb="slider"] > div > div {
    border-radius: 999px !important;
}
div[data-testid="stSliderTickBarMin"],
div[data-testid="stSliderTickBarMax"] {
    color: #8E8E93 !important;
    font-size: 0.76rem !important;
    font-weight: 500 !important;
}

/* ── Apple-Style Liquid Glass Numeric Entry Box ───────────── */
div[data-testid="stNumberInput"] {
    padding-top: 0px !important;
    margin-top: 0px !important;
}
div[data-testid="stNumberInput"] div[data-baseweb="input"] {
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.38) 0%, rgba(255, 255, 255, 0.22) 100%) !important;
    backdrop-filter: blur(28px) saturate(210%) !important;
    -webkit-backdrop-filter: blur(28px) saturate(210%) !important;
    border-radius: 12px !important;
    border: 1px solid rgba(255, 255, 255, 0.45) !important;
    border-top: 1px solid rgba(255, 255, 255, 0.75) !important;
    box-shadow: 0 4px 14px 0 rgba(0, 0, 0, 0.04), inset 0 1px 1px 0 rgba(255, 255, 255, 0.7) !important;
    transition: all 0.2s ease !important;
    height: 42px !important;
}
div[data-testid="stNumberInput"] div[data-baseweb="input"]:focus-within {
    border-color: #007AFF !important;
    box-shadow: 0 0 0 3px rgba(0, 122, 255, 0.22), inset 0 1px 1px 0 rgba(255, 255, 255, 0.9) !important;
}
div[data-testid="stNumberInput"] input {
    font-weight: 700 !important;
    font-size: 0.98rem !important;
    color: #1C1C1E !important;
    text-align: center !important;
    font-variant-numeric: tabular-nums !important;
}
div[data-testid="stNumberInput"] button {
    background: transparent !important;
    border: none !important;
    color: #8E8E93 !important;
    transition: color 0.15s ease !important;
}
div[data-testid="stNumberInput"] button:hover {
    color: #007AFF !important;
}

/* ── Clinical Reference Range Cards (Liquid Frosted Glass) ─ */
.range-card {
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.38) 0%, rgba(255, 255, 255, 0.22) 100%) !important;
    backdrop-filter: blur(28px) saturate(210%) !important;
    -webkit-backdrop-filter: blur(28px) saturate(210%) !important;
    border-radius: 20px !important;
    border: 1px solid rgba(255, 255, 255, 0.45) !important;
    border-top: 1px solid rgba(255, 255, 255, 0.75) !important;
    box-shadow: 0 10px 32px 0 rgba(0, 0, 0, 0.06), inset 0 1px 1px 0 rgba(255, 255, 255, 0.7) !important;
    padding: 1.2rem 1.1rem !important;
    text-align: center;
    transition: all 0.22s cubic-bezier(0.16, 1, 0.3, 1);
    height: 100%;
}
.range-card:hover {
    box-shadow: 0 14px 44px 0 rgba(0, 0, 0, 0.08), inset 0 1px 1px rgba(255, 255, 255, 0.9) !important;
    transform: translateY(-2px);
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.45) 0%, rgba(255, 255, 255, 0.30) 100%) !important;
}
.range-card .range-icon {
    font-size: 1.8rem;
    margin-bottom: 0.4rem;
}
.range-card .range-metric {
    font-size: 0.92rem !important;
    font-weight: 700 !important;
    color: #1C1C1E !important;
    margin-bottom: 0.25rem;
}
.range-card .range-unit {
    font-size: 0.72rem !important;
    color: #8E8E93 !important;
    margin-bottom: 0.75rem;
    font-weight: 500;
}
.range-badge {
    display: inline-block;
    padding: 0.22rem 0.65rem;
    border-radius: 999px;
    font-size: 0.72rem !important;
    font-weight: 700 !important;
    margin: 0.15rem 0.1rem;
    letter-spacing: 0.2px;
}
.badge-normal {
    background: rgba(52, 199, 89, 0.18) !important;
    color: #248A3D !important;
}
.badge-danger {
    background: rgba(255, 59, 48, 0.18) !important;
    color: #D70015 !important;
}

/* ── Risk Result Glass Panels (Liquid Frosted Glass) ─────── */
.risk-card {
    border-radius: 20px !important;
    padding: 1.6rem 1.8rem !important;
    margin: 0.6rem 0 !important;
    backdrop-filter: blur(28px) saturate(210%) !important;
    -webkit-backdrop-filter: blur(28px) saturate(210%) !important;
    border-top: 1px solid rgba(255, 255, 255, 0.75) !important;
    box-shadow: 0 10px 32px 0 rgba(0, 0, 0, 0.06), inset 0 1px 1px 0 rgba(255, 255, 255, 0.7) !important;
    transition: all 0.2s ease;
}
.risk-card h3 {
    margin: 0 0 0.5rem 0 !important;
    font-size: 1.45rem !important;
    font-weight: 700 !important;
}
.risk-high {
    background: linear-gradient(135deg, rgba(255, 238, 238, 0.38) 0%, rgba(255, 238, 238, 0.22) 100%) !important;
    border: 1.5px solid rgba(255, 59, 48, 0.35) !important;
    border-top: 1px solid rgba(255, 255, 255, 0.85) !important;
}
.risk-high h3 { color: #D70015 !important; }
.risk-low {
    background: linear-gradient(135deg, rgba(234, 252, 240, 0.38) 0%, rgba(234, 252, 240, 0.22) 100%) !important;
    border: 1.5px solid rgba(52, 199, 89, 0.35) !important;
    border-top: 1px solid rgba(255, 255, 255, 0.85) !important;
}
.risk-low h3 { color: #248A3D !important; }
.risk-medium {
    background: linear-gradient(135deg, rgba(255, 248, 230, 0.38) 0%, rgba(255, 248, 230, 0.22) 100%) !important;
    border: 1.5px solid rgba(255, 149, 0, 0.35) !important;
    border-top: 1px solid rgba(255, 255, 255, 0.85) !important;
}
.risk-medium h3 { color: #C93400 !important; }

/* ── Metric Summary Tiles (Liquid Frosted Glass) ────────── */
.metric-tile {
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.38) 0%, rgba(255, 255, 255, 0.22) 100%) !important;
    backdrop-filter: blur(28px) saturate(210%) !important;
    -webkit-backdrop-filter: blur(28px) saturate(210%) !important;
    border-radius: 20px !important;
    border: 1px solid rgba(255, 255, 255, 0.45) !important;
    border-top: 1px solid rgba(255, 255, 255, 0.75) !important;
    padding: 1.1rem 1.2rem !important;
    text-align: center;
    box-shadow: 0 10px 32px 0 rgba(0, 0, 0, 0.06), inset 0 1px 1px 0 rgba(255, 255, 255, 0.7) !important;
}
.metric-tile .value {
    font-size: 2rem !important;
    font-weight: 800 !important;
    font-variant-numeric: tabular-nums;
    letter-spacing: -0.02em;
}
.metric-tile .label {
    font-size: 0.78rem !important;
    color: #8E8E93 !important;
    margin-top: 0.25rem;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.4px;
}

/* ── Medical Disclaimer Box (Liquid Frosted Glass) ──────── */
.disclaimer {
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.38) 0%, rgba(255, 255, 255, 0.22) 100%) !important;
    backdrop-filter: blur(28px) saturate(210%) !important;
    -webkit-backdrop-filter: blur(28px) saturate(210%) !important;
    border: 1px solid rgba(255, 255, 255, 0.45) !important;
    border-left: 5px solid #007AFF !important;
    border-top: 1px solid rgba(255, 255, 255, 0.75) !important;
    border-radius: 20px !important;
    padding: 1.2rem 1.5rem !important;
    font-size: 0.88rem !important;
    color: #3A3A3C !important;
    line-height: 1.55;
    margin-top: 1.5rem !important;
    box-shadow: 0 10px 32px 0 rgba(0, 0, 0, 0.06), inset 0 1px 1px 0 rgba(255, 255, 255, 0.7) !important;
}

/* ── Force all Markdown text & headings to high-contrast Apple dark text ── */
div[data-testid="stMarkdownContainer"] h1,
div[data-testid="stMarkdownContainer"] h2,
div[data-testid="stMarkdownContainer"] h3,
div[data-testid="stMarkdownContainer"] h4,
div[data-testid="stMarkdownContainer"] h5,
div[data-testid="stMarkdownContainer"] h6,
div[data-testid="stMarkdownContainer"] p,
div[data-testid="stMarkdownContainer"] span {
    color: #1C1C1E !important;
}
.hero-banner div[data-testid="stMarkdownContainer"] *,
.hero-banner * {
    color: #FFFFFF !important;
}

/* ── Primary Action Button (Apple Blue Pill) ── */
.stButton > button[kind="primary"] {
    background: #007AFF !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 999px !important;
    font-weight: 700 !important;
    font-size: 1.05rem !important;
    padding: 0.75rem 2rem !important;
    letter-spacing: -0.01em !important;
    transition: all 0.18s cubic-bezier(0.16, 1, 0.3, 1) !important;
    box-shadow: 0 6px 18px rgba(0, 122, 255, 0.32) !important;
}
.stButton > button[kind="primary"]:hover {
    background: #0066DD !important;
    box-shadow: 0 8px 24px rgba(0, 122, 255, 0.42) !important;
    transform: translateY(-1px) !important;
}
.stButton > button[kind="primary"]:active {
    transform: scale(0.98) !important;
    box-shadow: 0 2px 6px rgba(0, 122, 255, 0.2) !important;
}

/* ── Developer Sidebar Drawer (High Transparency Liquid Glass) ── */
section[data-testid="stSidebar"] {
    background-color: rgba(248, 248, 250, 0.55) !important;
    backdrop-filter: blur(30px) saturate(220%) !important;
    -webkit-backdrop-filter: blur(30px) saturate(220%) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.6) !important;
    box-shadow: 4px 0 24px rgba(0, 0, 0, 0.04) !important;
}
section[data-testid="stSidebar"] .block-container {
    padding-top: 1.2rem !important;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] h4,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label {
    color: #1C1C1E !important;
}

/* ── Expanders & Accordions (Liquid Frosted Glass) ───────── */
details {
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.38) 0%, rgba(255, 255, 255, 0.22) 100%) !important;
    backdrop-filter: blur(28px) saturate(210%) !important;
    -webkit-backdrop-filter: blur(28px) saturate(210%) !important;
    border-radius: 20px !important;
    border: 1px solid rgba(255, 255, 255, 0.45) !important;
    border-top: 1px solid rgba(255, 255, 255, 0.75) !important;
    box-shadow: 0 10px 32px 0 rgba(0, 0, 0, 0.04), inset 0 1px 1px 0 rgba(255, 255, 255, 0.7) !important;
    margin-bottom: 0.9rem !important;
    overflow: hidden;
}
summary {
    font-weight: 600 !important;
    color: #1C1C1E !important;
    padding: 0.85rem 1.1rem !important;
}

/* ── Metrics in Developer Panel (Liquid Frosted Glass) ──── */
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.38) 0%, rgba(255, 255, 255, 0.22) 100%) !important;
    backdrop-filter: blur(28px) saturate(210%) !important;
    -webkit-backdrop-filter: blur(28px) saturate(210%) !important;
    border-radius: 16px !important;
    border: 1px solid rgba(255, 255, 255, 0.45) !important;
    border-top: 1px solid rgba(255, 255, 255, 0.75) !important;
    padding: 0.85rem 1rem !important;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.03), inset 0 1px 1px 0 rgba(255, 255, 255, 0.7) !important;
}
div[data-testid="stMetric"] label {
    color: #8E8E93 !important;
    font-weight: 600 !important;
    font-size: 0.74rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
}
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    color: #1C1C1E !important;
    font-weight: 700 !important;
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
    """Train a fresh pipeline on the full 8-feature set and return it with metrics."""
    df = _clean_df(df)
    X = df[ALL_FEATURES]
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

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(best_pipe, f)

    return best_pipe, best_name, ALL_FEATURES, pd.DataFrame(metrics_list)


@st.cache_resource
def load_model():
    """
    Load Diabetesmodel.pkl.
    If missing or corrupt, retrain from diabetes.csv automatically.
    """
    df = load_dataset()

    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, "rb") as f:
                pipeline = pickle.load(f)
            n_features = pipeline.n_features_in_
            if n_features == len(ALL_FEATURES):
                return pipeline, ALL_FEATURES, None
            elif n_features == 4:
                return pipeline, ["Glucose", "BloodPressure", "BMI", "Age"], None
            else:
                feature_names = ALL_FEATURES[:n_features]
                return pipeline, feature_names, None
        except Exception:
            pass

    pipeline, best_name, feature_names, metrics_df = _train_fresh_pipeline(df)
    return pipeline, feature_names, metrics_df


@st.cache_data
def compute_benchmark_metrics(df, feature_names):
    """Train and evaluate comparative models for Developer Analytics."""
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
#  PLOTTING HELPERS (Apple-Themed Light Mode)
# ─────────────────────────────────────────────
def _fig_to_buf(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    buf.seek(0)
    plt.close(fig)
    return buf


CHART_BG   = "#FFFFFF"
CHART_FG   = "#1C1C1E"
GRID_CLR   = "#E5E5EA"
PALETTE    = ["#007AFF", "#5856D6", "#34C759", "#FF9500", "#FF3B30", "#AF52DE"]


def plot_correlation_heatmap(df):
    fig, ax = plt.subplots(figsize=(9, 7))
    fig.set_facecolor(CHART_BG)
    ax.set_facecolor(CHART_BG)
    numeric = df.select_dtypes(include=[np.number])
    mask = np.triu(np.ones_like(numeric.corr(), dtype=bool))
    sns.heatmap(
        numeric.corr(), mask=mask, annot=True, fmt=".2f",
        cmap="RdBu_r", center=0, linewidths=.5,
        ax=ax, cbar_kws={"shrink": .8},
        annot_kws={"size": 9, "color": CHART_FG},
    )
    ax.tick_params(colors=CHART_FG, labelsize=9)
    ax.set_title("Feature Correlation Matrix", color=CHART_FG, fontsize=14, fontweight="bold", pad=14)
    return _fig_to_buf(fig)


def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.set_facecolor(CHART_BG)
    ax.set_facecolor(CHART_BG)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Non-Diabetic", "Diabetic"],
        yticklabels=["Non-Diabetic", "Diabetic"],
        ax=ax, annot_kws={"size": 14},
    )
    ax.set_xlabel("Predicted", color=CHART_FG, fontsize=11)
    ax.set_ylabel("Actual", color=CHART_FG, fontsize=11)
    ax.set_title(title, color=CHART_FG, fontsize=13, fontweight="bold", pad=10)
    ax.tick_params(colors=CHART_FG)
    return _fig_to_buf(fig)


def plot_feature_distributions(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.drop(TARGET, errors="ignore")
    n = len(numeric_cols)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 3.2))
    fig.set_facecolor(CHART_BG)
    axes = axes.flatten()
    for i, col in enumerate(numeric_cols):
        ax = axes[i]
        ax.set_facecolor("#FAFAFA")
        ax.hist(df[col].dropna(), bins=25, color=PALETTE[i % len(PALETTE)], alpha=.85, edgecolor="#FFFFFF")
        ax.set_title(col, color=CHART_FG, fontsize=11, fontweight="bold")
        ax.tick_params(colors=CHART_FG, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(GRID_CLR)
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    fig.tight_layout()
    return _fig_to_buf(fig)


def plot_outcome_distribution(df):
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.set_facecolor(CHART_BG)
    ax.set_facecolor("#FAFAFA")
    counts = df[TARGET].value_counts().sort_index()
    bars = ax.bar(["Non-Diabetic (0)", "Diabetic (1)"], counts.values,
                  color=[PALETTE[2], PALETTE[4]], edgecolor=CHART_BG, width=.55)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 8, str(int(h)),
                ha="center", va="bottom", color=CHART_FG, fontweight="bold", fontsize=13)
    ax.set_ylabel("Count", color=CHART_FG)
    ax.set_title("Class Balance (Outcome)", color=CHART_FG, fontsize=13, fontweight="bold", pad=10)
    ax.tick_params(colors=CHART_FG)
    for spine in ax.spines.values():
        spine.set_color(GRID_CLR)
    return _fig_to_buf(fig)


def plot_benchmark_bars(metrics_df):
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.set_facecolor(CHART_BG)
    ax.set_facecolor("#FAFAFA")
    x = np.arange(len(metrics_df))
    w = 0.2
    for i, col in enumerate(["Accuracy", "Precision", "Recall", "F1"]):
        bars = ax.bar(x + i * w, metrics_df[col], w, label=col, color=PALETTE[i], alpha=.9)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.015, f"{h*100:.0f}%",
                    ha="center", va="bottom", fontsize=8, color=CHART_FG, rotation=90)
    ax.set_xticks(x + 1.5 * w)
    ax.set_xticklabels(metrics_df["Model"], color=CHART_FG, fontweight="600")
    ax.set_ylabel("Score", color=CHART_FG)
    ax.set_ylim(0, 1.15)
    ax.set_title("Model Performance Benchmark", color=CHART_FG, fontsize=14, fontweight="bold", pad=12)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.6)
    ax.tick_params(colors=CHART_FG)
    for spine in ax.spines.values():
        spine.set_color(GRID_CLR)
    return _fig_to_buf(fig)


def plot_risk_gauge(probability):
    """Semicircular polar gauge showing risk probability with Apple Health styling."""
    fig, ax = plt.subplots(figsize=(4.5, 2.8), subplot_kw={"projection": "polar"})
    fig.set_facecolor(CHART_BG)
    ax.set_facecolor(CHART_BG)

    theta_bg = np.linspace(np.pi, 0, 200)
    ax.fill_between(theta_bg, 0.6, 1.0, color="#E5E5EA", alpha=.5)

    # Green zone: 0 to 40%
    th_green = np.linspace(np.pi, np.pi * 0.6, 100)
    ax.fill_between(th_green, 0.62, 0.98, color="#34C759", alpha=.55)

    # Orange zone: 40% to 65%
    th_orange = np.linspace(np.pi * 0.6, np.pi * 0.35, 100)
    ax.fill_between(th_orange, 0.62, 0.98, color="#FF9500", alpha=.55)

    # Red zone: 65% to 100%
    th_red = np.linspace(np.pi * 0.35, 0, 100)
    ax.fill_between(th_red, 0.62, 0.98, color="#FF3B30", alpha=.55)

    # Needle
    needle_theta = np.pi * (1.0 - np.clip(probability, 0.0, 1.0))
    ax.plot([needle_theta, needle_theta], [0, 0.88], color="#1C1C1E", linewidth=3.5, solid_capstyle="round")
    ax.plot([needle_theta], [0], marker="o", markersize=10, color="#1C1C1E")

    ax.set_thetamin(0)
    ax.set_thetamax(180)
    ax.set_ylim(0, 1.05)
    ax.axis("off")

    fig.text(0.5, 0.18, f"{probability*100:.1f}%", ha="center", va="center",
             fontsize=26, fontweight="800", color=CHART_FG)
    fig.text(0.5, 0.04, "Diabetes Risk Score", ha="center", va="center",
             fontsize=10, fontweight="600", color="#8E8E93")
    fig.text(0.12, 0.28, "Low", ha="center", fontsize=9, fontweight="700", color="#34C759")
    fig.text(0.88, 0.28, "High", ha="center", fontsize=9, fontweight="700", color="#FF3B30")

    return _fig_to_buf(fig)


# ─────────────────────────────────────────────
#  MAIN APPLICATION
# ─────────────────────────────────────────────

# ─────────────────────────────────────────────
#  SYNCHRONIZED DUAL-INPUT STATE HANDLERS
# ─────────────────────────────────────────────
def sync_from_slider(feat):
    """Update canonical value and numeric box when slider thumb moves."""
    val = st.session_state[f"slider_{feat}"]
    st.session_state[f"input_{feat}"] = val
    st.session_state[f"num_{feat}"] = val


def sync_from_num(feat):
    """Update canonical value and slider when numeric box input changes."""
    meta = FEATURE_META[feat]
    val = st.session_state[f"num_{feat}"]
    # Clamp bounds and round based on step increment to prevent feedback loops
    val = max(meta["min"], min(meta["max"], val))
    val = meta["type"](round(val, 2) if meta["type"] == float else round(val))
    st.session_state[f"input_{feat}"] = val
    st.session_state[f"slider_{feat}"] = val
    st.session_state[f"num_{feat}"] = val

def main():
    if "predict_counter" not in st.session_state:
        st.session_state.predict_counter = 0

    # Initialize bidirectional synchronized state for all features
    for feat in ALL_FEATURES:
        meta = FEATURE_META[feat]
        default_val = meta["default"]
        if f"input_{feat}" not in st.session_state:
            typed_val = meta["type"](default_val)
            st.session_state[f"input_{feat}"] = typed_val
            st.session_state[f"slider_{feat}"] = typed_val
            st.session_state[f"num_{feat}"] = typed_val

    # Hero Banner
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-badge">RCAI Classroom Project</div>
        <h1>🩺 AI Diabetes Risk Assessment</h1>
        <p>Machine-learning powered diabetes screening using the Pima Indians Diabetes dataset.
        Enter clinical parameters to receive an AI-assisted risk estimate in real time.</p>
    </div>
    """, unsafe_allow_html=True)

    df_raw = load_dataset()
    pipeline, feature_names, _ = load_model()

    # ═══════════════════════════════════════════
    #  DEVELOPER SIDEBAR DRAWER
    # ═══════════════════════════════════════════
    with st.sidebar:
        st.markdown("""
        <div style="padding: 0.6rem 0 0.8rem 0;">
            <div style="font-size: 1.15rem; font-weight: 800; color: #1C1C1E;">🛠️ Developer Tools & Model Telemetry</div>
            <div style="font-size: 0.72rem; color: #8E8E93; font-weight: 700; letter-spacing: 0.6px; margin-top: 2px;">
                INTERNAL DATA INSPECTION & BENCHMARKS
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        st.markdown('<div class="section-label">QUICK PRESETS</div>', unsafe_allow_html=True)
        selected_preset = st.selectbox(
            "Load Clinical Profile",
            list(PRESETS.keys()),
            index=0,
            help="Pre-fill patient metrics with a representative profile.",
            label_visibility="collapsed",
        )

        # Synchronize preset change into session state sliders
        if "active_preset_name" not in st.session_state:
            st.session_state.active_preset_name = selected_preset

        if selected_preset != st.session_state.active_preset_name:
            st.session_state.active_preset_name = selected_preset
            preset_data = PRESETS[selected_preset]
            if preset_data:
                for k, v in preset_data.items():
                    target_type = FEATURE_META[k]["type"]
                    val = target_type(v)
                    st.session_state[f"input_{k}"] = val
                    st.session_state[f"slider_{k}"] = val
                    st.session_state[f"num_{k}"] = val
            st.rerun()

        preset = PRESETS[selected_preset]

        st.caption(f"Active Features: {', '.join(feature_names)}")
        st.markdown("---")

        # ── Developer Data Explorer Expander ──
        with st.expander("📊  Data Explorer", expanded=False):
            st.markdown('<div class="section-label">DATASET OVERVIEW</div>', unsafe_allow_html=True)
            m1, m2 = st.columns(2)
            with m1:
                st.metric("Total Records", f"{len(df_raw):,}")
                st.metric("Diabetic (1)", int(df_raw[TARGET].sum()))
            with m2:
                st.metric("Features", len(df_raw.columns) - 1)
                st.metric("Non-Diabetic (0)", int((df_raw[TARGET] == 0).sum()))

            st.markdown("**Sample Records (Head 10)**")
            st.dataframe(df_raw.head(10), use_container_width=True, hide_index=True)

            st.markdown("**Summary Statistics**")
            st.dataframe(df_raw.describe().T.round(2), use_container_width=True)

            st.markdown("---")
            st.markdown("**Feature Distributions**")
            st.image(plot_feature_distributions(df_raw), use_container_width=True)

            st.markdown("**Class Balance**")
            st.image(plot_outcome_distribution(df_raw), use_container_width=True)

            st.markdown("---")
            st.markdown("**Correlation Matrix**")
            st.image(plot_correlation_heatmap(df_raw), use_container_width=True)

        st.markdown("---")

        # ── Developer Model Analytics Expander ──
        with st.expander("🧪  Model Analytics", expanded=False):
            st.markdown('<div class="section-label">MODEL BENCHMARKS</div>', unsafe_allow_html=True)
            st.caption("Evaluated on 20% held-out test split (Stratified)")

            with st.spinner("Computing comparative model benchmarks..."):
                metrics_df, preds = compute_benchmark_metrics(df_raw, feature_names)

            fmt_df = metrics_df.copy()
            for col in ["Accuracy", "Precision", "Recall", "F1"]:
                fmt_df[col] = fmt_df[col].apply(lambda v: f"{v*100:.2f}%")
            st.dataframe(fmt_df, use_container_width=True, hide_index=True)

            st.image(plot_benchmark_bars(metrics_df), use_container_width=True)

            st.markdown("---")
            st.markdown("**Confusion Matrices**")
            for name, (y_true, y_pred) in preds.items():
                buf = plot_confusion_matrix(y_true, y_pred, title=name)
                st.image(buf, use_container_width=True)

    # ═══════════════════════════════════════════
    #  MAIN INTERFACE — PATIENT METRICS
    # ═══════════════════════════════════════════
    st.markdown('<div class="section-label">PATIENT METRICS</div>', unsafe_allow_html=True)
    st.markdown("### Enter Clinical Parameters")

    primary_feats   = [f for f in feature_names if FEATURE_META[f]["primary"]]
    secondary_feats = [f for f in feature_names if not FEATURE_META[f]["primary"]]

    input_values = {}

    # ── Primary Dual-Input Parameters (Liquid Glass Card) ──
    with st.container(border=True):
        col_a, col_b = st.columns(2)
        for idx, feat in enumerate(primary_feats):
            meta = FEATURE_META[feat]
            container = col_a if idx % 2 == 0 else col_b
            with container:
                unit_str = f"({meta['unit']})" if meta["unit"] else ""
                st.markdown(f'''
                <div class="metric-row-header">
                    <span class="metric-row-title">{meta["label"]}</span>
                    <span class="metric-row-unit">{unit_str}</span>
                </div>
                ''', unsafe_allow_html=True)
                
                c_slide, c_num = st.columns([3.4, 1.25])
                with c_slide:
                    st.slider(
                        label=meta["label"],
                        min_value=meta["type"](meta["min"]),
                        max_value=meta["type"](meta["max"]),
                        value=meta["type"](st.session_state[f"slider_{feat}"]),
                        step=meta["type"](meta["step"]) if meta["type"] == float else meta["step"],
                        key=f"slider_{feat}",
                        on_change=sync_from_slider,
                        args=(feat,),
                        label_visibility="collapsed",
                    )
                with c_num:
                    st.number_input(
                        label=meta["label"],
                        min_value=meta["type"](meta["min"]),
                        max_value=meta["type"](meta["max"]),
                        value=meta["type"](st.session_state[f"num_{feat}"]),
                        step=meta["type"](meta["step"]) if meta["type"] == float else meta["step"],
                        key=f"num_{feat}",
                        on_change=sync_from_num,
                        args=(feat,),
                        label_visibility="collapsed",
                    )
                input_values[feat] = st.session_state[f"input_{feat}"]

    # ── Secondary Dual-Input Parameters (Collapsible Liquid Glass Expander) ──
    if secondary_feats:
        with st.expander("🔧  Advanced Clinical Parameters (click to expand)", expanded=False):
            sec_cols = st.columns(2)
            for idx, feat in enumerate(secondary_feats):
                meta = FEATURE_META[feat]
                with sec_cols[idx % 2]:
                    unit_str = f"({meta['unit']})" if meta["unit"] else ""
                    st.markdown(f'''
                    <div class="metric-row-header">
                        <span class="metric-row-title">{meta["label"]}</span>
                        <span class="metric-row-unit">{unit_str}</span>
                    </div>
                    ''', unsafe_allow_html=True)
                    
                    c_slide, c_num = st.columns([3.4, 1.25])
                    with c_slide:
                        st.slider(
                            label=meta["label"],
                            min_value=meta["type"](meta["min"]),
                            max_value=meta["type"](meta["max"]),
                            value=meta["type"](st.session_state[f"slider_{feat}"]),
                            step=meta["type"](meta["step"]) if meta["type"] == float else meta["step"],
                            key=f"slider_{feat}",
                            on_change=sync_from_slider,
                            args=(feat,),
                            label_visibility="collapsed",
                        )
                    with c_num:
                        st.number_input(
                            label=meta["label"],
                            min_value=meta["type"](meta["min"]),
                            max_value=meta["type"](meta["max"]),
                            value=meta["type"](st.session_state[f"num_{feat}"]),
                            step=meta["type"](meta["step"]) if meta["type"] == float else meta["step"],
                            key=f"num_{feat}",
                            on_change=sync_from_num,
                            args=(feat,),
                            label_visibility="collapsed",
                        )
                    input_values[feat] = st.session_state[f"input_{feat}"]

    st.markdown("")

    # ═══════════════════════════════════════════
    #  MAIN INTERFACE — CLINICAL REFERENCE RANGES
    # ═══════════════════════════════════════════
    st.markdown('<div class="section-label">CLINICAL REFERENCE RANGES</div>', unsafe_allow_html=True)
    range_cols = st.columns(len(CLINICAL_RANGES))
    for i, ref in enumerate(CLINICAL_RANGES):
        with range_cols[i]:
            st.markdown(f"""
            <div class="range-card">
                <div class="range-icon">{ref['icon']}</div>
                <div class="range-metric">{ref['metric']}</div>
                <div class="range-unit">{ref['unit']}</div>
                <div>
                    <span class="range-badge badge-normal">Normal: {ref['normal']}</span>
                </div>
                <div style="margin-top: 4px;">
                    <span class="range-badge badge-danger">Risk: {ref['concern']}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("")

    # Input validation checks
    warnings_list = []
    if input_values.get("Glucose", 999) == 0:
        warnings_list.append("⚠️ Fasting Glucose of 0 mg/dL is physiologically implausible; imputer will handle this.")
    if input_values.get("BloodPressure", 999) == 0:
        warnings_list.append("⚠️ Blood Pressure of 0 mm Hg is physiologically implausible; imputer will handle this.")
    if input_values.get("BMI", 999) <= 0:
        warnings_list.append("⚠️ BMI must be a positive number.")
    if input_values.get("Age", 1) < 1:
        warnings_list.append("⚠️ Age must be at least 1 year.")

    for w in warnings_list:
        st.warning(w)

    st.markdown("")

    # Action Button
    predict_btn = st.button("🩺  Run Assessment", type="primary", use_container_width=True)

    # ═══════════════════════════════════════════
    #  ASSESSMENT RESULTS & PERSISTENT AUTO-SCROLL
    # ═══════════════════════════════════════════
    if predict_btn:
        st.session_state.predict_counter += 1
        current_run_id = st.session_state.predict_counter

        # Target DOM anchor for scrolling
        st.markdown('<div id="assessment-result-anchor" style="padding-top: 1.5rem;"></div>', unsafe_allow_html=True)

        # ── Persistent Auto-Scroll Hook (executed via components.html iframe) ──
        # Guaranteed to re-fire smoothly on every button click without full page refresh
        components.html(
            f"""
            <script>
                (function() {{
                    const runId = {current_run_id};
                    setTimeout(function() {{
                        const el = window.parent.document.getElementById('assessment-result-anchor');
                        if (el) {{
                            el.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
                        }}
                    }}, 120);
                }})();
            </script>
            """,
            height=0,
            width=0,
        )

        # Build inference payload
        row = {feat: input_values.get(feat, FEATURE_META[feat]["default"]) for feat in feature_names}
        for zf in ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]:
            if zf in row and row[zf] == 0:
                row[zf] = np.nan
        patient_df = pd.DataFrame([row], columns=feature_names)

        prediction = pipeline.predict(patient_df)[0]
        if hasattr(pipeline, "predict_proba"):
            probs = pipeline.predict_proba(patient_df)[0]
        else:
            probs = np.array([1 - prediction, prediction], dtype=float)

        prob_diabetic     = float(probs[1])
        prob_non_diabetic = float(probs[0])

        st.markdown('<div class="section-label">RISK SUMMARY</div>', unsafe_allow_html=True)
        st.markdown("### 📋 Assessment Results")

        res_left, res_right = st.columns([3, 2])

        with res_left:
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
                <p style="font-size: 1.05rem; margin-bottom: 0.5rem; color: #1C1C1E;">
                    The model predicts a <strong>{prob_diabetic*100:.1f}%</strong> probability of diabetes
                    based on the provided clinical parameters.
                </p>
                <p style="font-size: 0.92rem; color: #636366; margin: 0;">
                    Prediction Classification: <strong>{'Diabetic (Class 1)' if prediction == 1 else 'Non-Diabetic (Class 0)'}</strong>
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="section-label" style="margin-top: 1.2rem;">KEY CLINICAL RISK FACTORS</div>', unsafe_allow_html=True)
            flags = []
            g = input_values.get("Glucose", 0)
            if g >= 126:
                flags.append("🔴 **Fasting Glucose ≥ 126 mg/dL** — Diagnostic of diabetes threshold")
            elif g >= 100:
                flags.append("🟡 **Fasting Glucose 100–125 mg/dL** — Impaired fasting glucose (Prediabetes)")

            bp = input_values.get("BloodPressure", 0)
            if bp >= 90:
                flags.append("🔴 **Blood Pressure ≥ 90 mm Hg** — Stage 2 Hypertension range")
            elif bp >= 80:
                flags.append("🟡 **Blood Pressure 80–89 mm Hg** — Prehypertension / Elevated")

            bmi_val = input_values.get("BMI", 0)
            if bmi_val >= 30:
                flags.append("🔴 **BMI ≥ 30 kg/m²** — Clinical Obesity (high insulin resistance risk)")
            elif bmi_val >= 25:
                flags.append("🟡 **BMI 25–29.9 kg/m²** — Overweight")

            age_val = input_values.get("Age", 0)
            if age_val >= 45:
                flags.append("🟡 **Age ≥ 45 years** — Heightened age-related metabolic risk")

            ins = input_values.get("Insulin", 0)
            if ins > 166:
                flags.append("🟡 **Insulin > 166 µU/mL** — Elevated fasting/post-load hyperinsulinemia")

            if flags:
                for f in flags:
                    st.markdown(f)
            else:
                st.success("✅ All evaluated metrics are within optimal clinical reference ranges.")

        with res_right:
            gauge_buf = plot_risk_gauge(prob_diabetic)
            st.image(gauge_buf, use_container_width=True)

            mcol1, mcol2 = st.columns(2)
            with mcol1:
                st.markdown(f"""
                <div class="metric-tile">
                    <div class="value" style="color: #34C759;">{prob_non_diabetic*100:.1f}%</div>
                    <div class="label">Non-Diabetic</div>
                </div>
                """, unsafe_allow_html=True)
            with mcol2:
                st.markdown(f"""
                <div class="metric-tile">
                    <div class="value" style="color: #FF3B30;">{prob_diabetic*100:.1f}%</div>
                    <div class="label">Diabetic</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("""
        <div class="disclaimer">
            <strong>⚕️ Medical Disclaimer:</strong> This clinical decision-support tool is an AI-assisted screening
            estimate developed for academic purposes using the Pima Indians Diabetes Dataset. It is
            <strong>not a substitute for professional medical diagnosis</strong>. Always consult a licensed healthcare
            provider for clinical evaluation, laboratory diagnostic testing, and therapeutic decisions.
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
