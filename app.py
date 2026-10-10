"""
app.py — AI Diabetes Risk Assessment
=====================================
Rocheston RCAI Classroom Project
A production-ready Streamlit web application for diabetes risk prediction
using machine learning trained on the Pima Indians Diabetes dataset.

Design: Apple Health UI Design System (iOS/macOS Human Interface Guidelines)
Layout: Balanced Dual-Column Desktop Architecture with Zero Vertical Scrolling
Icons: Apple Minimalist Vector SVG Line Icons (No Unicode Emojis)
Report: Clinical Single-Page PDF Generation via ReportLab
Performance: Zero-Latency Real-Time Client-Side Dual-Input Synchronization & Stepper Buffer
Geometry: Capsule Pill Slider Track with Seamless Rounded Fill Under Translucent Thumb
Inference: Dual-Model Routing & Dataset Median Imputation for Optional Advanced Biomarkers
"""

import os
import pickle
import warnings
from io import BytesIO
from datetime import datetime

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

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  CONFIGURATION & FEATURE METADATA
# ─────────────────────────────────────────────
DATA_PATH    = os.path.join(os.path.dirname(__file__), "diabetes.csv")
MODEL_PATH_8 = os.path.join(os.path.dirname(__file__), "Diabetesmodel.pkl")
MODEL_PATH_4 = os.path.join(os.path.dirname(__file__), "Diabetesmodel_4.pkl")

# Standard training feature vector order (Pima Indians dataset)
ALL_FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
]
PRIMARY_FEATURES  = ["Glucose", "BloodPressure", "BMI", "Age"]
ADVANCED_FEATURES = ["Insulin", "SkinThickness", "Pregnancies", "DiabetesPedigreeFunction"]
TARGET = "Outcome"

# Precomputed training population medians (biologically clean medians from diabetes.csv)
POPULATION_MEDIANS = {
    "Insulin": 125.0,
    "SkinThickness": 29.0,
    "Pregnancies": 0,
    "DiabetesPedigreeFunction": 0.3725,
}

FEATURE_META = {
    "Glucose": {
        "label": "Fasting Blood Glucose",
        "min": 0, "max": 300, "default": 120, "step": 1,
        "unit": "mg/dL", "type": int, "primary": True,
        "icon": "droplet", "badge": "icon-badge-rose",
        "norm_desc": "Normal Target: 70–99 mg/dL",
    },
    "BloodPressure": {
        "label": "Blood Pressure (Diastolic)",
        "min": 0, "max": 200, "default": 70, "step": 1,
        "unit": "mm Hg", "type": int, "primary": True,
        "icon": "heart_pulse", "badge": "icon-badge-crimson",
        "norm_desc": "Normal Target: < 80 mm Hg",
    },
    "BMI": {
        "label": "Body Mass Index (BMI)",
        "min": 10.0, "max": 60.0, "default": 25.0, "step": 0.1,
        "unit": "kg/m²", "type": float, "primary": True,
        "icon": "scale", "badge": "icon-badge-blue",
        "norm_desc": "Healthy Target: 18.5–24.9 kg/m²",
    },
    "Age": {
        "label": "Patient Age",
        "min": 1, "max": 120, "default": 30, "step": 1,
        "unit": "years", "type": int, "primary": True,
        "icon": "calendar", "badge": "icon-badge-amber",
        "norm_desc": "Standard Baseline: 18–44 yrs",
    },
    "Insulin": {
        "label": "2-Hour Serum Insulin",
        "min": 0, "max": 900, "default": 80, "step": 1,
        "unit": "µU/mL", "type": int, "primary": False,
        "icon": "syringe", "badge": "icon-badge-teal",
        "norm_desc": "Reference Target: 16–166 µU/mL",
        "skip_label": "Skip / Lab test not performed",
    },
    "SkinThickness": {
        "label": "Triceps Skin Fold Thickness",
        "min": 0, "max": 100, "default": 20, "step": 1,
        "unit": "mm", "type": int, "primary": False,
        "icon": "layers", "badge": "icon-badge-amber",
        "norm_desc": "Reference Target: 10–30 mm",
        "skip_label": "Skip / Caliper test not performed",
    },
    "Pregnancies": {
        "label": "Number of Pregnancies",
        "min": 0, "max": 20, "default": 1, "step": 1,
        "unit": "count", "type": int, "primary": False,
        "icon": "user", "badge": "icon-badge-purple",
        "norm_desc": "Obstetric Baseline: 0–2",
        "skip_label": "Skip / Not applicable (male or non-parous patient)",
    },
    "DiabetesPedigreeFunction": {
        "label": "Diabetes Pedigree Function",
        "min": 0.0, "max": 2.5, "default": 0.47, "step": 0.01,
        "unit": "score", "type": float, "primary": False,
        "icon": "dna", "badge": "icon-badge-purple",
        "norm_desc": "Low Hereditary Risk: < 0.50",
        "skip_label": "Skip / Family pedigree history unknown",
    },
}

PRESETS = {
    "Custom Profile": None,
    "Normal Clinical Profile": {
        "Pregnancies": 1, "Glucose": 88, "BloodPressure": 68, "SkinThickness": 22,
        "Insulin": 45, "BMI": 22.5, "DiabetesPedigreeFunction": 0.25, "Age": 28,
    },
    "Borderline Risk Profile": {
        "Pregnancies": 3, "Glucose": 118, "BloodPressure": 78, "SkinThickness": 28,
        "Insulin": 130, "BMI": 27.8, "DiabetesPedigreeFunction": 0.45, "Age": 42,
    },
    "Elevated Risk Profile": {
        "Pregnancies": 6, "Glucose": 168, "BloodPressure": 88, "SkinThickness": 38,
        "Insulin": 240, "BMI": 36.2, "DiabetesPedigreeFunction": 0.88, "Age": 54,
    },
    "Clinical Obesity & Hypertension Profile": {
        "Pregnancies": 2, "Glucose": 142, "BloodPressure": 96, "SkinThickness": 42,
        "Insulin": 195, "BMI": 41.5, "DiabetesPedigreeFunction": 0.62, "Age": 49,
    },
}

# ─────────────────────────────────────────────
#  VECTOR SVG ICONS (Apple Minimalist Line Style)
# ─────────────────────────────────────────────
SVG_ICONS = {
    "activity": '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#007AFF" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline></svg>',
    "droplet": '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#FF2D55" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2.69l5.66 5.66a8 8 0 1 1-11.31 0z"></path></svg>',
    "heart_pulse": '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#FF3B30" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M19 14c1.49-1.46 3-3.21 3-5.5A5.5 5.5 0 0 0 16.5 3c-1.76 0-3 .5-4.5 2-1.5-1.5-2.74-2-4.5-2A5.5 5.5 0 0 0 2 8.5c0 2.3 1.5 4.05 3 5.5l7 7Z"></path><path d="M3.22 12H9.5l.5-1 2 4.5 2-7 1.5 3.5h5.27"></path></svg>',
    "scale": '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#007AFF" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m16 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z"></path><path d="m2 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z"></path><path d="M7 21h10"></path><path d="M12 3v18"></path><path d="M3 7h2c2 0 5-1 7-2 2 1 5 2 7 2h2"></path></svg>',
    "calendar": '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#FF9500" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="18" height="18" x="3" y="4" rx="2" ry="2"></rect><line x1="16" x2="16" y1="2" y2="6"></line><line x1="8" x2="8" y1="2" y2="6"></line><line x1="3" x2="21" y1="10" y2="10"></line></svg>',
    "syringe": '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#30B0C7" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m18 2 4 4"></path><path d="m17 7 3-3"></path><path d="M19 9 8.7 19.3c-1 1-2.5 1-3.4 0l-.6-.6c-1-1-1-2.5 0-3.4L15 5"></path><path d="m9 11 4 4"></path><path d="m5 19-3 3"></path><path d="m14 4 6 6"></path></svg>',
    "layers": '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#FF9500" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="12 2 2 7 12 12 22 7 12 2"></polygon><polyline points="2 17 12 22 22 17"></polyline><polyline points="2 12 12 17 22 12"></polyline></svg>',
    "user": '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#AF52DE" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2"></path><circle cx="12" cy="7" r="4"></circle></svg>',
    "dna": '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#AF52DE" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m8 16 1.5-1.5"></path><path d="m14.5 9.5 1.5-1.5"></path><path d="m17 6-2.5-2.5"></path><path d="m9.5 14.5-2.5-2.5"></path><path d="m20 9-4.5 4.5"></path><path d="m8.5 19.5-4.5-4.5"></path><path d="m14 4-9.5 9.5a5 5 0 0 0 7 7l9.5-9.5a5 5 0 0 0-7-7Z"></path></svg>',
    "check": '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#34C759" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"></polyline></svg>',
    "shield_check": '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#34C759" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path><path d="m9 12 2 2 4-4"></path></svg>',
    "alert_triangle": '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#FF3B30" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>',
    "alert_circle": '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#FF9500" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>',
    "info": '<svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="#007AFF" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg>',
    "flask": '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#007AFF" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10 2v7.31L4.1 19.34A2 2 0 0 0 5.8 22h12.4a2 2 0 0 0 1.7-2.66L14 9.31V2"></path><line x1="8.5" y1="2" x2="15.5" y2="2"></line><line x1="14" y1="14" x2="10" y2="14"></line></svg>',
}

def render_badge(icon_name, badge_class):
    svg = SVG_ICONS.get(icon_name, SVG_ICONS["activity"])
    return f'<div class="icon-badge {badge_class}">{svg}</div>'

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
/* ── Typography & Apple System Font Stack ────────────────── */
html, body, [class*="css"], [class*="st-"] {
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text",
                 "Helvetica Neue", Roboto, sans-serif !important;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
    color: #1C1C1E;
}

/* ── App Canvas Background & Refraction Orbs ─────────────── */
.stApp {
    background-color: #F2F2F7 !important;
    background-attachment: fixed;
}
.ambient-orb-blue {
    position: fixed;
    top: 3%;
    left: 8%;
    width: 480px;
    height: 480px;
    background: radial-gradient(circle, rgba(0, 122, 255, 0.16) 0%, rgba(0, 122, 255, 0) 70%);
    filter: blur(60px);
    z-index: 0;
    pointer-events: none;
    animation: orbFloat1 18s ease-in-out infinite alternate;
}
.ambient-orb-cyan {
    position: fixed;
    bottom: 8%;
    right: 6%;
    width: 520px;
    height: 520px;
    background: radial-gradient(circle, rgba(48, 176, 199, 0.15) 0%, rgba(48, 176, 199, 0) 70%);
    filter: blur(70px);
    z-index: 0;
    pointer-events: none;
    animation: orbFloat2 22s ease-in-out infinite alternate;
}
@keyframes orbFloat1 {
    0%   { transform: translate(0, 0) scale(1); }
    100% { transform: translate(50px, 40px) scale(1.08); }
}
@keyframes orbFloat2 {
    0%   { transform: translate(0, 0) scale(1); }
    100% { transform: translate(-45px, -35px) scale(0.95); }
}

/* ── Layout Tuning: Zero Vertical Scroll Desktop Balance ─── */
.block-container {
    padding-top: 1.0rem !important;
    padding-bottom: 2.0rem !important;
    padding-left: 1.75rem !important;
    padding-right: 1.75rem !important;
    max-width: 1440px !important;
    position: relative;
    z-index: 1;
}

/* ── Hero Telemetry Banner ──────────────────────────────── */
.hero-banner {
    background: linear-gradient(135deg, #007AFF 0%, #5856D6 55%, #AF52DE 100%);
    border-radius: 20px;
    padding: 1.25rem 1.75rem;
    color: #FFFFFF !important;
    box-shadow: 0 10px 28px rgba(0, 122, 255, 0.22);
    margin-bottom: 0.85rem;
    position: relative;
    overflow: hidden;
}
.hero-banner h1 {
    font-size: 1.85rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.025em;
    margin-bottom: 0.25rem !important;
    color: #FFFFFF !important;
}
.hero-banner p {
    font-size: 0.94rem !important;
    opacity: 0.95;
    max-width: 760px;
    line-height: 1.45;
    margin: 0 !important;
    color: #FFFFFF !important;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(255, 255, 255, 0.2);
    border: 1px solid rgba(255, 255, 255, 0.4);
    border-radius: 999px;
    padding: 0.20rem 0.75rem;
    font-size: 0.70rem;
    font-weight: 700;
    letter-spacing: 0.6px;
    color: #FFFFFF !important;
    margin-bottom: 0.45rem;
    text-transform: uppercase;
}

/* ── Section Category Labels (Apple Health Small Caps) ──── */
.section-label {
    font-size: 0.74rem !important;
    font-weight: 700 !important;
    color: #8E8E93 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.8px !important;
    margin-top: 0.15rem !important;
    margin-bottom: 0.45rem !important;
    padding-left: 2px !important;
}

/* ── True Apple Liquid Glass Container ──────────────────── */
.apple-glass-card {
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.42) 0%, rgba(255, 255, 255, 0.26) 100%) !important;
    backdrop-filter: blur(28px) saturate(210%) !important;
    -webkit-backdrop-filter: blur(28px) saturate(210%) !important;
    border-radius: 20px !important;
    border: 1px solid rgba(255, 255, 255, 0.50) !important;
    border-top: 1px solid rgba(255, 255, 255, 0.85) !important;
    box-shadow: 0 10px 32px 0 rgba(0, 0, 0, 0.05), inset 0 1px 1px 0 rgba(255, 255, 255, 0.8) !important;
    padding: 1.15rem 1.35rem !important;
    margin-bottom: 0.85rem !important;
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.40) 0%, rgba(255, 255, 255, 0.24) 100%) !important;
    backdrop-filter: blur(28px) saturate(210%) !important;
    -webkit-backdrop-filter: blur(28px) saturate(210%) !important;
    border-radius: 20px !important;
    border: 1px solid rgba(255, 255, 255, 0.50) !important;
    border-top: 1px solid rgba(255, 255, 255, 0.85) !important;
    box-shadow: 0 10px 32px 0 rgba(0, 0, 0, 0.05), inset 0 1px 1px 0 rgba(255, 255, 255, 0.8) !important;
    padding: 1.15rem 1.35rem !important;
    margin-bottom: 0.85rem !important;
}

/* ── Frosted Circular Icon Badges ───────────────────────── */
.icon-badge {
    width: 32px;
    height: 32px;
    border-radius: 50%;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    transition: transform 0.2s ease;
}
.icon-badge-rose    { background: rgba(255, 45, 85, 0.12); }
.icon-badge-crimson { background: rgba(255, 59, 48, 0.12); }
.icon-badge-blue    { background: rgba(0, 122, 255, 0.12); }
.icon-badge-amber   { background: rgba(255, 149, 0, 0.12); }
.icon-badge-green   { background: rgba(52, 199, 89, 0.12); }
.icon-badge-teal    { background: rgba(48, 176, 199, 0.12); }
.icon-badge-purple  { background: rgba(175, 82, 222, 0.12); }

/* ── Metric Dual-Input Header Row ───────────────────────── */
.metric-row-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.12rem;
    padding-left: 2px;
    padding-right: 2px;
}
.metric-title-group {
    display: flex;
    align-items: center;
    gap: 8px;
}
.metric-row-title {
    font-weight: 700;
    font-size: 0.89rem;
    color: #1C1C1E;
    letter-spacing: -0.01em;
}
.metric-row-unit {
    font-size: 0.72rem;
    font-weight: 600;
    color: #8E8E93;
    letter-spacing: 0.3px;
}

/* ══════════════════════════════════════════════════════════
   1 & 2. SLIDER TRACK GEOMETRY: SEAMLESS CAPSULE & PILL ROUNDING
   Eliminates all sharp 90-degree internal edges under translucent thumb
   ══════════════════════════════════════════════════════════ */
div[data-testid="stSlider"] {
    padding-top: 3px !important;
    padding-bottom: 2px !important;
}
div[data-testid="stSlider"] div[data-baseweb="slider"] {
    height: 24px !important;
    display: flex !important;
    align-items: center !important;
}
div[data-testid="stSlider"] div[data-rac][data-orientation="horizontal"] {
    display: block !important;
    width: 100% !important;
    height: 38px !important;
    position: relative !important;
}

/* Main Track Container: Perfect 6px capsule with complete clipping */
div[data-testid="stSlider"] div[class*="efbyxod5"],
div[data-testid="stSlider"] div[data-rac] > div[class*="efbyxod5"],
div[data-testid="stSlider"] div[data-baseweb="slider"] > div {
    height: 6px !important;
    border-radius: 9999px !important;
    background-color: #E5E5EA !important;
    background-image: none !important;
    overflow: hidden !important; /* Clips all fills inside smooth capsule */
    position: relative !important;
    top: calc(50% - 3px) !important;
    width: 100% !important;
    box-shadow: inset 0 1px 2px rgba(0, 0, 0, 0.05) !important;
}

/* Active colored progress fill: Seamless rounded capsule end */
div[data-testid="stSlider"] div[class*="efbyxod5"]::before {
    content: "" !important;
    position: absolute !important;
    top: 0 !important;
    left: 0 !important;
    bottom: 0 !important;
    width: var(--slider-fill, 50%) !important;
    background: #007AFF !important;
    border-radius: 9999px !important;
    border-top-right-radius: 9999px !important;
    border-bottom-right-radius: 9999px !important;
    box-shadow: 0 0 6px rgba(0, 122, 255, 0.35) !important;
    pointer-events: none !important;
    transition: width 0.03s linear !important;
}

/* Slider Thumb (Translucent Liquid Glass Pill: 38px x 24px, 13px radius) */
div[data-testid="stSlider"] div[role="slider"],
div[data-testid="stSlider"] div[class*="efbyxod3"],
div[data-testid="stSlider"] div[data-rac][style*="position: absolute"] {
    width: 38px !important;
    height: 24px !important;
    border-radius: 13px !important;
    background: rgba(235, 235, 240, 0.70) !important;
    backdrop-filter: blur(16px) saturate(180%) !important;
    -webkit-backdrop-filter: blur(16px) saturate(180%) !important;
    border: 1.5px solid rgba(255, 255, 255, 0.95) !important;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.12) !important;
    cursor: grab !important;
    outline: none !important;
    top: 50% !important;
    transition: transform 0.15s cubic-bezier(0.16, 1, 0.3, 1), box-shadow 0.15s ease !important;
}

/* Hover: Subtle scale bump */
div[data-testid="stSlider"] div[role="slider"]:hover,
div[data-testid="stSlider"] div[class*="efbyxod3"]:hover,
div[data-testid="stSlider"] div[data-rac][style*="position: absolute"]:hover {
    transform: translate(-50%, -50%) scale(1.05) !important;
    cursor: grab !important;
}

/* Active / Dragging: Tactile expansion with Apple blue aura */
div[data-testid="stSlider"] div[role="slider"]:active,
div[data-testid="stSlider"] div[role="slider"]:focus,
div[data-testid="stSlider"] div[class*="efbyxod3"]:active,
div[data-testid="stSlider"] div[class*="efbyxod3"]:focus,
div[data-testid="stSlider"] div[data-rac][style*="position: absolute"]:active,
div[data-testid="stSlider"] div[data-rac][style*="position: absolute"]:focus {
    transform: translate(-50%, -50%) scale(1.10) !important;
    cursor: grabbing !important;
    box-shadow: 0 4px 14px rgba(0, 122, 255, 0.28) !important;
}

/* Hide default tooltip over thumb to maintain clean appearance */
div[data-testid="stThumbValue"],
div[data-testid="stSliderThumbValue"] {
    display: none !important;
}

div[data-testid="stSliderTickBarMin"],
div[data-testid="stSliderTickBarMax"],
div[data-testid="stSliderTickBar"] p {
    color: #8E8E93 !important;
    font-size: 0.73rem !important;
    font-weight: 500 !important;
}

/* ── Synchronized Number Input Box ──────────────────────── */
div[data-testid="stNumberInput"] {
    padding-top: 0px !important;
    margin-top: 0px !important;
}
div[data-testid="stNumberInput"] div[data-baseweb="input"],
div[data-testid="stNumberInput"] div[class*="st-emotion-cache"] {
    border-radius: 12px !important;
}
div[data-testid="stNumberInput"] input {
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.40) 0%, rgba(255, 255, 255, 0.22) 100%) !important;
    backdrop-filter: blur(24px) saturate(200%) !important;
    -webkit-backdrop-filter: blur(24px) saturate(200%) !important;
    border-radius: 12px !important;
    border: 1px solid rgba(255, 255, 255, 0.50) !important;
    border-top: 1px solid rgba(255, 255, 255, 0.85) !important;
    box-shadow: 0 3px 10px 0 rgba(0, 0, 0, 0.03), inset 0 1px 1px 0 rgba(255, 255, 255, 0.7) !important;
    font-weight: 700 !important;
    font-size: 0.93rem !important;
    color: #1C1C1E !important;
    text-align: center !important;
    font-variant-numeric: tabular-nums !important;
    height: 38px !important;
    transition: all 0.2s ease !important;
}
div[data-testid="stNumberInput"] input:focus {
    border-color: #007AFF !important;
    box-shadow: 0 0 0 3px rgba(0, 122, 255, 0.22), inset 0 1px 1px 0 rgba(255, 255, 255, 0.9) !important;
}
div[data-testid="stNumberInput"] button {
    background: transparent !important;
    border: none !important;
    color: #8E8E93 !important;
    transition: color 0.15s ease, transform 0.1s ease !important;
}
div[data-testid="stNumberInput"] button:hover {
    color: #007AFF !important;
    transform: scale(1.15) !important;
}
div[data-testid="stNumberInput"] button:active {
    transform: scale(0.95) !important;
}

/* ── Transparency Disclosure Pill ───────────────────────── */
.transparency-pill {
    display: flex;
    align-items: center;
    gap: 8px;
    background: linear-gradient(135deg, rgba(0, 122, 255, 0.08) 0%, rgba(0, 122, 255, 0.03) 100%);
    border: 1px solid rgba(0, 122, 255, 0.25);
    border-radius: 12px;
    padding: 0.45rem 0.85rem;
    font-size: 0.82rem;
    font-weight: 600;
    color: #007AFF;
    margin-bottom: 0.75rem;
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
}

/* ── Baseline Imputation Status Tag ─────────────────────── */
.imputation-tag {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    background: rgba(142, 142, 147, 0.12);
    border: 1px solid rgba(142, 142, 147, 0.25);
    border-radius: 6px;
    padding: 0.15rem 0.50rem;
    font-size: 0.72rem;
    font-weight: 600;
    color: #636366;
}

/* ── Reference Benchmark Metric Cards ───────────────────── */
.ref-card-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.42) 0%, rgba(255, 255, 255, 0.22) 100%);
    backdrop-filter: blur(28px) saturate(210%);
    -webkit-backdrop-filter: blur(28px) saturate(210%);
    border-radius: 14px;
    border: 1px solid rgba(255, 255, 255, 0.45);
    border-top: 1px solid rgba(255, 255, 255, 0.80);
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.04), inset 0 1px 1px rgba(255, 255, 255, 0.7);
    padding: 0.60rem 0.90rem;
    margin-bottom: 0.40rem;
    min-height: 54px;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.ref-card-row:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.06);
}
.ref-meta-left {
    display: flex;
    align-items: center;
    gap: 10px;
}
.ref-metric-name {
    font-size: 0.84rem;
    font-weight: 700;
    color: #1C1C1E;
    line-height: 1.2;
}
.ref-metric-unit {
    font-size: 0.72rem;
    color: #8E8E93;
    font-weight: 500;
}
.ref-badges-right {
    display: flex;
    gap: 6px;
    align-items: center;
}
.range-badge {
    display: inline-flex;
    align-items: center;
    padding: 0.20rem 0.55rem;
    border-radius: 8px;
    font-size: 0.70rem;
    font-weight: 700;
    letter-spacing: -0.01em;
}
.badge-normal {
    background: rgba(52, 199, 89, 0.14);
    color: #248A3D;
    border: 1px solid rgba(52, 199, 89, 0.35);
}
.badge-danger {
    background: rgba(255, 59, 48, 0.14);
    color: #D70015;
    border: 1px solid rgba(255, 59, 48, 0.35);
}

/* ── Primary Action Button ──────────────────────────────── */
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #007AFF 0%, #0056B3 100%) !important;
    color: #FFFFFF !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    border-radius: 14px !important;
    border: none !important;
    padding: 0.65rem 1.4rem !important;
    box-shadow: 0 4px 16px rgba(0, 122, 255, 0.35) !important;
    transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
}
div[data-testid="stButton"] > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(0, 122, 255, 0.45) !important;
    background: linear-gradient(135deg, #0A84FF 0%, #0062D2 100%) !important;
}
div[data-testid="stButton"] > button:active {
    transform: translateY(0px) !important;
}

/* ── PDF Download Button ────────────────────────────────── */
div[data-testid="stDownloadButton"] > button {
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.65) 0%, rgba(255, 255, 255, 0.40) 100%) !important;
    color: #007AFF !important;
    font-weight: 700 !important;
    font-size: 0.90rem !important;
    border-radius: 14px !important;
    border: 1.5px solid rgba(0, 122, 255, 0.4) !important;
    padding: 0.65rem 1.4rem !important;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.05) !important;
    transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
}
div[data-testid="stDownloadButton"] > button:hover {
    background: linear-gradient(135deg, #007AFF 0%, #0056B3 100%) !important;
    color: #FFFFFF !important;
    border-color: transparent !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(0, 122, 255, 0.35) !important;
}

/* ── Risk Cards ─────────────────────────────────────────── */
.risk-card {
    border-radius: 20px;
    padding: 1.35rem 1.6rem;
    box-shadow: 0 10px 32px 0 rgba(0, 0, 0, 0.06);
    margin-bottom: 0.9rem;
    backdrop-filter: blur(28px) saturate(210%);
    -webkit-backdrop-filter: blur(28px) saturate(210%);
}
.risk-card-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 0.5rem;
}
.risk-card-header h3 {
    margin: 0 !important;
    font-size: 1.25rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.015em;
}
.risk-low {
    background: linear-gradient(135deg, rgba(224, 249, 231, 0.70) 0%, rgba(255, 255, 255, 0.45) 100%);
    border: 1px solid rgba(52, 199, 89, 0.4);
    border-top: 1px solid rgba(52, 199, 89, 0.7);
}
.risk-low h3 { color: #248A3D !important; }

.risk-medium {
    background: linear-gradient(135deg, rgba(255, 243, 214, 0.70) 0%, rgba(255, 255, 255, 0.45) 100%);
    border: 1px solid rgba(255, 149, 0, 0.4);
    border-top: 1px solid rgba(255, 149, 0, 0.7);
}
.risk-medium h3 { color: #C93400 !important; }

.risk-high {
    background: linear-gradient(135deg, rgba(255, 229, 229, 0.70) 0%, rgba(255, 255, 255, 0.45) 100%);
    border: 1px solid rgba(255, 59, 48, 0.4);
    border-top: 1px solid rgba(255, 59, 48, 0.7);
}
.risk-high h3 { color: #D70015 !important; }

/* ── Metric Summary Tiles ───────────────────────────────── */
.metric-tile {
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.48) 0%, rgba(255, 255, 255, 0.28) 100%);
    backdrop-filter: blur(28px) saturate(210%);
    -webkit-backdrop-filter: blur(28px) saturate(210%);
    border-radius: 16px;
    border: 1px solid rgba(255, 255, 255, 0.50);
    border-top: 1px solid rgba(255, 255, 255, 0.85);
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.04), inset 0 1px 1px rgba(255, 255, 255, 0.8);
    padding: 0.95rem 1.15rem;
    text-align: center;
}
.metric-tile .value {
    font-size: 1.65rem;
    font-weight: 800;
    letter-spacing: -0.02em;
}
.metric-tile .label {
    font-size: 0.73rem;
    font-weight: 700;
    color: #8E8E93;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    margin-top: 2px;
}

/* ── Medical Disclaimer ─────────────────────────────────── */
.disclaimer {
    background: linear-gradient(135deg, rgba(242, 242, 247, 0.75) 0%, rgba(255, 255, 255, 0.55) 100%);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border-radius: 14px;
    border: 1px solid rgba(0, 0, 0, 0.06);
    border-left: 4px solid #007AFF;
    padding: 0.85rem 1.15rem;
    font-size: 0.78rem;
    line-height: 1.45;
    color: #636366;
    margin-top: 1.25rem;
}

/* ── Expander Glass Styling ─────────────────────────────── */
div[data-testid="stExpander"] {
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.40) 0%, rgba(255, 255, 255, 0.22) 100%) !important;
    backdrop-filter: blur(28px) saturate(210%) !important;
    -webkit-backdrop-filter: blur(28px) saturate(210%) !important;
    border-radius: 18px !important;
    border: 1px solid rgba(255, 255, 255, 0.45) !important;
    border-top: 1px solid rgba(255, 255, 255, 0.80) !important;
    box-shadow: 0 8px 24px 0 rgba(0, 0, 0, 0.04) !important;
    margin-top: 0.65rem !important;
    overflow: hidden !important;
}
div[data-testid="stExpander"] details summary {
    font-weight: 700 !important;
    color: #1C1C1E !important;
    padding: 0.65rem 0.95rem !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  CLINICAL PDF REPORT BUILDER (ReportLab)
# ─────────────────────────────────────────────
def generate_clinical_pdf(patient_data, prediction, prob_diabetic, prob_non_diabetic, flags, omitted_features=None, patient_id="PT-2026-8841"):
    """
    Generate a print-ready clinical screening report adhering strictly to
    standard hospital/clinical documentation standards.
    """
    if omitted_features is None:
        omitted_features = []

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=letter,
        rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36
    )
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=colors.HexColor('#1C1C1E'),
    )
    subtitle_style = ParagraphStyle(
        'DocSub',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#8E8E93'),
    )
    section_heading = ParagraphStyle(
        'SectionHead',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=10.5,
        leading=14,
        textColor=colors.HexColor('#007AFF'),
        spaceAfter=4,
    )
    cell_bold = ParagraphStyle(
        'CellBold',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor('#1C1C1E'),
    )
    cell_regular = ParagraphStyle(
        'CellReg',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor('#3A3A3C'),
    )
    
    elements = []
    
    # ── 1. Clinical Header ──
    elements.append(Paragraph("AI Diabetes Risk Assessment — Clinical Screening Report", title_style))
    elements.append(Paragraph("Rocheston RCAI Classroom Project &bull; Clinical Decision Support Telemetry", subtitle_style))
    elements.append(Spacer(1, 8))
    elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#007AFF"), spaceBefore=2, spaceAfter=8))
    
    # ── 2. Telemetry Metadata Block ──
    curr_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    model_desc = "ML Pipeline (Dedicated 4-Core Feature Architecture)" if len(omitted_features) == 4 else (
        "ML Pipeline (8-Features with Population Median Imputation)" if omitted_features else "ML Random Forest Pipeline (Full 8-Feature Vector)"
    )
    meta_data = [
        [
            Paragraph(f"<b>Patient Identifier:</b> {patient_id}", cell_regular),
            Paragraph(f"<b>Generated Timestamp:</b> {curr_time}", cell_regular),
        ],
        [
            Paragraph(f"<b>Diagnostic Model:</b> {model_desc}", cell_regular),
            Paragraph("<b>Target Domain:</b> Type 2 Diabetes Risk Screening", cell_regular),
        ]
    ]
    meta_table = Table(meta_data, colWidths=[270, 270])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F2F2F7')),
        ('PADDING', (0,0), (-1,-1), 5),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    elements.append(meta_table)
    elements.append(Spacer(1, 10))
    
    # ── 3. Executive Assessment Findings ──
    elements.append(Paragraph("1. ASSESSMENT FINDINGS & RISK STRATIFICATION", section_heading))
    risk_label = "ELEVATED RISK / LIKELY TO HAVE DIABETES" if prob_diabetic >= 0.65 else ("MODERATE RISK / BORDERLINE" if prob_diabetic >= 0.40 else "LOW RISK / UNLIKELY TO HAVE DIABETES")
    risk_bg = colors.HexColor('#FFE5E5') if prob_diabetic >= 0.65 else (colors.HexColor('#FFF3D6') if prob_diabetic >= 0.40 else colors.HexColor('#E0F9E7'))
    risk_fg = colors.HexColor('#D70015') if prob_diabetic >= 0.65 else (colors.HexColor('#C93400') if prob_diabetic >= 0.40 else colors.HexColor('#248A3D'))
    
    findings_data = [
        [
            Paragraph(f"<b>Predicted Status:</b> <font color='{risk_fg.hexval()}'><b>{risk_label}</b></font>", cell_bold),
            Paragraph(f"<b>Confidence Score:</b> <b>{prob_diabetic*100:.1f}% Elevated Risk</b> (Non-Diabetic: {prob_non_diabetic*100:.1f}%)", cell_bold)
        ]
    ]
    findings_table = Table(findings_data, colWidths=[270, 270])
    findings_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), risk_bg),
        ('PADDING', (0,0), (-1,-1), 7),
        ('BOX', (0,0), (-1,-1), 1, risk_fg),
    ]))
    elements.append(findings_table)

    # Transparency Note in PDF if parameters omitted
    if omitted_features:
        elements.append(Spacer(1, 5))
        omitted_text = (
            f"<i><b>Clinical Transparency Note:</b> Assessment calculated using 4 core metrics + standard population "
            f"baselines for omitted lab tests ({', '.join(omitted_features)}).</i>"
        )
        elements.append(Paragraph(omitted_text, ParagraphStyle('OmitNote', parent=styles['Normal'], fontSize=7.5, leading=10, textColor=colors.HexColor('#007AFF'))))

    elements.append(Spacer(1, 10))
    
    # ── 4. Patient Input Profile & Benchmark Comparison ──
    elements.append(Paragraph("2. SUBMITTED CLINICAL PARAMETERS VS. BASELINE BENCHMARKS", section_heading))
    param_rows = [
        [
            Paragraph("Clinical Parameter", cell_bold),
            Paragraph("Value (Submitted / Imputed)", cell_bold),
            Paragraph("Status", cell_bold),
            Paragraph("Standard Unit", cell_bold),
            Paragraph("Baseline Target Norm", cell_bold)
        ]
    ]
    ref_norms = {
        "Glucose": ("mg/dL", "70 – 99 mg/dL"),
        "BloodPressure": ("mm Hg", "< 80 mm Hg (diastolic)"),
        "BMI": ("kg/m²", "18.5 – 24.9 kg/m²"),
        "Age": ("years", "18 – 44 years"),
        "Insulin": ("µU/mL", "16 – 166 µU/mL"),
        "SkinThickness": ("mm", "10 – 30 mm"),
        "Pregnancies": ("count", "0 – 2"),
        "DiabetesPedigreeFunction": ("score", "< 0.50"),
    }
    for k in ALL_FEATURES:
        if k in patient_data:
            v = patient_data[k]
            unit, norm = ref_norms.get(k, ("", "—"))
            val_str = f"{v:.2f}" if isinstance(v, float) else f"{v}"
            is_imputed = k in omitted_features
            status_str = "<font color='#007AFF'><b>Imputed Baseline</b></font>" if is_imputed else "<font color='#248A3D'><b>Measured</b></font>"
            param_rows.append([
                Paragraph(FEATURE_META.get(k, {}).get("label", k), cell_regular),
                Paragraph(f"<b>{val_str}</b>", cell_bold),
                Paragraph(status_str, cell_regular),
                Paragraph(unit, cell_regular),
                Paragraph(norm, cell_regular),
            ])
    
    param_table = Table(param_rows, colWidths=[150, 110, 95, 75, 110])
    param_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#E5E5EA')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#D1D1D6')),
        ('PADDING', (0,0), (-1,-1), 4),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#F9F9FB')]),
    ]))
    elements.append(param_table)
    elements.append(Spacer(1, 10))
    
    # ── 5. Identified Risk Flags ──
    elements.append(Paragraph("3. CLINICAL RISK FACTOR FLAGS", section_heading))
    if flags:
        flag_data = [[Paragraph(f"&bull; {f}", cell_regular)] for f in flags]
    else:
        flag_data = [[Paragraph("All submitted clinical indicators reside within standard non-diabetic target parameters.", cell_regular)]]
    flag_table = Table(flag_data, colWidths=[540])
    flag_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#FAFAFA')),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#E5E5EA')),
        ('PADDING', (0,0), (-1,-1), 5),
    ]))
    elements.append(flag_table)
    elements.append(Spacer(1, 12))
    
    # ── 6. Regulatory Disclaimer ──
    disclaimer_text = (
        "<b>Regulatory & Medical Disclaimer:</b> This document is generated automatically by a statistical machine learning "
        "supervised classification pipeline for screening purposes and does not replace professional medical diagnosis. "
        "Always consult a licensed medical practitioner for comprehensive evaluation, official blood testing "
        "(e.g., HbA1c, OGTT), and tailored healthcare decisions."
    )
    disc_para = Paragraph(disclaimer_text, ParagraphStyle('Disc', parent=styles['Normal'], fontSize=7.5, leading=10, textColor=colors.HexColor('#636366')))
    disc_table = Table([[disc_para]], colWidths=[540])
    disc_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F2F2F7')),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#D1D1D6')),
        ('LINELEFT', (0,0), (0,-1), 3, colors.HexColor('#007AFF')),
        ('PADDING', (0,0), (-1,-1), 5.5),
    ]))
    elements.append(disc_table)
    
    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()

# ─────────────────────────────────────────────
#  DATA & MODEL LOADING (DUAL-ROUTING)
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

def _train_models(df):
    """Train both 8-feature and 4-feature pipelines."""
    clean = _clean_df(df)
    y = clean[TARGET]

    # 1. Train 8-feature pipeline
    X8 = clean[ALL_FEATURES]
    X8_train, X8_test, y_train, y_test = train_test_split(
        X8, y, test_size=0.20, random_state=42, stratify=y
    )
    pipe_8 = Pipeline([
        ("imputer",    SimpleImputer(strategy="mean")),
        ("scaler",     MinMaxScaler()),
        ("classifier", RandomForestClassifier(n_estimators=100, criterion="entropy", random_state=42)),
    ])
    pipe_8.fit(X8_train, y_train)

    with open(MODEL_PATH_8, "wb") as f:
        pickle.dump(pipe_8, f)

    # 2. Train dedicated 4-core feature pipeline
    X4 = clean[PRIMARY_FEATURES]
    X4_train, X4_test, _, _ = train_test_split(
        X4, y, test_size=0.20, random_state=42, stratify=y
    )
    pipe_4 = Pipeline([
        ("imputer",    SimpleImputer(strategy="mean")),
        ("scaler",     MinMaxScaler()),
        ("classifier", LogisticRegression(max_iter=500, random_state=42)),
    ])
    pipe_4.fit(X4_train, y_train)

    with open(MODEL_PATH_4, "wb") as f:
        pickle.dump(pipe_4, f)

    return pipe_8, pipe_4

@st.cache_resource
def load_models():
    """
    Load both 8-feature model (Diabetesmodel.pkl) and 4-feature model (Diabetesmodel_4.pkl).
    Trains automatically if missing.
    """
    df = load_dataset()
    pipe_8, pipe_4 = None, None

    if os.path.exists(MODEL_PATH_8):
        try:
            with open(MODEL_PATH_8, "rb") as f:
                pipe_8 = pickle.load(f)
        except Exception:
            pipe_8 = None

    if os.path.exists(MODEL_PATH_4):
        try:
            with open(MODEL_PATH_4, "rb") as f:
                pipe_4 = pickle.load(f)
        except Exception:
            pipe_4 = None

    if pipe_8 is None or pipe_4 is None:
        p8, p4 = _train_models(df)
        if pipe_8 is None: pipe_8 = p8
        if pipe_4 is None: pipe_4 = p4

    return pipe_8, pipe_4

@st.cache_data
def compute_benchmark_metrics(df):
    """Train and evaluate comparative models for Developer Analytics."""
    clean = _clean_df(df)
    X = clean[ALL_FEATURES]
    y = clean[TARGET]

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
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150, transparent=True)
    buf.seek(0)
    plt.close(fig)
    return buf

def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4.0, 3.2))
    fig.patch.set_facecolor('#F2F2F7')
    ax.set_facecolor('#F2F2F7')

    sns.heatmap(
        cm, annot=True, fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Non-Diabetic", "Diabetic"],
        yticklabels=["Non-Diabetic", "Diabetic"],
        ax=ax,
        annot_kws={"size": 13, "weight": "bold", "color": "#1C1C1E"},
    )
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=11, fontweight="bold", pad=8, color="#1C1C1E")
    ax.set_xlabel("Predicted Class", fontsize=9.5, fontweight="600", color="#636366")
    ax.set_ylabel("True Class", fontsize=9.5, fontweight="600", color="#636366")
    ax.tick_params(colors="#636366", labelsize=8.5)
    plt.tight_layout()
    return _fig_to_buf(fig)

def plot_feature_importance(pipeline, feature_names):
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    fig.patch.set_facecolor('#F2F2F7')
    ax.set_facecolor('#F2F2F7')

    clf = pipeline.named_steps.get("classifier", None)
    if hasattr(clf, "feature_importances_"):
        imps = clf.feature_importances_
        title = "Random Forest Feature Importance"
    elif hasattr(clf, "coef_"):
        imps = np.abs(clf.coef_[0])
        title = "Model Feature Weight |coef|"
    else:
        imps = np.ones(len(feature_names))
        title = "Feature Relative Influence"

    idx = np.argsort(imps)
    sorted_feats = [feature_names[i] for i in idx]
    sorted_imps  = imps[idx]

    bars = ax.barh(sorted_feats, sorted_imps, color="#007AFF", height=0.55, edgecolor="none")
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.005, bar.get_y() + bar.get_height()/2, f"{w:.3f}",
                va="center", ha="left", fontsize=7.5, color="#636366", fontweight="600")

    ax.set_title(title, fontsize=10.5, fontweight="bold", pad=8, color="#1C1C1E")
    ax.set_xlabel("Importance Weight", fontsize=8.5, fontweight="600", color="#636366")
    ax.tick_params(colors="#636366", labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#D1D1D6")
    ax.spines["bottom"].set_color("#D1D1D6")
    plt.tight_layout()
    return _fig_to_buf(fig)

def plot_risk_gauge(prob_diabetic):
    fig, ax = plt.subplots(figsize=(4.6, 2.3), subplot_kw={"projection": "polar"})
    fig.patch.set_facecolor('#F2F2F7')
    ax.set_facecolor('#F2F2F7')

    theta_full = np.linspace(np.pi, 0, 300)
    r_inner = 0.65
    r_outer = 1.0

    ax.fill_between(theta_full[:100], r_inner, r_outer, color="#34C759", alpha=0.85)
    ax.fill_between(theta_full[100:200], r_inner, r_outer, color="#FF9500", alpha=0.85)
    ax.fill_between(theta_full[200:], r_inner, r_outer, color="#FF3B30", alpha=0.85)

    needle_angle = np.pi - (prob_diabetic * np.pi)
    ax.annotate(
        "",
        xy=(needle_angle, 0.96),
        xytext=(0, 0),
        arrowprops=dict(arrowstyle="->", color="#1C1C1E", lw=3.0, mutation_scale=16),
    )
    ax.plot(0, 0, "o", color="#1C1C1E", markersize=8)

    ax.set_ylim(0, 1.05)
    ax.set_yticklabels([])
    ax.set_xticks([np.pi, np.pi/2, 0])
    ax.set_xticklabels(["Low", "", "High"], fontsize=9.5, fontweight="bold", color="#636366")
    ax.spines["polar"].set_visible(False)
    ax.grid(False)

    pct_str = f"{prob_diabetic*100:.1f}%"
    ax.text(
        0, -0.22, pct_str,
        ha="center", va="center",
        fontsize=22, fontweight="900", color="#1C1C1E",
    )
    ax.text(
        0, -0.42, "Diabetes Risk Score",
        ha="center", va="center",
        fontsize=8.5, fontweight="700", color="#8E8E93",
    )
    plt.tight_layout()
    return _fig_to_buf(fig)

# ─────────────────────────────────────────────
#  PRESET STATE HANDLER
# ─────────────────────────────────────────────
def apply_preset(preset_name):
    """Apply clinical preset values to all session states."""
    if preset_name in PRESETS and PRESETS[preset_name] is not None:
        values = PRESETS[preset_name]
        for f, v in values.items():
            if f in FEATURE_META:
                cast_v = FEATURE_META[f]["type"](v)
                st.session_state[f"slider_{f}"] = cast_v
                st.session_state[f"num_{f}"] = cast_v
                st.session_state[f"input_{f}"] = cast_v
        # When an explicit clinical preset is picked, uncheck skips
        for f in ADVANCED_FEATURES:
            st.session_state[f"skip_{f}"] = False
        st.session_state["toggle_advanced_biomarkers"] = True

# ─────────────────────────────────────────────
#  CLIENT-SIDE DUAL-INPUT SYNC ENGINE (JS)
# ─────────────────────────────────────────────
def inject_client_sync_engine():
    """
    Zero-Latency Dual-Input Synchronization & Stepper Buffer Engine.
    Runs in the browser DOM to provide:
      1. 0 ms perceptible latency on every slider drag tick.
      2. Stepper '+' / '-' decoupling with a fast local state buffer,
         preventing dropped clicks and UI freezes.
      3. Smooth capsule progress bar geometry under the translucent thumb.
    """
    components.html(
        """
        <script>
        (function() {
            function initDualInputSync() {
                const rootDoc = window.parent.document || document;
                const sliders = Array.from(rootDoc.querySelectorAll('div[data-testid="stSlider"]'));
                const numbers = Array.from(rootDoc.querySelectorAll('div[data-testid="stNumberInput"]'));

                if (sliders.length === 0 || numbers.length === 0) return;

                const nativeInputSetter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;

                sliders.forEach(sliderElem => {
                    if (sliderElem.dataset.syncBound === "true") return;

                    const labelElem = sliderElem.querySelector('label') || sliderElem.querySelector('[data-testid="stWidgetLabel"]');
                    const labelText = labelElem ? labelElem.innerText.trim() : '';

                    const numberElem = numbers.find(n => {
                        const nLabel = n.querySelector('label') || n.querySelector('[data-testid="stWidgetLabel"]');
                        return nLabel && nLabel.innerText.trim() === labelText;
                    });

                    if (!numberElem) return;

                    sliderElem.dataset.syncBound = "true";
                    numberElem.dataset.syncBound = "true";

                    const numInput = numberElem.querySelector('input');
                    const rangeInput = sliderElem.querySelector('input[type="range"]');
                    const thumb = sliderElem.querySelector('div[class*="efbyxod3"]') || sliderElem.querySelector('[role="slider"]');
                    const track = sliderElem.querySelector('div[class*="efbyxod5"]') || sliderElem.querySelector('div[data-baseweb="slider"] > div');
                    const incBtn = numberElem.querySelector('button[aria-label="Increment"]');
                    const decBtn = numberElem.querySelector('button[aria-label="Decrement"]');

                    if (!numInput || !rangeInput) return;

                    const min = parseFloat(rangeInput.min) || 0;
                    const max = parseFloat(rangeInput.max) || 100;
                    const step = parseFloat(rangeInput.step) || 1;
                    const isFloat = step < 1 || (step % 1 !== 0);

                    function formatVal(v) {
                        if (isFloat) {
                            return (Math.round(v * 100) / 100).toFixed(step < 0.1 ? 2 : 1);
                        }
                        return Math.round(v).toString();
                    }

                    function updatePillVisuals(val, source) {
                        const clamped = Math.max(min, Math.min(max, val));
                        const pct = ((clamped - min) / (max - min)) * 100;

                        // Capsule progress bar fill
                        if (track) {
                            track.style.setProperty('--slider-fill', pct + '%');
                        }
                        // Pill thumb position
                        if (thumb) {
                            thumb.style.left = pct + '%';
                        }

                        // Sync text box on slider movement
                        if (source === 'slider' && numInput) {
                            nativeInputSetter.call(numInput, formatVal(clamped));
                            numInput.dispatchEvent(new Event('input', { bubbles: true }));
                        }

                        // Sync range input on number box movement
                        if (source === 'number' && rangeInput) {
                            nativeInputSetter.call(rangeInput, clamped.toString());
                            rangeInput.dispatchEvent(new Event('input', { bubbles: true }));
                        }
                    }

                    // Initial coordinate sync
                    const initialVal = parseFloat(numInput.value || rangeInput.value);
                    updatePillVisuals(initialVal, 'init');

                    // 1. INSTANT CLIENT-SIDE SLIDER DRAG SYNCHRONIZATION (0 ms)
                    rangeInput.addEventListener('input', (e) => {
                        const v = parseFloat(e.target.value);
                        updatePillVisuals(v, 'slider');
                    });

                    let isDragging = false;
                    function onPointerMove(e) {
                        if (!track) return;
                        const rect = track.getBoundingClientRect();
                        const clientX = e.touches ? e.touches[0].clientX : e.clientX;
                        const pct = Math.max(0, Math.min(100, ((clientX - rect.left) / rect.width) * 100));
                        const raw = min + (pct / 100) * (max - min);
                        const stepped = isFloat ? Math.round(raw / step) * step : Math.round(raw);
                        updatePillVisuals(stepped, 'slider');
                    }

                    sliderElem.addEventListener('pointerdown', (e) => {
                        isDragging = true;
                        onPointerMove(e);
                    });

                    rootDoc.addEventListener('pointermove', (e) => {
                        if (!isDragging) return;
                        onPointerMove(e);
                    });

                    rootDoc.addEventListener('pointerup', () => {
                        if (!isDragging) return;
                        isDragging = false;
                        if (numInput) numInput.dispatchEvent(new Event('change', { bubbles: true }));
                        if (rangeInput) rangeInput.dispatchEvent(new Event('change', { bubbles: true }));
                    });

                    // 2. STEPPER RAPID-CLICK FREEZE BUG FIX (DECOUPLED LOCAL BUFFER)
                    let stepperBuffer = parseFloat(numInput.value) || initialVal;
                    let stepperTimer = null;

                    function onStepperClick(delta, e) {
                        e.preventDefault();
                        e.stopPropagation();
                        e.stopImmediatePropagation();

                        stepperBuffer = parseFloat(numInput.value) || stepperBuffer;
                        stepperBuffer = Math.max(min, Math.min(max, stepperBuffer + delta));
                        if (!isFloat) stepperBuffer = Math.round(stepperBuffer);
                        else stepperBuffer = Math.round(stepperBuffer / step) * step;

                        // Immediate 0 ms update to input box and slider progress bar
                        updatePillVisuals(stepperBuffer, 'number');

                        // Debounce commitment to prevent UI-blocking server/React reruns
                        clearTimeout(stepperTimer);
                        stepperTimer = setTimeout(() => {
                            nativeInputSetter.call(numInput, formatVal(stepperBuffer));
                            numInput.dispatchEvent(new Event('change', { bubbles: true }));
                            nativeInputSetter.call(rangeInput, stepperBuffer.toString());
                            rangeInput.dispatchEvent(new Event('change', { bubbles: true }));
                        }, 320);
                    }

                    if (incBtn) {
                        incBtn.addEventListener('click', (e) => onStepperClick(step, e), true);
                        incBtn.addEventListener('pointerdown', (e) => e.stopPropagation(), true);
                    }
                    if (decBtn) {
                        decBtn.addEventListener('click', (e) => onStepperClick(-step, e), true);
                        decBtn.addEventListener('pointerdown', (e) => e.stopPropagation(), true);
                    }

                    // 3. DIRECT TEXT BOX TYPING SYNCHRONIZATION
                    numInput.addEventListener('input', (e) => {
                        const val = parseFloat(e.target.value);
                        if (!isNaN(val)) {
                            stepperBuffer = val;
                            updatePillVisuals(val, 'number');
                        }
                    });
                    numInput.addEventListener('change', (e) => {
                        const val = parseFloat(e.target.value);
                        if (!isNaN(val)) {
                            const clamped = Math.max(min, Math.min(max, val));
                            stepperBuffer = clamped;
                            updatePillVisuals(clamped, 'number');
                        }
                    });
                });
            }

            // Execute immediately and monitor DOM additions
            initDualInputSync();
            setInterval(initDualInputSync, 350);
        })();
        </script>
        """,
        height=0,
        width=0,
    )

# ─────────────────────────────────────────────
#  MAIN APPLICATION
# ─────────────────────────────────────────────
def main():
    # Background Refraction Ambient Orbs
    st.markdown('<div class="ambient-orb-blue"></div><div class="ambient-orb-cyan"></div>', unsafe_allow_html=True)

    # Initialize predict run counter
    if "predict_counter" not in st.session_state:
        st.session_state.predict_counter = 0

    # Load data & machine learning models (Dual-Routing: 8-feature & 4-feature)
    df = load_dataset()
    pipeline_8, pipeline_4 = load_models()

    # Initialize session states for all features
    for f in ALL_FEATURES:
        meta = FEATURE_META[f]
        if f"slider_{f}" not in st.session_state:
            def_val = meta["type"](meta["default"])
            st.session_state[f"slider_{f}"] = def_val
            st.session_state[f"num_{f}"] = def_val
            st.session_state[f"input_{f}"] = def_val
        if f"skip_{f}" not in st.session_state:
            st.session_state[f"skip_{f}"] = False

    if "toggle_advanced_biomarkers" not in st.session_state:
        st.session_state["toggle_advanced_biomarkers"] = False

    # ═══════════════════════════════════════════
    #  DEVELOPER DRAWER (SIDEBAR)
    # ═══════════════════════════════════════════
    with st.sidebar:
        st.markdown(f'''
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 0.5rem;">
            {SVG_ICONS["activity"]}
            <h2 style="font-size: 1.25rem; font-weight: 800; margin: 0; color: #1C1C1E;">Developer Drawer</h2>
        </div>
        <p style="font-size: 0.80rem; color: #8E8E93; margin-bottom: 1rem;">
            Diagnostic Telemetry & Model Benchmarking
        </p>
        ''', unsafe_allow_html=True)

        benchmark_df, benchmark_preds = compute_benchmark_metrics(df)

        st.markdown("#### Model Performance Benchmarks")
        display_df = benchmark_df.copy()
        for col in ["Accuracy", "Precision", "Recall", "F1"]:
            display_df[col] = display_df[col].apply(lambda v: f"{v*100:.2f}%")
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("#### Diagnostic Confusion Matrix")
        selected_model = st.selectbox(
            "Evaluation Classifier",
            list(benchmark_preds.keys()),
            index=0,
            key="dev_model_select"
        )
        y_test, y_pred = benchmark_preds[selected_model]
        cm_buf = plot_confusion_matrix(y_test, y_pred, selected_model)
        st.image(cm_buf, use_container_width=True)

        st.markdown("---")
        st.markdown("#### Feature Importance Profile")
        fi_buf = plot_feature_importance(pipeline_8, ALL_FEATURES)
        st.image(fi_buf, use_container_width=True)

    # ═══════════════════════════════════════════
    #  HERO TELEMETRY BANNER
    # ═══════════════════════════════════════════
    st.markdown(f'''
    <div class="hero-banner">
        <div class="hero-badge">
            {SVG_ICONS["activity"]}
            Clinical Decision Support Telemetry
        </div>
        <h1>AI Diabetes Risk Assessment</h1>
        <p>
            Evidence-based clinical risk stratification powered by supervised machine learning.
            Adjust patient biomarkers to analyze individualized metabolic risk and export formal clinical documentation.
        </p>
    </div>
    ''', unsafe_allow_html=True)

    # ═══════════════════════════════════════════
    #  BALANCED DUAL-COLUMN DESKTOP ARCHITECTURE
    # ═══════════════════════════════════════════
    col_input_panel, col_ref_panel = st.columns([5.9, 4.1], gap="medium")

    input_values = {}
    omitted_list = []

    # ─────────────────────────────────────────────────────────────
    #  LEFT COLUMN (59%): PATIENT CLINICAL INPUT PROFILE
    # ─────────────────────────────────────────────────────────────
    with col_input_panel:
        # ── Clinical Preset Selector Row ──
        st.markdown('<div class="section-label">CLINICAL PRESET PROFILE</div>', unsafe_allow_html=True)
        selected_preset = st.selectbox(
            label="Clinical Preset",
            options=list(PRESETS.keys()),
            index=0,
            key="preset_selector",
            label_visibility="collapsed",
        )
        if selected_preset != st.session_state.get("current_preset", "Custom Profile"):
            apply_preset(selected_preset)
            st.session_state["current_preset"] = selected_preset
            st.rerun()

        # ── 1. Primary Clinical Metrics (Required) Glass Container ──
        st.markdown('<div class="section-label" style="margin-top: 0.6rem;">PRIMARY CLINICAL METRICS (REQUIRED)</div>', unsafe_allow_html=True)
        with st.container(border=True):
            for feat in PRIMARY_FEATURES:
                meta = FEATURE_META[feat]
                unit_str = f"({meta['unit']})" if meta["unit"] else ""
                
                # Header row: Vector Icon Badge + Title + Unit
                st.markdown(f'''
                <div class="metric-row-header">
                    <div class="metric-title-group">
                        {render_badge(meta["icon"], meta["badge"])}
                        <span class="metric-row-title">{meta["label"]}</span>
                    </div>
                    <span class="metric-row-unit">{unit_str}</span>
                </div>
                ''', unsafe_allow_html=True)

                # Control row: Sleek Slider (73%) + Synchronized Number Input (27%)
                c_slide, c_num = st.columns([3.4, 1.25])
                with c_slide:
                    st.slider(
                        label=meta["label"],
                        min_value=meta["type"](meta["min"]),
                        max_value=meta["type"](meta["max"]),
                        value=meta["type"](st.session_state[f"slider_{feat}"]),
                        step=meta["type"](meta["step"]) if meta["type"] == float else meta["step"],
                        key=f"slider_{feat}",
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
                        label_visibility="collapsed",
                    )
                input_values[feat] = st.session_state.get(f"num_{feat}", st.session_state[f"slider_{feat}"])

        # ── 2. Advanced Laboratory Biomarkers (Optional) Collapsible Card ──
        st.markdown('<div class="section-label" style="margin-top: 0.6rem;">ADVANCED LABORATORY BIOMARKERS (OPTIONAL)</div>', unsafe_allow_html=True)
        with st.expander("Advanced Clinical Metrics (Optional) — Click to include laboratory records", expanded=st.session_state.get("toggle_advanced_biomarkers", False)):
            st.markdown(f'''
            <div style="font-size: 0.82rem; color: #636366; margin-bottom: 0.65rem; line-height: 1.4;">
                Include only if you have recent laboratory bloodwork or medical records.
                Unmeasured parameters will automatically use standardized population medians.
            </div>
            ''', unsafe_allow_html=True)

            has_advanced = st.toggle(
                "I have specialized laboratory bloodwork results",
                value=st.session_state.get("toggle_advanced_biomarkers", False),
                key="toggle_advanced_biomarkers"
            )

            for feat in ADVANCED_FEATURES:
                meta = FEATURE_META[feat]
                unit_str = f"({meta['unit']})" if meta["unit"] else ""
                median_val = POPULATION_MEDIANS[feat]

                if not has_advanced:
                    # Biomarker omitted by default
                    omitted_list.append(feat)
                    st.markdown(f'''
                    <div class="metric-row-header" style="margin-top: 6px; opacity: 0.85;">
                        <div class="metric-title-group">
                            {render_badge(meta["icon"], meta["badge"])}
                            <span class="metric-row-title">{meta["label"]}</span>
                        </div>
                        <span class="imputation-tag">Population Baseline: {median_val} {meta["unit"]} (Imputed)</span>
                    </div>
                    ''', unsafe_allow_html=True)
                    input_values[feat] = median_val
                else:
                    # User toggled advanced section on: offer individual skip option
                    skip_this = st.checkbox(
                        meta.get("skip_label", f"Skip / Not tested for {meta['label']}"),
                        value=st.session_state.get(f"skip_{feat}", False),
                        key=f"skip_{feat}"
                    )
                    if skip_this:
                        omitted_list.append(feat)
                        st.markdown(f'''
                        <div class="metric-row-header" style="margin-top: 4px; opacity: 0.75;">
                            <div class="metric-title-group">
                                {render_badge(meta["icon"], meta["badge"])}
                                <span class="metric-row-title">{meta["label"]}</span>
                            </div>
                            <span class="imputation-tag">Baseline: {median_val} {meta["unit"]} (Imputed)</span>
                        </div>
                        ''', unsafe_allow_html=True)
                        input_values[feat] = median_val
                    else:
                        st.markdown(f'''
                        <div class="metric-row-header" style="margin-top: 6px;">
                            <div class="metric-title-group">
                                {render_badge(meta["icon"], meta["badge"])}
                                <span class="metric-row-title">{meta["label"]}</span>
                            </div>
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
                                label_visibility="collapsed",
                            )
                        input_values[feat] = st.session_state.get(f"num_{feat}", st.session_state[f"slider_{feat}"])

        # Action Button anchored at the base of the input card
        st.markdown('<div style="margin-top: 10px;"></div>', unsafe_allow_html=True)
        predict_btn = st.button("Run Risk Assessment", type="primary", use_container_width=True)

    # ─────────────────────────────────────────────────────────────
    #  RIGHT COLUMN (41%): CLINICAL REFERENCE BENCHMARKS CONTAINER
    # ─────────────────────────────────────────────────────────────
    with col_ref_panel:
        st.markdown('<div class="section-label">CLINICAL REFERENCE BENCHMARKS</div>', unsafe_allow_html=True)
        
        # Reference Row 1: Glucose (Aligned parallel with Glucose Slider)
        st.markdown(f'''
        <div class="ref-card-row">
            <div class="ref-meta-left">
                {render_badge("droplet", "icon-badge-rose")}
                <div>
                    <div class="ref-metric-name">Fasting Glucose</div>
                    <div class="ref-metric-unit">Normal Target: 70–99 mg/dL</div>
                </div>
            </div>
            <div class="ref-badges-right">
                <span class="range-badge badge-normal">Normal: 70–99</span>
                <span class="range-badge badge-danger">Risk: ≥ 126 mg/dL</span>
            </div>
        </div>
        ''', unsafe_allow_html=True)

        # Reference Row 2: Blood Pressure (Aligned parallel with BP Slider)
        st.markdown(f'''
        <div class="ref-card-row">
            <div class="ref-meta-left">
                {render_badge("heart_pulse", "icon-badge-crimson")}
                <div>
                    <div class="ref-metric-name">Diastolic Blood Pressure</div>
                    <div class="ref-metric-unit">Normal Target: < 80 mm Hg</div>
                </div>
            </div>
            <div class="ref-badges-right">
                <span class="range-badge badge-normal">Normal: < 80</span>
                <span class="range-badge badge-danger">Risk: ≥ 90 mm Hg</span>
            </div>
        </div>
        ''', unsafe_allow_html=True)

        # Reference Row 3: BMI (Aligned parallel with BMI Slider)
        st.markdown(f'''
        <div class="ref-card-row">
            <div class="ref-meta-left">
                {render_badge("scale", "icon-badge-blue")}
                <div>
                    <div class="ref-metric-name">Body Mass Index (BMI)</div>
                    <div class="ref-metric-unit">Healthy Target: 18.5–24.9 kg/m²</div>
                </div>
            </div>
            <div class="ref-badges-right">
                <span class="range-badge badge-normal">Healthy: 18.5–24.9</span>
                <span class="range-badge badge-danger">Obesity: ≥ 30.0</span>
            </div>
        </div>
        ''', unsafe_allow_html=True)

        # Reference Row 4: Age (Aligned parallel with Age Slider)
        st.markdown(f'''
        <div class="ref-card-row">
            <div class="ref-meta-left">
                {render_badge("calendar", "icon-badge-amber")}
                <div>
                    <div class="ref-metric-name">Age & Metabolic Baseline</div>
                    <div class="ref-metric-unit">Standard Baseline: 18–44 yrs</div>
                </div>
            </div>
            <div class="ref-badges-right">
                <span class="range-badge badge-normal">Baseline: < 45</span>
                <span class="range-badge badge-danger">Elevated: ≥ 45 yrs</span>
            </div>
        </div>
        ''', unsafe_allow_html=True)

        # Secondary Reference Note for Serum Insulin
        st.markdown(f'''
        <div class="ref-card-row" style="opacity: 0.95;">
            <div class="ref-meta-left">
                {render_badge("syringe", "icon-badge-teal")}
                <div>
                    <div class="ref-metric-name">Serum Insulin (2-Hr)</div>
                    <div class="ref-metric-unit">Reference Target: 16–166 µU/mL</div>
                </div>
            </div>
            <div class="ref-badges-right">
                <span class="range-badge badge-normal">Normal: 16–166</span>
                <span class="range-badge badge-danger">Elevated: > 166</span>
            </div>
        </div>
        ''', unsafe_allow_html=True)

    # ═══════════════════════════════════════════
    #  INJECT CLIENT-SIDE SYNC ENGINE
    # ═══════════════════════════════════════════
    inject_client_sync_engine()

    # ═══════════════════════════════════════════
    #  ASSESSMENT RESULTS & PDF REPORT EXPORT
    # ═══════════════════════════════════════════
    if predict_btn:
        st.session_state.predict_counter += 1
        current_run_id = st.session_state.predict_counter

        # Target DOM anchor for scrolling
        st.markdown('<div id="assessment-result-anchor" style="padding-top: 1rem;"></div>', unsafe_allow_html=True)

        # Smooth auto-scroll hook
        components.html(
            f"""
            <script>
                (function() {{
                    const runId = {current_run_id};
                    setTimeout(function() {{
                        const doc = window.parent.document;
                        const el = doc.getElementById('assessment-result-anchor');
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

        # Determine features that were omitted/imputed
        omitted_in_assessment = [f for f in ADVANCED_FEATURES if f in omitted_list]

        # ── DUAL-MODEL ROUTING & IMPUTATION INFERENCE ──
        if len(omitted_in_assessment) == len(ADVANCED_FEATURES) and pipeline_4 is not None:
            # Approach B: Dedicated 4-feature model routing
            core_row = {feat: input_values[feat] for feat in PRIMARY_FEATURES}
            patient_df = pd.DataFrame([core_row], columns=PRIMARY_FEATURES)
            prediction = pipeline_4.predict(patient_df)[0]
            if hasattr(pipeline_4, "predict_proba"):
                probs = pipeline_4.predict_proba(patient_df)[0]
            else:
                probs = np.array([1 - prediction, prediction], dtype=float)
        else:
            # Approach A: 8-feature model with median imputation for omitted parameters
            row = {feat: input_values.get(feat, FEATURE_META[feat]["default"]) for feat in ALL_FEATURES}
            for zf in ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]:
                if zf in row and row[zf] == 0:
                    row[zf] = np.nan
            patient_df = pd.DataFrame([row], columns=ALL_FEATURES)
            prediction = pipeline_8.predict(patient_df)[0]
            if hasattr(pipeline_8, "predict_proba"):
                probs = pipeline_8.predict_proba(patient_df)[0]
            else:
                probs = np.array([1 - prediction, prediction], dtype=float)

        prob_diabetic     = float(probs[1])
        prob_non_diabetic = float(probs[0])

        st.markdown('<div class="section-label" style="margin-top: 1.4rem;">RISK STRATIFICATION & FINDINGS</div>', unsafe_allow_html=True)
        st.markdown("### Assessment Findings")

        # ── Transparency Status Pill (Client Disclosure) ──
        if omitted_in_assessment:
            st.markdown(f'''
            <div class="transparency-pill">
                {SVG_ICONS["info"]}
                <span>Assessment calculated using 4 core metrics + standard population baselines for omitted lab tests.</span>
            </div>
            ''', unsafe_allow_html=True)

        res_left, res_right = st.columns([3.2, 2.0], gap="large")

        # Compile clinical risk factor flags
        flags = []
        g = input_values.get("Glucose", 0)
        if g >= 126:
            flags.append("Fasting Glucose ≥ 126 mg/dL — Diagnostic of diabetes threshold")
        elif g >= 100:
            flags.append("Fasting Glucose 100–125 mg/dL — Impaired fasting glucose (Prediabetes)")

        bp = input_values.get("BloodPressure", 0)
        if bp >= 90:
            flags.append("Blood Pressure ≥ 90 mm Hg — Stage 2 Hypertension range")
        elif bp >= 80:
            flags.append("Blood Pressure 80–89 mm Hg — Prehypertension / Elevated")

        bmi_val = input_values.get("BMI", 0)
        if bmi_val >= 30:
            flags.append("BMI ≥ 30 kg/m² — Clinical Obesity (high insulin resistance risk)")
        elif bmi_val >= 25:
            flags.append("BMI 25–29.9 kg/m² — Overweight category")

        age_val = input_values.get("Age", 0)
        if age_val >= 45:
            flags.append("Age ≥ 45 years — Heightened age-related metabolic risk")

        ins = input_values.get("Insulin", 0)
        if "Insulin" not in omitted_in_assessment and ins > 166:
            flags.append("Insulin > 166 µU/mL — Elevated fasting/post-load hyperinsulinemia")

        with res_left:
            if prob_diabetic >= 0.65:
                card_class = "risk-high"
                icon_markup = SVG_ICONS["alert_triangle"]
                label = "Elevated Risk — Likely to have Diabetes"
            elif prob_diabetic >= 0.40:
                card_class = "risk-medium"
                icon_markup = SVG_ICONS["alert_circle"]
                label = "Moderate Risk — Borderline Screening"
            else:
                card_class = "risk-low"
                icon_markup = SVG_ICONS["shield_check"]
                label = "Low Risk — Unlikely to have Diabetes"

            st.markdown(f'''
            <div class="risk-card {card_class}">
                <div class="risk-card-header">
                    {icon_markup}
                    <h3>{label}</h3>
                </div>
                <p style="font-size: 1.05rem; margin-bottom: 0.45rem; color: #1C1C1E;">
                    The statistical machine learning model estimates a <strong>{prob_diabetic*100:.1f}%</strong>
                    probability of elevated diabetes risk based on the submitted clinical biomarker profile.
                </p>
                <p style="font-size: 0.90rem; color: #636366; margin: 0;">
                    Supervised Classification: <strong>{'Diabetic (Class 1)' if prediction == 1 else 'Non-Diabetic (Class 0)'}</strong>
                </p>
            </div>
            ''', unsafe_allow_html=True)

            st.markdown('<div class="section-label" style="margin-top: 1.1rem;">IDENTIFIED CLINICAL RISK FACTORS</div>', unsafe_allow_html=True)
            if flags:
                for f in flags:
                    st.markdown(f"- {f}")
            else:
                st.markdown(f'''
                <div style="display: flex; align-items: center; gap: 8px; color: #248A3D; font-weight: 600; padding: 6px 0;">
                    {SVG_ICONS["check"]}
                    <span>All submitted biomarker metrics fall within recommended non-diabetic target ranges.</span>
                </div>
                ''', unsafe_allow_html=True)

            # ── 4. PDF CLINICAL REPORT GENERATION & DOWNLOAD BUTTON ──
            patient_id_code = f"PT-{datetime.now().strftime('%Y%m%d')}-8841"
            pdf_bytes = generate_clinical_pdf(
                patient_data=input_values,
                prediction=prediction,
                prob_diabetic=prob_diabetic,
                prob_non_diabetic=prob_non_diabetic,
                flags=flags,
                omitted_features=omitted_in_assessment,
                patient_id=patient_id_code,
            )

            st.markdown('<div style="margin-top: 1.2rem;"></div>', unsafe_allow_html=True)
            st.download_button(
                label="Download / Print Clinical Report (PDF)",
                data=pdf_bytes,
                file_name=f"Clinical_Diabetes_Assessment_{patient_id_code}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

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
                    <div class="label">Diabetic Risk</div>
                </div>
                """, unsafe_allow_html=True)

        # Regulatory Disclaimer
        st.markdown(f'''
        <div class="disclaimer">
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 4px;">
                {SVG_ICONS["info"]}
                <strong>Regulatory & Medical Disclaimer</strong>
            </div>
            This clinical decision-support tool is an AI-assisted screening model developed for academic purposes.
            It does not replace professional medical diagnosis. Always consult a licensed healthcare provider
            for clinical evaluation, laboratory diagnostic testing (HbA1c, OGTT), and individualized care.
        </div>
        ''', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
