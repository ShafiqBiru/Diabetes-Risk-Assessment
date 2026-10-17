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
import base64

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
from reportlab.pdfgen import canvas
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
    "SystolicBP": {
        "label": "Systolic Blood Pressure (SBP)",
        "min": 70, "max": 200, "default": 120, "step": 1,
        "unit": "mm Hg", "type": int, "primary": True,
        "icon": "heart_pulse", "badge": "icon-badge-crimson",
        "norm_desc": "Normal Baseline: < 120 mm Hg",
    },
    "BloodPressure": {
        "label": "Diastolic Blood Pressure (DBP)",
        "min": 40, "max": 130, "default": 70, "step": 1,
        "unit": "mm Hg", "type": int, "primary": True,
        "icon": "heart_pulse", "badge": "icon-badge-crimson",
        "norm_desc": "Normal Baseline: < 80 mm Hg",
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
        "Pregnancies": 1, "Glucose": 88, "SystolicBP": 115, "BloodPressure": 68, "SkinThickness": 22,
        "Insulin": 45, "BMI": 22.5, "DiabetesPedigreeFunction": 0.25, "Age": 28,
    },
    "Borderline Risk Profile": {
        "Pregnancies": 3, "Glucose": 118, "SystolicBP": 128, "BloodPressure": 78, "SkinThickness": 28,
        "Insulin": 130, "BMI": 27.8, "DiabetesPedigreeFunction": 0.45, "Age": 42,
    },
    "Elevated Risk Profile": {
        "Pregnancies": 6, "Glucose": 168, "SystolicBP": 142, "BloodPressure": 88, "SkinThickness": 38,
        "Insulin": 240, "BMI": 36.2, "DiabetesPedigreeFunction": 0.88, "Age": 54,
    },
    "Clinical Obesity & Hypertension Profile": {
        "Pregnancies": 2, "Glucose": 142, "SystolicBP": 155, "BloodPressure": 96, "SkinThickness": 42,
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
    "salad": '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#34C759" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M7 21h10a5 5 0 0 0 5-5c0-2-1-3-2-4-1-1-1.5-2.5-1.5-4a7.5 7.5 0 0 0-15 0c0 1.5-.5 3-1.5 4-1 1-2 2-2 4a5 5 0 0 0 5 5z"></path><path d="M12 2v6"></path><path d="m9 5 3 3 3-3"></path></svg>',
    "flame": '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#FF9500" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M8.5 14.5A2.5 2.5 0 0 0 11 12c0-1.38-.5-2-1-3-1.072-2.143-.224-4.054 2-6 .5 2.5 2 4.9 4 6.5 2 1.6 3 3.5 3 5.5a7 7 0 1 1-14 0c0-1.153.433-2.294 1-3a2.5 2.5 0 0 0 2.5 2.5z"></path></svg>',
    "stethoscope": '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#007AFF" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4.8 2.3A.3.3 0 1 0 5 2H4a2 2 0 0 0-2 2v5a6 6 0 0 0 6 6v0a6 6 0 0 0 6-6V4a2 2 0 0 0-2-2h-1a.2.2 0 1 0 .3.3"></path><path d="M8 15v1a6 6 0 0 0 6 6v0a6 6 0 0 0 6-6v-4"></path><circle cx="20" cy="10" r="2"></circle></svg>',
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
    background: linear-gradient(to right, #007AFF 0%, #007AFF var(--slider-fill, var(--progress, 0%)), #E5E5EA var(--slider-fill, var(--progress, 0%)), #E5E5EA 100%) !important;
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
    width: var(--slider-fill, var(--progress, 0%)) !important;
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
.badge-amber {
    background: rgba(255, 149, 0, 0.14);
    color: #C93400;
    border: 1px solid rgba(255, 149, 0, 0.35);
}
.badge-danger {
    background: rgba(255, 59, 48, 0.14);
    color: #D70015;
    border: 1px solid rgba(255, 59, 48, 0.35);
}

/* ── Action Buttons (Primary & Secondary Apple Pills) ───── */
div[data-testid="stButton"] > button[kind="primary"],
button[data-testid="baseButton-primary"] {
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
div[data-testid="stButton"] > button[kind="primary"]:hover,
button[data-testid="baseButton-primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(0, 122, 255, 0.45) !important;
    background: linear-gradient(135deg, #0A84FF 0%, #0062D2 100%) !important;
}
div[data-testid="stButton"] > button[kind="secondary"],
button[data-testid="baseButton-secondary"] {
    background: rgba(255, 255, 255, 0.65) !important;
    backdrop-filter: blur(16px) saturate(180%) !important;
    -webkit-backdrop-filter: blur(16px) saturate(180%) !important;
    color: #1C1C1E !important;
    font-weight: 600 !important;
    font-size: 0.77rem !important;
    border-radius: 10px !important;
    border: 1px solid rgba(255, 255, 255, 0.75) !important;
    padding: 0.32rem 0.5rem !important;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.04) !important;
    white-space: nowrap !important;
    transition: all 0.18s ease !important;
}
div[data-testid="stButton"] > button[kind="secondary"]:hover,
button[data-testid="baseButton-secondary"]:hover {
    background: #FFFFFF !important;
    color: #007AFF !important;
    border-color: rgba(0, 122, 255, 0.45) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(0, 122, 255, 0.15) !important;
}

/* ── PDF Download Button (Centered Apple Pill) ───────────── */
div[data-testid="stDownloadButton"] {
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
    margin: 24px auto !important;
    width: 100% !important;
}
div[data-testid="stDownloadButton"] > button {
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    width: auto !important;
    max-width: fit-content !important;
    padding: 12px 28px !important;
    border-radius: 9999px !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    letter-spacing: -0.01em !important;
    border: 1px solid rgba(0, 122, 255, 0.35) !important;
    background: rgba(0, 122, 255, 0.08) !important;
    color: #007AFF !important;
    backdrop-filter: blur(20px) saturate(180%) !important;
    -webkit-backdrop-filter: blur(20px) saturate(180%) !important;
    box-shadow: 0 4px 14px rgba(0, 122, 255, 0.12) !important;
    transition: all 0.22s cubic-bezier(0.16, 1, 0.3, 1) !important;
    cursor: pointer !important;
}
div[data-testid="stDownloadButton"] > button:hover {
    background: #007AFF !important;
    color: #FFFFFF !important;
    border-color: #007AFF !important;
    transform: translateY(-2px) scale(1.02) !important;
    box-shadow: 0 8px 24px rgba(0, 122, 255, 0.35) !important;
}
div[data-testid="stDownloadButton"] > button:active {
    transform: translateY(0) scale(0.99) !important;
    box-shadow: 0 2px 8px rgba(0, 122, 255, 0.25) !important;
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

/* ── Apple Health Recommendations Liquid Glass Container ── */
.liquid-glass-card {
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.45) 0%, rgba(255, 255, 255, 0.28) 100%) !important;
    backdrop-filter: blur(28px) saturate(210%) !important;
    -webkit-backdrop-filter: blur(28px) saturate(210%) !important;
    border-radius: 20px !important;
    border: 1px solid rgba(255, 255, 255, 0.50) !important;
    border-top: 1px solid rgba(255, 255, 255, 0.85) !important;
    box-shadow: 0 10px 32px 0 rgba(0, 0, 0, 0.05), inset 0 1px 1px 0 rgba(255, 255, 255, 0.8) !important;
    padding: 1.35rem 1.6rem !important;
    margin-top: 1.25rem !important;
    margin-bottom: 1.0rem !important;
}
.recommendation-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 12px;
    margin-top: 0.9rem;
}
@media (max-width: 900px) {
    .recommendation-grid {
        grid-template-columns: 1fr;
    }
}
.recommendation-card {
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.52) 0%, rgba(255, 255, 255, 0.26) 100%);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border-radius: 16px;
    border: 1px solid rgba(255, 255, 255, 0.55);
    border-top: 1px solid rgba(255, 255, 255, 0.85);
    padding: 1.05rem 1.2rem;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.03), inset 0 1px 1px rgba(255, 255, 255, 0.7);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.recommendation-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.06);
}
.rec-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 0.55rem;
}
.rec-cat-title {
    font-size: 0.88rem;
    font-weight: 700;
    color: #1C1C1E;
}
.rec-bullet-list {
    margin: 0;
    padding: 0;
    list-style: none;
}
.rec-bullet-item {
    display: flex;
    align-items: flex-start;
    gap: 8px;
    font-size: 0.81rem;
    color: #1C1C1E;
    line-height: 1.42;
    margin-bottom: 0.40rem;
}
.rec-bullet-item:last-child {
    margin-bottom: 0;
}
.rec-check-icon {
    flex-shrink: 0;
    margin-top: 2px;
}
.bp-telemetry-note {
    font-size: 0.74rem;
    color: #8E8E93;
    margin-top: -2px;
    margin-bottom: 8px;
    padding-left: 2px;
    line-height: 1.35;
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

/* ── Expander Glass Styling (Borderless Container) ───────── */
div[data-testid="stExpander"],
div[data-testid="stExpander"] details {
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.38) 0%, rgba(255, 255, 255, 0.22) 100%) !important;
    backdrop-filter: blur(28px) saturate(210%) !important;
    -webkit-backdrop-filter: blur(28px) saturate(210%) !important;
    border-radius: 18px !important;
    border: none !important;
    box-shadow: 0 8px 24px 0 rgba(0, 0, 0, 0.04) !important;
    margin-top: 0.65rem !important;
    overflow: hidden !important;
}
div[data-testid="stExpander"] summary,
div[data-testid="stExpander"] details summary {
    border: none !important;
}
div[data-testid="stExpander"] details summary {
    font-weight: 700 !important;
    color: #1C1C1E !important;
    padding: 0.65rem 0.95rem !important;
}

/* ── Hide Broken Material Text Ligatures (Enforce Apple SVG Icons) ── */
span[data-testid="stIconMaterial"],
[data-testid="stIconMaterial"] {
    display: none !important;
}

/* ── Apple SF-Style Expander Chevron & Smooth Rotation ───── */
.apple-expander-chevron {
    width: 18px;
    height: 18px;
    stroke: currentColor;
    stroke-width: 2;
    fill: none;
    stroke-linecap: round;
    stroke-linejoin: round;
    color: #007AFF;
    transform: rotate(0deg);
    transition: transform 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    display: inline-block;
    vertical-align: middle;
    margin-right: 8px;
    flex-shrink: 0;
}
div[data-testid="stExpander"] details[open] .apple-expander-chevron {
    transform: rotate(180deg) !important;
}

/* ── Developer Sidebar Toggle (Compact Apple Pill) ── */
button[data-testid="stExpandSidebarButton"] {
    background: rgba(255, 255, 255, 0.75) !important;
    backdrop-filter: blur(24px) saturate(200%) !important;
    -webkit-backdrop-filter: blur(24px) saturate(200%) !important;
    border: 1px solid rgba(255, 255, 255, 0.80) !important;
    border-radius: 9999px !important;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.06), inset 0 1px 1px rgba(255, 255, 255, 0.8) !important;
    color: #1C1C1E !important;
    width: auto !important;
    max-width: fit-content !important;
    height: auto !important;
    min-height: unset !important;
    padding: 4px 11px !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 4px !important;
    cursor: pointer !important;
    transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
    margin-top: 4px !important;
    margin-left: 4px !important;
}
button[data-testid="stExpandSidebarButton"]:hover {
    background: rgba(255, 255, 255, 0.92) !important;
    box-shadow: 0 6px 18px rgba(0, 122, 255, 0.18) !important;
    transform: translateY(-1px) !important;
}
button[data-testid="stExpandSidebarButton"] > svg:not(.apple-sidebar-toggle-icon) {
    display: none !important;
}
.apple-dev-toggle-label {
    font-size: 0.78rem !important;
    font-weight: 700 !important;
    color: #1C1C1E !important;
    letter-spacing: -0.01em !important;
    line-height: 1 !important;
    margin: 0 !important;
    padding: 0 !important;
}
.apple-sidebar-toggle-icon {
    width: 15px !important;
    height: 15px !important;
    stroke: currentColor;
    stroke-width: 2.2;
    fill: none;
    stroke-linecap: round;
    stroke-linejoin: round;
    color: rgba(28, 28, 30, 0.75);
    display: inline-block;
    flex-shrink: 0;
    transition: transform 0.2s ease;
    margin: 0 !important;
}
div[data-testid="stSidebarCollapseButton"] {
    visibility: visible !important;
    opacity: 1 !important;
}
div[data-testid="stSidebarCollapseButton"] button {
    visibility: visible !important;
    opacity: 1 !important;
    background: rgba(255, 255, 255, 0.65) !important;
    backdrop-filter: blur(20px) saturate(180%) !important;
    -webkit-backdrop-filter: blur(20px) saturate(180%) !important;
    border: 1px solid rgba(255, 255, 255, 0.75) !important;
    border-radius: 12px !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04) !important;
    color: #1C1C1E !important;
    padding: 5px 8px !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
}
div[data-testid="stSidebarCollapseButton"] button:hover {
    background: rgba(255, 255, 255, 0.90) !important;
    transform: translateX(-1px) !important;
}

/* ── Apple Segmented Control (stRadio Horizontal Pill) ── */
div[data-testid="stRadio"] > div {
    background: rgba(118, 118, 128, 0.12) !important;
    backdrop-filter: blur(20px) saturate(180%) !important;
    -webkit-backdrop-filter: blur(20px) saturate(180%) !important;
    border-radius: 12px !important;
    padding: 3px !important;
    display: inline-flex !important;
    flex-direction: row !important;
    gap: 3px !important;
    border: 1px solid rgba(255, 255, 255, 0.50) !important;
    box-shadow: inset 0 1px 2px rgba(0, 0, 0, 0.04) !important;
    width: 100% !important;
    box-sizing: border-box !important;
}
div[data-testid="stRadio"] label {
    flex: 1 1 0 !important;
    margin: 0 !important;
    padding: 6px 12px !important;
    border-radius: 9px !important;
    font-size: 0.81rem !important;
    font-weight: 600 !important;
    color: #3A3A3C !important;
    cursor: pointer !important;
    transition: all 0.18s cubic-bezier(0.16, 1, 0.3, 1) !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    text-align: center !important;
}
div[data-testid="stRadio"] label > div:first-child {
    display: none !important;
}
div[data-testid="stRadio"] label:has(input:checked),
div[data-testid="stRadio"] label[data-checked="true"] {
    background: #FFFFFF !important;
    color: #007AFF !important;
    font-weight: 700 !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.12), 0 0.5px 1.5px rgba(0, 0, 0, 0.08) !important;
}

/* ── Apple Toast Notification ───────────────────────────── */
.apple-toast {
    position: fixed;
    bottom: 32px;
    left: 50%;
    transform: translateX(-50%) translateY(24px);
    background: rgba(28, 28, 30, 0.90);
    backdrop-filter: blur(28px) saturate(200%);
    -webkit-backdrop-filter: blur(28px) saturate(200%);
    color: #FFFFFF;
    padding: 12px 24px;
    border-radius: 9999px;
    font-size: 0.88rem;
    font-weight: 600;
    box-shadow: 0 12px 36px rgba(0, 0, 0, 0.28), inset 0 1px 1px rgba(255, 255, 255, 0.25);
    display: flex;
    align-items: center;
    gap: 10px;
    z-index: 9999999;
    opacity: 0;
    pointer-events: none;
    transition: opacity 0.3s cubic-bezier(0.16, 1, 0.3, 1), transform 0.3s cubic-bezier(0.16, 1, 0.3, 1);
}
.apple-toast.show {
    opacity: 1;
    transform: translateX(-50%) translateY(0);
}

</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  CLINICAL PDF REPORT BUILDER (ReportLab)
# ─────────────────────────────────────────────
class NumberedCanvas(canvas.Canvas):
    """
    Two-pass canvas renderer that dynamically inserts official clinical page
    numbers and document metadata headers/footers across all generated pages.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#8E8E93"))
        self.drawString(36, 18, "Confidential Clinical Telemetry Report • ROC-AI-2026")
        self.drawRightString(612 - 36, 18, f"Page {self._pageNumber} of {page_count}")
        self.restoreState()


def generate_lifestyle_recommendations(prob_diabetic, input_values, flags, is_detailed_bp=True):
    """
    Dynamically generates personalized lifestyle, nutrition, physical activity,
    weight & vitals management, and medical follow-up recommendations based on:
    1. Overall risk tier (Low Risk < 0.40, Moderate 0.40–0.64, Elevated/High >= 0.65)
    2. Specific biomarker triggers (Glucose > 100, BMI >= 25, SBP >= 130 / DBP >= 80, Age >= 45)
    """
    g = input_values.get("Glucose", 100)
    sbp = input_values.get("SystolicBP", 120) if is_detailed_bp else None
    dbp = input_values.get("BloodPressure", 70)
    bmi = input_values.get("BMI", 22.0)
    age = input_values.get("Age", 30)

    is_high = prob_diabetic >= 0.65
    is_mod = 0.40 <= prob_diabetic < 0.65
    is_low = prob_diabetic < 0.40

    if is_high:
        tier_label = "Elevated Risk Tier"
        tier_color = "#D70015"
        tier_badge_class = "badge-danger"
    elif is_mod:
        tier_label = "Moderate Risk Tier"
        tier_color = "#C93400"
        tier_badge_class = "badge-amber"
    else:
        tier_label = "Low Risk Tier"
        tier_color = "#248A3D"
        tier_badge_class = "badge-normal"

    highlights = []
    if is_low:
        highlights.append({"cat": "Nutrition", "text": "Low-GI Whole Foods", "badge": "badge-normal"})
        highlights.append({"cat": "Activity", "text": "150 min/wk Aerobic Base", "badge": "badge-normal"})
        highlights.append({"cat": "Vitals", "text": "Stable Metabolic Weight", "badge": "badge-normal"})
        highlights.append({"cat": "Monitoring", "text": "Annual Wellness Check", "badge": "badge-normal"})
    else:
        highlights.append({"cat": "Nutrition", "text": "Fiber-First Carbohydrate Moderation", "badge": "badge-danger" if is_high else "badge-amber"})
        highlights.append({"cat": "Activity", "text": "Post-Meal 15m Walks & Resistance", "badge": "badge-danger" if is_high else "badge-amber"})
        if bmi >= 25:
            highlights.append({"cat": "Vitals", "text": "Target 5%–7% Weight Reduction", "badge": "badge-danger" if bmi >= 30 else "badge-amber"})
        if is_detailed_bp and sbp is not None:
            if sbp >= 130 or dbp >= 80:
                highlights.append({"cat": "Vitals", "text": "DASH Diet & Sodium < 2.3g/day", "badge": "badge-danger" if sbp >= 140 else "badge-amber"})
        elif not is_detailed_bp:
            if dbp >= 80:
                highlights.append({"cat": "Vitals", "text": "DASH Diet & Sodium < 2.3g/day", "badge": "badge-danger" if dbp >= 90 else "badge-amber"})
        highlights.append({"cat": "Clinical", "text": "Confirmatory HbA1c Lab Panel", "badge": "badge-danger" if is_high else "badge-amber"})

    nutrition_items = []
    if is_low and g < 100:
        nutrition_items.append("Focus on a nutrient-dense whole foods diet emphasizing complex carbohydrates with a low glycemic index (GI), such as quinoa, steel-cut oats, and legumes.")
        nutrition_items.append("Integrate monounsaturated and omega-3 healthy fats (extra virgin olive oil, avocado, walnuts) to support insulin signaling and lipid homeostasis.")
        nutrition_items.append("Maintain optimal daily hydration of 2.0–2.5 L of water; avoid sugar-sweetened beverages and excessive fruit juices.")
    else:
        nutrition_items.append("Minimize refined carbohydrates, added sugars, white flour, and ultra-processed convenience foods to curb rapid postprandial glucose surges.")
        nutrition_items.append("Prioritize soluble dietary fiber (cruciferous vegetables, chia seeds, lentils, legumes) to delay gastric emptying and flatten glycemic excursions.")
        if g >= 100:
            nutrition_items.append("Practice low-glycemic meal sequencing: consume dietary fiber and lean proteins prior to starches to significantly reduce post-meal insulin spikes.")
        if (is_detailed_bp and sbp is not None and (sbp >= 130 or dbp >= 80)) or (not is_detailed_bp and dbp >= 80):
            nutrition_items.append("Adopt DASH dietary principles: increase potassium-rich whole foods and strictly restrict daily sodium to < 2,300 mg (ideally < 1,500 mg if hypertensive).")

    activity_items = []
    if is_low:
        activity_items.append("Maintain at least 150 minutes of moderate-intensity aerobic exercise per week (e.g., brisk walking, cycling, lap swimming) spread over 3–5 days.")
        activity_items.append("Incorporate 2 non-consecutive days of multi-joint resistance training to preserve lean skeletal muscle and metabolic resilience.")
    else:
        activity_items.append("Incorporate a 10–15 minute brisk walk immediately following main meals (breakfast/dinner) to facilitate non-insulin-mediated GLUT4 glucose uptake.")
        activity_items.append("Engage in structured progressive resistance training 2–3 times per week targeting major muscle groups to optimize peripheral insulin sensitivity.")
        if age >= 45:
            activity_items.append("Focus on low-impact, joint-friendly metabolic conditioning (e.g., incline treadmill walking, elliptical, swimming, pilates) to build muscular endurance.")

    vitals_items = []
    bp_elevated = (is_detailed_bp and sbp is not None and (sbp >= 130 or dbp >= 80)) or (not is_detailed_bp and dbp >= 80)
    if is_low and bmi < 25 and not bp_elevated:
        vitals_items.append("Sustain current body composition with balanced energy intake and consistent restorative sleep (7–9 hours nightly) to maintain cortisol balance.")
        if is_detailed_bp:
            vitals_items.append("Periodically self-monitor blood pressure to ensure systolic remains < 120 mm Hg and diastolic < 80 mm Hg.")
        else:
            vitals_items.append("Periodically self-monitor blood pressure to ensure diastolic remains < 80 mm Hg.")
    else:
        if bmi >= 25:
            vitals_items.append("Establish a moderate caloric deficit (300–500 kcal/day) targeting a gradual 5%–7% reduction in total body weight, which clinically improves beta-cell function.")
        else:
            vitals_items.append("Maintain lean body mass and prevent visceral fat deposition through balanced macronutrient distribution and adequate protein (1.2–1.6 g/kg).")
        if bp_elevated:
            if is_detailed_bp and sbp is not None:
                vitals_items.append(f"Monitor and log seated blood pressure twice daily; current reading ({sbp}/{dbp} mm Hg) indicates elevated vascular tone requiring sodium and stress modulation.")
            else:
                vitals_items.append(f"Monitor and log seated blood pressure twice daily; current diastolic reading ({dbp} mm Hg) indicates elevated diastolic vascular tone requiring sodium and stress modulation.")
        else:
            if is_detailed_bp:
                vitals_items.append("Maintain cardiovascular tracking; keep systolic blood pressure < 120 mm Hg and diastolic < 80 mm Hg through active lifestyle measures.")
            else:
                vitals_items.append("Maintain cardiovascular tracking; keep diastolic blood pressure < 80 mm Hg through active lifestyle measures.")

    monitoring_items = []
    if is_low:
        monitoring_items.append("Continue routine preventive annual or bi-annual health screenings including comprehensive metabolic panels and fasting lipid profiles.")
        monitoring_items.append("Repeat diabetes risk assessment annually or whenever substantial changes in weight, physical activity, or health status occur.")
    else:
        monitoring_items.append("Schedule an in-person clinical consultation with your primary healthcare provider to review this risk assessment and discuss confirmatory lab testing.")
        monitoring_items.append("Request an official diagnostic laboratory Glycated Hemoglobin (HbA1c) test and Fasting Plasma Glucose (FPG) panel to definitively evaluate glycemic control.")
        if is_detailed_bp and sbp is not None and (sbp >= 140 or dbp >= 90):
            monitoring_items.append("Consult your physician regarding formal clinical staging for hypertension and evaluation of cardiovascular risk markers.")
        elif not is_detailed_bp and dbp >= 90:
            monitoring_items.append("Consult your physician regarding formal clinical staging for diastolic hypertension and evaluation of cardiovascular risk markers.")
        if age >= 45:
            monitoring_items.append("Given patient age (≥ 45), adhere to ADA clinical guidelines recommending systematic multi-system metabolic and cardiovascular surveillance.")

    return {
        "risk_tier": tier_label,
        "tier_color": tier_color,
        "tier_badge_class": tier_badge_class,
        "highlights": highlights,
        "categories": [
            {
                "id": "nutrition",
                "name": "Nutrition & Dietary Strategy",
                "icon": "salad",
                "badge": "icon-badge-green",
                "items": nutrition_items,
            },
            {
                "id": "activity",
                "name": "Physical Activity & Metabolic Training",
                "icon": "flame",
                "badge": "icon-badge-amber",
                "items": activity_items,
            },
            {
                "id": "vitals",
                "name": "Weight & Vitals Management",
                "icon": "scale",
                "badge": "icon-badge-blue",
                "items": vitals_items,
            },
            {
                "id": "monitoring",
                "name": "Medical Monitoring & Clinical Follow-Up",
                "icon": "heart_pulse",
                "badge": "icon-badge-crimson",
                "items": monitoring_items,
            },
        ],
    }


def generate_clinical_pdf(patient_data, prediction, prob_diabetic, prob_non_diabetic, flags, omitted_features=None, patient_id="PT-2026-8841", recommendations=None, is_detailed_bp=True):
    """
    Generate a print-ready clinical screening report adhering strictly to
    standard hospital/clinical documentation standards.
    """
    if omitted_features is None:
        omitted_features = []
    if recommendations is None:
        recommendations = generate_lifestyle_recommendations(prob_diabetic, patient_data, flags, is_detailed_bp=is_detailed_bp)

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=letter,
        rightMargin=36, leftMargin=36, topMargin=32, bottomMargin=32
    )
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=15,
        leading=19,
        textColor=colors.HexColor('#1C1C1E'),
    )
    subtitle_style = ParagraphStyle(
        'DocSub',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor('#8E8E93'),
    )
    section_heading = ParagraphStyle(
        'SectionHead',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor('#007AFF'),
        spaceAfter=3,
        spaceBefore=2,
    )
    cell_bold = ParagraphStyle(
        'CellBold',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10,
        textColor=colors.HexColor('#1C1C1E'),
    )
    cell_regular = ParagraphStyle(
        'CellReg',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=10,
        textColor=colors.HexColor('#3A3A3C'),
    )
    
    elements = []
    
    # ── 1. Clinical Header ──
    elements.append(Paragraph("AI Diabetes Risk Assessment — Clinical Screening Report", title_style))
    elements.append(Paragraph("Rocheston RCAI Classroom Project &bull; Clinical Decision Support Telemetry", subtitle_style))
    elements.append(Spacer(1, 3))
    elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#007AFF"), spaceBefore=2, spaceAfter=5))
    
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
        ('PADDING', (0,0), (-1,-1), 3.5),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    elements.append(meta_table)
    elements.append(Spacer(1, 5))
    
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
        ('PADDING', (0,0), (-1,-1), 4.5),
        ('BOX', (0,0), (-1,-1), 1, risk_fg),
    ]))
    elements.append(findings_table)

    if omitted_features:
        elements.append(Spacer(1, 2))
        omitted_text = (
            f"<i><b>Clinical Transparency Note:</b> Assessment calculated using 4 core metrics + standard population "
            f"baselines for omitted lab tests ({', '.join(omitted_features)}).</i>"
        )
        elements.append(Paragraph(omitted_text, ParagraphStyle('OmitNote', parent=styles['Normal'], fontSize=7.5, leading=9, textColor=colors.HexColor('#007AFF'))))

    elements.append(Spacer(1, 5))
    
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
        "SystolicBP": ("mm Hg", "< 120 mm Hg (SBP)"),
        "BloodPressure": ("mm Hg", "< 80 mm Hg (DBP)"),
        "BMI": ("kg/m²", "18.5 – 24.9 kg/m²"),
        "Age": ("years", "18 – 44 years"),
        "Insulin": ("µU/mL", "16 – 166 µU/mL"),
        "SkinThickness": ("mm", "10 – 30 mm"),
        "Pregnancies": ("count", "0 – 2"),
        "DiabetesPedigreeFunction": ("score", "< 0.50"),
    }
    if is_detailed_bp:
        DISPLAY_REPORT_FEATURES = ["Glucose", "SystolicBP", "BloodPressure", "BMI", "Age", "Insulin", "SkinThickness", "Pregnancies", "DiabetesPedigreeFunction"]
    else:
        DISPLAY_REPORT_FEATURES = ["Glucose", "BloodPressure", "BMI", "Age", "Insulin", "SkinThickness", "Pregnancies", "DiabetesPedigreeFunction"]
    for k in DISPLAY_REPORT_FEATURES:
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
    
    param_table = Table(param_rows, colWidths=[155, 100, 95, 75, 115])
    param_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#E5E5EA')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#D1D1D6')),
        ('PADDING', (0,0), (-1,-1), 2.8),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#F9F9FB')]),
    ]))
    elements.append(param_table)
    elements.append(Spacer(1, 5))
    
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
        ('PADDING', (0,0), (-1,-1), 3.5),
    ]))
    elements.append(flag_table)
    elements.append(Spacer(1, 5))

    # ── 6. Personalized Health & Lifestyle Guidance ──
    elements.append(Paragraph("4. PERSONALIZED HEALTH & LIFESTYLE GUIDANCE", section_heading))
    guidance_rows = []
    for cat in recommendations["categories"]:
        cat_title = Paragraph(f"<b>&bull; {cat['name']}</b>", cell_bold)
        cat_items_text = "<br/>".join([f"&nbsp;&nbsp;&ndash;&nbsp;{it}" for it in cat["items"]])
        cat_body = Paragraph(cat_items_text, cell_regular)
        guidance_rows.append([cat_title])
        guidance_rows.append([cat_body])
    
    notice_text = (
        "<i><b>Algorithmic Lifestyle Notice:</b> These recommendations are algorithmic lifestyle suggestions based on "
        "clinical screening thresholds and should be discussed with a qualified medical provider.</i>"
    )
    guidance_rows.append([Paragraph(notice_text, ParagraphStyle('AlgNotice', parent=styles['Normal'], fontSize=7.5, leading=9.5, textColor=colors.HexColor('#007AFF')))])

    guidance_table = Table(guidance_rows, colWidths=[540])
    guidance_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#FAFAFA')),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#E5E5EA')),
        ('PADDING', (0,0), (-1,-1), 3),
        ('ROWBACKGROUNDS', (0,0), (-1,-1), [colors.HexColor('#F2F2F7'), colors.white]),
    ]))
    elements.append(guidance_table)
    elements.append(Spacer(1, 5))
    
    # ── 7. Regulatory Disclaimer ──
    elements.append(Paragraph("5. REGULATORY & MEDICAL DISCLAIMER", section_heading))
    disclaimer_text = (
        "<b>Regulatory & Medical Disclaimer:</b> This document is generated automatically by a statistical machine learning "
        "supervised classification pipeline for screening purposes and does not replace professional medical diagnosis. "
        "Always consult a licensed medical practitioner for comprehensive evaluation, official blood testing "
        "(e.g., HbA1c, OGTT), and tailored healthcare decisions."
    )
    disc_para = Paragraph(disclaimer_text, ParagraphStyle('Disc', parent=styles['Normal'], fontSize=7.5, leading=9.5, textColor=colors.HexColor('#636366')))
    disc_table = Table([[disc_para]], colWidths=[540])
    disc_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F2F2F7')),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#D1D1D6')),
        ('LINELEFT', (0,0), (0,-1), 3, colors.HexColor('#007AFF')),
        ('PADDING', (0,0), (-1,-1), 4),
    ]))
    elements.append(disc_table)
    
    doc.build(elements, canvasmaker=NumberedCanvas)
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
    raw_bytes = buf.getvalue()
    buf.close()
    plt.close(fig)
    return raw_bytes

@st.cache_data
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

@st.cache_data
def plot_feature_importance(_pipeline, feature_names):
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    fig.patch.set_facecolor('#F2F2F7')
    ax.set_facecolor('#F2F2F7')

    clf = _pipeline.named_steps.get("classifier", None)
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

@st.cache_data
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
def apply_preset_callback(preset_name):
    """Apply clinical preset values before widgets render without causing rerun loops."""
    st.session_state["profile_selection"] = preset_name
    st.session_state["preset_selector"] = preset_name
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

def on_preset_selector_change():
    """Callback triggered when the preset selectbox changes."""
    selected = st.session_state.get("preset_selector", "Custom Profile")
    apply_preset_callback(selected)

def on_custom_input_change(feat, source):
    """
    Unidirectional callback for all sliders and number inputs.
    Synchronizes twin inputs and resets active profile to Custom Profile immediately.
    """
    st.session_state["profile_selection"] = "Custom Profile"
    st.session_state["preset_selector"] = "Custom Profile"
    if source == "slider":
        st.session_state[f"num_{feat}"] = st.session_state[f"slider_{feat}"]
    elif source == "num":
        st.session_state[f"slider_{feat}"] = st.session_state[f"num_{feat}"]

# ─────────────────────────────────────────────
#  CLIENT-SIDE DUAL-INPUT SYNC ENGINE (JS)
# ─────────────────────────────────────────────
def inject_client_sync_engine():
    """
    Zero-Latency Dual-Input Synchronization & Dynamic Track Fill Engine.
    Runs in the browser DOM to provide:
      1. Centralized updateSliderProgress recalculating dynamic CSS variable
         (--progress and --slider-fill) and background gradient immediately on preset updates.
      2. 0 ms perceptible latency on every slider drag tick without blocking reruns.
      3. Stepper '+' / '-' decoupling with a fast local state buffer,
         preventing dropped clicks and UI freezes.
      4. Seamless capsule progress bar geometry under the translucent thumb.
      5. Compact Apple-style pill Developer drawer toggle.
    """
    components.html(
        """
        <script>
        (function() {
            const rootDoc = window.parent.document || document;

            // ── CENTRALIZED SLIDER TRACK PROGRESS CALCULATOR ──
            function updateSliderProgress(sliderElem) {
                if (!sliderElem) return;
                const rangeInput = sliderElem.querySelector('input[type="range"]');
                const thumb = sliderElem.querySelector('div[class*="efbyxod3"]') || sliderElem.querySelector('[role="slider"]');
                const track = sliderElem.querySelector('div[class*="efbyxod5"]') || sliderElem.querySelector('div[data-baseweb="slider"] > div') || sliderElem.querySelector('div[data-rac] > div');

                let min = 0, max = 100, val = 50;
                if (rangeInput) {
                    min = parseFloat(rangeInput.min) || 0;
                    max = parseFloat(rangeInput.max) || 100;
                    val = parseFloat(rangeInput.value) || 0;
                } else if (thumb) {
                    min = parseFloat(thumb.getAttribute('aria-valuemin')) || 0;
                    max = parseFloat(thumb.getAttribute('aria-valuemax')) || 100;
                    val = parseFloat(thumb.getAttribute('aria-valuenow')) || 0;
                }

                const pct = Math.max(0, Math.min(100, ((val - min) / (max - min)) * 100));
                const pctStr = pct.toFixed(2) + '%';

                sliderElem.style.setProperty('--progress', pctStr);
                sliderElem.style.setProperty('--slider-fill', pctStr);

                if (track) {
                    track.style.setProperty('--progress', pctStr);
                    track.style.setProperty('--slider-fill', pctStr);
                    track.style.background = 'linear-gradient(to right, #007AFF 0%, #007AFF ' + pctStr + ', #E5E5EA ' + pctStr + ', #E5E5EA 100%)';
                }
                if (thumb && !thumb.dataset.userDragging) {
                    thumb.style.left = pctStr;
                }
            }

            function updateAllSlidersProgress() {
                const sliders = rootDoc.querySelectorAll('div[data-testid="stSlider"]');
                sliders.forEach(s => updateSliderProgress(s));
            }

            function markProfileAsCustom() {
                const badge = rootDoc.getElementById('active-profile-badge');
                if (badge && badge.innerText !== 'Custom Profile') {
                    badge.innerText = 'Custom Profile';
                    badge.className = 'range-badge badge-amber';
                }
            }

            function setReactInputValue(input, val) {
                const lastVal = input.value;
                input.value = val;
                const tracker = input._valueTracker;
                if (tracker) {
                    tracker.setValue(lastVal);
                }
                input.dispatchEvent(new Event('input', { bubbles: true }));
                input.dispatchEvent(new Event('change', { bubbles: true }));
            }

            // ── ZERO-LATENCY DUAL-INPUT SYNCHRONIZATION & STEPPER BUFFER ──
            function initDualInputSync() {
                const sliders = Array.from(rootDoc.querySelectorAll('div[data-testid="stSlider"]'));
                const numbers = Array.from(rootDoc.querySelectorAll('div[data-testid="stNumberInput"]'));

                if (sliders.length === 0 || numbers.length === 0) return;

                const nativeInputSetter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;

                sliders.forEach(sliderElem => {
                    if (sliderElem.dataset.syncBound === "true") {
                        updateSliderProgress(sliderElem);
                        return;
                    }

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
                        const pctStr = pct.toFixed(2) + '%';

                        sliderElem.style.setProperty('--progress', pctStr);
                        sliderElem.style.setProperty('--slider-fill', pctStr);

                        if (track) {
                            track.style.setProperty('--progress', pctStr);
                            track.style.setProperty('--slider-fill', pctStr);
                            track.style.background = 'linear-gradient(to right, #007AFF 0%, #007AFF ' + pctStr + ', #E5E5EA ' + pctStr + ', #E5E5EA 100%)';
                        }
                        if (thumb) {
                            thumb.style.left = pctStr;
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

                    // 1. Instant client-side slider drag synchronization (0 ms)
                    rangeInput.addEventListener('input', (e) => {
                        markProfileAsCustom();
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
                        markProfileAsCustom();
                        if (thumb) thumb.dataset.userDragging = "true";
                        onPointerMove(e);
                    });

                    rootDoc.addEventListener('pointermove', (e) => {
                        if (!isDragging) return;
                        onPointerMove(e);
                    });

                    rootDoc.addEventListener('pointerup', () => {
                        if (!isDragging) return;
                        isDragging = false;
                        if (thumb) delete thumb.dataset.userDragging;
                        if (numInput) numInput.dispatchEvent(new Event('change', { bubbles: true }));
                        if (rangeInput) rangeInput.dispatchEvent(new Event('change', { bubbles: true }));
                        updateSliderProgress(sliderElem);
                    });

                    // 2. Stepper rapid-click freeze bug fix (decoupled local buffer)
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

                        markProfileAsCustom();
                        updatePillVisuals(stepperBuffer, 'number');

                        clearTimeout(stepperTimer);
                        stepperTimer = setTimeout(() => {
                            nativeInputSetter.call(numInput, formatVal(stepperBuffer));
                            numInput.dispatchEvent(new Event('change', { bubbles: true }));
                            nativeInputSetter.call(rangeInput, stepperBuffer.toString());
                            rangeInput.dispatchEvent(new Event('change', { bubbles: true }));
                            updateSliderProgress(sliderElem);
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

                    // 3. Direct text box typing synchronization
                    numInput.addEventListener('input', (e) => {
                        markProfileAsCustom();
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

            // ── RESTORE APPLE SF-STYLE ICONS & CHEVRONS ──
            function updateAppleIcons() {
                // 1. Collapsed Developer Toggle Button
                const expandBtn = rootDoc.querySelector('button[data-testid="stExpandSidebarButton"]');
                if (expandBtn) {
                    const defaultSvgs = expandBtn.querySelectorAll('svg:not(.apple-sidebar-toggle-icon)');
                    defaultSvgs.forEach(s => s.style.display = 'none');

                    if (!expandBtn.querySelector('.apple-chevrons-right-icon')) {
                        const rightSvg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
                        rightSvg.setAttribute('viewBox', '0 0 24 24');
                        rightSvg.setAttribute('class', 'apple-sidebar-toggle-icon apple-chevrons-right-icon');
                        rightSvg.innerHTML = '<path d="M13 17l5-5-5-5M6 17l5-5-5-5"/>';

                        let labelSpan = expandBtn.querySelector('.apple-dev-toggle-label');
                        if (!labelSpan) {
                            labelSpan = document.createElement('span');
                            labelSpan.className = 'apple-dev-toggle-label';
                            labelSpan.innerText = 'Developer';
                        }
                        expandBtn.appendChild(rightSvg);
                        expandBtn.appendChild(labelSpan);
                    }
                }

                // 2. Expanded Developer Collapse Button
                const collapseBtn = rootDoc.querySelector('div[data-testid="stSidebarCollapseButton"] button');
                if (collapseBtn) {
                    if (!collapseBtn.querySelector('.apple-chevrons-left-icon')) {
                        const leftSvg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
                        leftSvg.setAttribute('viewBox', '0 0 24 24');
                        leftSvg.setAttribute('class', 'apple-sidebar-toggle-icon apple-chevrons-left-icon');
                        leftSvg.innerHTML = '<path d="M11 19l-7-7 7-7M18 19l-7-7 7-7"/>';
                        collapseBtn.appendChild(leftSvg);
                    }
                }

                // 3. Advanced Clinical Parameters Expander Dropdown Chevron
                const expanders = rootDoc.querySelectorAll('div[data-testid="stExpander"]');
                expanders.forEach(expander => {
                    const summary = expander.querySelector('summary');
                    if (!summary) return;
                    if (!summary.querySelector('.apple-expander-chevron')) {
                        const chevronSvg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
                        chevronSvg.setAttribute('viewBox', '0 0 24 24');
                        chevronSvg.setAttribute('class', 'apple-expander-chevron');
                        chevronSvg.innerHTML = '<path d="M6 9l6 6 6-6"/>';

                        const titleTarget = summary.querySelector('[data-testid="stMarkdownContainer"]') || summary;
                        if (titleTarget.parentElement === summary) {
                            summary.insertBefore(chevronSvg, titleTarget);
                        } else {
                            titleTarget.parentElement.insertBefore(chevronSvg, titleTarget);
                        }
                    }
                });

                // 4. Download Assessment Report Vector SVG Icon
                const dlContainers = rootDoc.querySelectorAll('div[data-testid="stDownloadButton"]');
                dlContainers.forEach(dlContainer => {
                    const btn = dlContainer.querySelector('button');
                    if (btn) {
                        const p = btn.querySelector('p') || btn;
                        if (p && p.innerText && (p.innerText.includes('📄') || p.innerText.includes('📥'))) {
                            p.innerText = p.innerText.replace('📄', '').replace('📥', '').trim();
                        }
                        if (!btn.querySelector('.apple-report-svg-icon')) {
                            const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
                            svg.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
                            svg.setAttribute('width', '18');
                            svg.setAttribute('height', '18');
                            svg.setAttribute('viewBox', '0 0 24 24');
                            svg.setAttribute('fill', 'none');
                            svg.setAttribute('stroke', 'currentColor');
                            svg.setAttribute('stroke-width', '2');
                            svg.setAttribute('stroke-linecap', 'round');
                            svg.setAttribute('stroke-linejoin', 'round');
                            svg.setAttribute('class', 'apple-report-svg-icon');
                            svg.style.marginRight = '8px';
                            svg.style.display = 'inline-block';
                            svg.style.verticalAlign = 'middle';
                            svg.innerHTML = '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="12" y1="18" x2="12" y2="12"></line><line x1="9" y1="15" x2="12" y2="18"></line><line x1="15" y1="15" x2="12" y2="18"></line>';
                            btn.insertBefore(svg, btn.firstChild);
                        }
                    }
                });
            }

            // ── DIRECT CLIENT-SIDE BLOB DOWNLOAD & TOAST CONFIRMATION ──
            function ensureToastContainer() {
                let toast = rootDoc.getElementById('apple-feedback-toast');
                if (!toast) {
                    toast = rootDoc.createElement('div');
                    toast.id = 'apple-feedback-toast';
                    toast.className = 'apple-toast';
                    toast.innerHTML = `
                        <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="#34C759" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                            <path d="M20 6L9 17l-5-5"/>
                        </svg>
                        <span>Report successfully generated and saved to your Downloads</span>
                    `;
                    rootDoc.body.appendChild(toast);
                }
                return toast;
            }

            window.parent.showAppleToast = function(msg) {
                const toast = ensureToastContainer();
                if (msg) toast.querySelector('span').innerText = msg;
                toast.classList.add('show');
                clearTimeout(window.parent.__toastTimeout);
                window.parent.__toastTimeout = setTimeout(() => {
                    toast.classList.remove('show');
                }, 3800);
            };

            window.parent.triggerDirectBlobDownload = function() {
                try {
                    const b64 = window.parent.__currentPdfBase64;
                    const filename = window.parent.__currentPdfFilename || 'Diabetes_Assessment_Report.pdf';
                    if (b64) {
                        const byteCharacters = atob(b64);
                        const byteNumbers = new Array(byteCharacters.length);
                        for (let i = 0; i < byteCharacters.length; i++) {
                            byteNumbers[i] = byteCharacters.charCodeAt(i);
                        }
                        const byteArray = new Uint8Array(byteNumbers);
                        const blob = new Blob([byteArray], { type: 'application/pdf' });
                        const blobUrl = URL.createObjectURL(blob);
                        const link = rootDoc.createElement('a');
                        link.href = blobUrl;
                        link.download = filename;
                        link.style.display = 'none';
                        rootDoc.body.appendChild(link);
                        link.click();
                        rootDoc.body.removeChild(link);
                        setTimeout(() => URL.revokeObjectURL(blobUrl), 1500);
                    }
                } catch (err) {
                    console.error('Direct PDF download error:', err);
                }
                window.parent.showAppleToast("Report successfully generated and saved to your Downloads");
            };

            // Run immediately and monitor DOM additions
            updateAllSlidersProgress();
            initDualInputSync();
            updateAppleIcons();

            setInterval(() => {
                updateAllSlidersProgress();
                initDualInputSync();
                updateAppleIcons();
            }, 120);

            // Hook into click and change events to update slider fills instantly on presets
            rootDoc.addEventListener('click', () => {
                setTimeout(updateAllSlidersProgress, 20);
                setTimeout(updateAllSlidersProgress, 100);
                setTimeout(updateAllSlidersProgress, 250);
            });
            rootDoc.addEventListener('change', () => {
                setTimeout(updateAllSlidersProgress, 20);
                setTimeout(updateAllSlidersProgress, 100);
            });

            const domObserver = new MutationObserver(() => {
                updateAllSlidersProgress();
                initDualInputSync();
                updateAppleIcons();
            });
            domObserver.observe(rootDoc.body, { childList: true, subtree: true });

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

    # Master state variables for unidirectional flow
    if "profile_selection" not in st.session_state:
        st.session_state["profile_selection"] = "Custom Profile"
    if "preset_selector" not in st.session_state:
        st.session_state["preset_selector"] = "Custom Profile"

    # Initialize session states for all features
    for f in FEATURE_META.keys():
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
            Diagnostic Telemetry &amp; Model Benchmarking
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
        curr_profile = st.session_state.get("profile_selection", "Custom Profile")
        badge_cls = "badge-amber" if curr_profile == "Custom Profile" else "badge-normal"
        st.markdown(f'''
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 6px;">
            <div class="section-label" style="margin-bottom: 0;">CLINICAL PRESET PROFILE</div>
            <span id="active-profile-badge" class="range-badge {badge_cls}" style="font-size: 0.76rem; padding: 2px 8px;">{curr_profile}</span>
        </div>
        ''', unsafe_allow_html=True)

        preset_options = list(PRESETS.keys())
        preset_idx = preset_options.index(st.session_state.get("preset_selector", "Custom Profile")) if st.session_state.get("preset_selector") in preset_options else 0
        st.selectbox(
            label="Clinical Preset",
            options=preset_options,
            index=preset_idx,
            key="preset_selector",
            on_change=on_preset_selector_change,
            label_visibility="collapsed",
        )

        # Quick Preset Buttons (Explicit on_click callbacks)
        col_p1, col_p2, col_p3, col_p4 = st.columns(4)
        with col_p1:
            st.button("Normal", type="secondary", key="btn_preset_normal", on_click=apply_preset_callback, args=("Normal Clinical Profile",), use_container_width=True)
        with col_p2:
            st.button("Borderline", type="secondary", key="btn_preset_borderline", on_click=apply_preset_callback, args=("Borderline Risk Profile",), use_container_width=True)
        with col_p3:
            st.button("Elevated", type="secondary", key="btn_preset_elevated", on_click=apply_preset_callback, args=("Elevated Risk Profile",), use_container_width=True)
        with col_p4:
            st.button("Obese / HTN", type="secondary", key="btn_preset_obese", on_click=apply_preset_callback, args=("Clinical Obesity & Hypertension Profile",), use_container_width=True)

        # ── 1. Primary Clinical Metrics (Required) Glass Container ──
        st.markdown('<div class="section-label" style="margin-top: 0.6rem;">PRIMARY CLINICAL METRICS (REQUIRED)</div>', unsafe_allow_html=True)
        with st.container(border=True):
            # 1. Glucose
            meta_g = FEATURE_META["Glucose"]
            st.markdown(f'''
            <div class="metric-row-header">
                <div class="metric-title-group">
                    {render_badge(meta_g["icon"], meta_g["badge"])}
                    <span class="metric-row-title">{meta_g["label"]}</span>
                </div>
                <span class="metric-row-unit">({meta_g["unit"]})</span>
            </div>
            ''', unsafe_allow_html=True)
            c_slide, c_num = st.columns([3.4, 1.25])
            with c_slide:
                st.slider(
                    label=meta_g["label"],
                    min_value=meta_g["type"](meta_g["min"]),
                    max_value=meta_g["type"](meta_g["max"]),
                    step=meta_g["type"](meta_g["step"]) if meta_g["type"] == float else meta_g["step"],
                    key="slider_Glucose",
                    on_change=on_custom_input_change,
                    args=("Glucose", "slider"),
                    label_visibility="collapsed",
                )
            with c_num:
                st.number_input(
                    label=meta_g["label"],
                    min_value=meta_g["type"](meta_g["min"]),
                    max_value=meta_g["type"](meta_g["max"]),
                    step=meta_g["type"](meta_g["step"]) if meta_g["type"] == float else meta_g["step"],
                    key="num_Glucose",
                    on_change=on_custom_input_change,
                    args=("Glucose", "num"),
                    label_visibility="collapsed",
                )
            input_values["Glucose"] = st.session_state.get("num_Glucose", st.session_state["slider_Glucose"])

            # 2. Blood Pressure (Detailed Entry Only: SBP + DBP)
            # Systolic Blood Pressure (SBP)
            meta_sbp = FEATURE_META["SystolicBP"]
            st.markdown(f'''
            <div class="metric-row-header" style="margin-top: 6px;">
                <div class="metric-title-group">
                    {render_badge(meta_sbp["icon"], meta_sbp["badge"])}
                    <span class="metric-row-title">{meta_sbp["label"]}</span>
                </div>
                <span class="metric-row-unit">({meta_sbp["unit"]})</span>
            </div>
            ''', unsafe_allow_html=True)
            c_slide, c_num = st.columns([3.4, 1.25])
            with c_slide:
                st.slider(
                    label=meta_sbp["label"],
                    min_value=meta_sbp["type"](meta_sbp["min"]),
                    max_value=meta_sbp["type"](meta_sbp["max"]),
                    step=meta_sbp["type"](meta_sbp["step"]) if meta_sbp["type"] == float else meta_sbp["step"],
                    key="slider_SystolicBP",
                    on_change=on_custom_input_change,
                    args=("SystolicBP", "slider"),
                    label_visibility="collapsed",
                )
            with c_num:
                st.number_input(
                    label=meta_sbp["label"],
                    min_value=meta_sbp["type"](meta_sbp["min"]),
                    max_value=meta_sbp["type"](meta_sbp["max"]),
                    step=meta_sbp["type"](meta_sbp["step"]) if meta_sbp["type"] == float else meta_sbp["step"],
                    key="num_SystolicBP",
                    on_change=on_custom_input_change,
                    args=("SystolicBP", "num"),
                    label_visibility="collapsed",
                )
            input_values["SystolicBP"] = st.session_state.get("num_SystolicBP", st.session_state["slider_SystolicBP"])

            # Diastolic Blood Pressure (DBP)
            meta_dbp = FEATURE_META["BloodPressure"]
            st.markdown(f'''
            <div class="metric-row-header" style="margin-top: 6px;">
                <div class="metric-title-group">
                    {render_badge(meta_dbp["icon"], meta_dbp["badge"])}
                    <span class="metric-row-title">{meta_dbp["label"]}</span>
                </div>
                <span class="metric-row-unit">({meta_dbp["unit"]})</span>
            </div>
            ''', unsafe_allow_html=True)
            c_slide, c_num = st.columns([3.4, 1.25])
            with c_slide:
                st.slider(
                    label=meta_dbp["label"],
                    min_value=meta_dbp["type"](meta_dbp["min"]),
                    max_value=meta_dbp["type"](meta_dbp["max"]),
                    step=meta_dbp["type"](meta_dbp["step"]) if meta_dbp["type"] == float else meta_dbp["step"],
                    key="slider_BloodPressure",
                    on_change=on_custom_input_change,
                    args=("BloodPressure", "slider"),
                    label_visibility="collapsed",
                )
            with c_num:
                st.number_input(
                    label=meta_dbp["label"],
                    min_value=meta_dbp["type"](meta_dbp["min"]),
                    max_value=meta_dbp["type"](meta_dbp["max"]),
                    step=meta_dbp["type"](meta_dbp["step"]) if meta_dbp["type"] == float else meta_dbp["step"],
                    key="num_BloodPressure",
                    on_change=on_custom_input_change,
                    args=("BloodPressure", "num"),
                    label_visibility="collapsed",
                )
            input_values["BloodPressure"] = st.session_state.get("num_BloodPressure", st.session_state["slider_BloodPressure"])
            is_detailed_bp = True

            st.markdown('''
            <div class="bp-telemetry-note">
                Dual BP Telemetry: Systolic SBP provides AHA cardiovascular staging &amp; lifestyle guidance, while Diastolic DBP feeds the ML classification model.
            </div>
            ''', unsafe_allow_html=True)

            # 3. BMI and Age
            for feat in ["BMI", "Age"]:
                meta = FEATURE_META[feat]
                unit_str = f"({meta['unit']})" if meta["unit"] else ""
                st.markdown(f'''
                <div class="metric-row-header">
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
                        step=meta["type"](meta["step"]) if meta["type"] == float else meta["step"],
                        key=f"slider_{feat}",
                        on_change=on_custom_input_change,
                        args=(feat, "slider"),
                        label_visibility="collapsed",
                    )
                with c_num:
                    st.number_input(
                        label=meta["label"],
                        min_value=meta["type"](meta["min"]),
                        max_value=meta["type"](meta["max"]),
                        step=meta["type"](meta["step"]) if meta["type"] == float else meta["step"],
                        key=f"num_{feat}",
                        on_change=on_custom_input_change,
                        args=(feat, "num"),
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
                                step=meta["type"](meta["step"]) if meta["type"] == float else meta["step"],
                                key=f"slider_{feat}",
                                on_change=on_custom_input_change,
                                args=(feat, "slider"),
                                label_visibility="collapsed",
                            )
                        with c_num:
                            st.number_input(
                                label=meta["label"],
                                min_value=meta["type"](meta["min"]),
                                max_value=meta["type"](meta["max"]),
                                step=meta["type"](meta["step"]) if meta["type"] == float else meta["step"],
                                key=f"num_{feat}",
                                on_change=on_custom_input_change,
                                args=(feat, "num"),
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
        
        # Reference Row 1: Glucose
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

        # Reference Row 2: Blood Pressure (SBP / DBP)
        st.markdown(f'''
        <div class="ref-card-row">
            <div class="ref-meta-left">
                {render_badge("heart_pulse", "icon-badge-crimson")}
                <div>
                    <div class="ref-metric-name">Blood Pressure (SBP / DBP)</div>
                    <div class="ref-metric-unit">Normal SBP: &lt; 120 mm Hg &bull; DBP: &lt; 80 mm Hg</div>
                </div>
            </div>
            <div class="ref-badges-right">
                <span class="range-badge badge-normal">Normal: &lt; 120/80</span>
                <span class="range-badge badge-danger">Hypertension: &ge; 130/80 mm Hg</span>
            </div>
        </div>
        ''', unsafe_allow_html=True)

        # Reference Row 3: BMI
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

        # Reference Row 4: Age
        st.markdown(f'''
        <div class="ref-card-row">
            <div class="ref-meta-left">
                {render_badge("calendar", "icon-badge-amber")}
                <div>
                    <div class="ref-metric-name">Age &amp; Metabolic Baseline</div>
                    <div class="ref-metric-unit">Standard Baseline: 18–44 yrs</div>
                </div>
            </div>
            <div class="ref-badges-right">
                <span class="range-badge badge-normal">Baseline: &lt; 45</span>
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
                <span class="range-badge badge-danger">Elevated: &gt; 166</span>
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
    if "has_run_prediction" not in st.session_state:
        st.session_state["has_run_prediction"] = False
    if "assessment_result" not in st.session_state:
        st.session_state["assessment_result"] = None

    if predict_btn:
        st.session_state.predict_counter += 1
        current_run_id = st.session_state.predict_counter

        # Determine features that were omitted/imputed
        omitted_in_assessment = [f for f in ADVANCED_FEATURES if f in omitted_list]

        # ── DUAL-MODEL ROUTING & IMPUTATION INFERENCE ──
        if len(omitted_in_assessment) == len(ADVANCED_FEATURES) and pipeline_4 is not None:
            # Approach B: Dedicated 4-feature model routing
            core_row = {feat: input_values[feat] for feat in PRIMARY_FEATURES}
            patient_df = pd.DataFrame([core_row], columns=PRIMARY_FEATURES)
            prediction = int(pipeline_4.predict(patient_df)[0])
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
            prediction = int(pipeline_8.predict(patient_df)[0])
            if hasattr(pipeline_8, "predict_proba"):
                probs = pipeline_8.predict_proba(patient_df)[0]
            else:
                probs = np.array([1 - prediction, prediction], dtype=float)

        prob_diabetic     = float(probs[1])
        prob_non_diabetic = float(probs[0])

        # Compile clinical risk factor flags
        flags = []
        g = input_values.get("Glucose", 0)
        if g >= 126:
            flags.append("Fasting Glucose ≥ 126 mg/dL — Diagnostic of diabetes threshold")
        elif g >= 100:
            flags.append("Fasting Glucose 100–125 mg/dL — Impaired fasting glucose (Prediabetes)")

        is_detailed_bp = True
        sbp = input_values.get("SystolicBP", 120)
        dbp = input_values.get("BloodPressure", 70)
        if sbp >= 140 or dbp >= 90:
            flags.append(f"Blood Pressure {sbp}/{dbp} mm Hg — Stage 2 Hypertension range (High cardiovascular & metabolic risk)")
        elif sbp >= 130 or dbp >= 80:
            flags.append(f"Blood Pressure {sbp}/{dbp} mm Hg — Stage 1 Hypertension / Elevated (Target < 120/80 mm Hg)")
        elif sbp >= 120 and dbp < 80:
            flags.append(f"Blood Pressure {sbp}/{dbp} mm Hg — Elevated Systolic Blood Pressure (AHA Guidelines)")

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

        rec_data = generate_lifestyle_recommendations(prob_diabetic, input_values, flags, is_detailed_bp=is_detailed_bp)

        patient_id_code = f"PT-{datetime.now().strftime('%Y%m%d')}-8841"
        pdf_bytes = generate_clinical_pdf(
            patient_data=input_values,
            prediction=prediction,
            prob_diabetic=prob_diabetic,
            prob_non_diabetic=prob_non_diabetic,
            flags=flags,
            omitted_features=omitted_in_assessment,
            patient_id=patient_id_code,
            recommendations=rec_data,
            is_detailed_bp=is_detailed_bp,
        )

        b64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
        date_stamp = datetime.now().strftime('%Y%m%d')
        report_filename = f"Diabetes_Assessment_Report_{date_stamp}.pdf"

        st.session_state["has_run_prediction"] = True
        st.session_state["assessment_result"] = {
            "prediction": prediction,
            "prob_diabetic": prob_diabetic,
            "prob_non_diabetic": prob_non_diabetic,
            "flags": flags,
            "omitted_in_assessment": omitted_in_assessment,
            "patient_id_code": patient_id_code,
            "pdf_bytes": pdf_bytes,
            "b64_pdf": b64_pdf,
            "filename": report_filename,
            "rec_data": rec_data,
            "should_scroll": True,
            "run_id": current_run_id,
            "is_detailed_bp": is_detailed_bp,
        }

    # ── PERSISTENT ASSESSMENT FINDINGS & CLINICAL PDF EXPORT ──
    if st.session_state.get("has_run_prediction", False) and st.session_state.get("assessment_result") is not None:
        res = st.session_state["assessment_result"]
        prediction = res["prediction"]
        prob_diabetic = res["prob_diabetic"]
        prob_non_diabetic = res["prob_non_diabetic"]
        flags = res["flags"]
        omitted_in_assessment = res["omitted_in_assessment"]
        patient_id_code = res["patient_id_code"]
        pdf_bytes = res["pdf_bytes"]
        b64_pdf = res["b64_pdf"]
        report_filename = res["filename"]
        res_is_detailed_bp = res.get("is_detailed_bp", True)

        # Target DOM anchor for scrolling
        st.markdown('<div id="assessment-result-anchor" style="padding-top: 1rem;"></div>', unsafe_allow_html=True)

        if res.get("should_scroll", False):
            res["should_scroll"] = False
            components.html(
                f"""
                <script>
                    (function() {{
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

        st.markdown('<div class="section-label" style="margin-top: 1.4rem;">RISK STRATIFICATION &amp; FINDINGS</div>', unsafe_allow_html=True)
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
                <div class="risk-prob-bar">
                    <div class="risk-prob-fill" style="width: {prob_diabetic*100:.1f}%;"></div>
                </div>
                <p style="font-size: 0.88rem; margin: 0; color: #1C1C1E; font-weight: 500;">
                    Calculated Diabetic Probability: <strong>{prob_diabetic*100:.1f}%</strong> &bull; Non-Diabetic Probability: <strong>{prob_non_diabetic*100:.1f}%</strong>
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

        # ── APPLE HEALTH PERSONALIZED LIFESTYLE & HEALTH RECOMMENDATIONS ──
        rec_data = res.get("rec_data")
        if not rec_data:
            rec_data = generate_lifestyle_recommendations(prob_diabetic, input_values, flags, is_detailed_bp=res_is_detailed_bp)

        rec_cards_html = "".join([
            f'<div class="recommendation-card">'
            f'<div class="rec-header">{render_badge(cat["icon"], cat["badge"])}<div class="rec-cat-title">{cat["name"]}</div></div>'
            f'<ul class="rec-bullet-list">' + "".join([
                f'<li class="rec-bullet-item"><span class="rec-check-icon">{SVG_ICONS["check"]}</span><span>{it}</span></li>'
                for it in cat["items"]
            ]) + '</ul></div>'
            for cat in rec_data["categories"]
        ])

        highlights_html = "".join([
            f'<span class="range-badge {hl["badge"]}" style="margin-right: 6px; margin-bottom: 4px;"><strong>{hl["cat"]}:</strong>&nbsp;{hl["text"]}</span>'
            for hl in rec_data["highlights"]
        ])

        rec_container_html = (
            f'<div class="liquid-glass-card">'
            f'<div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px; margin-bottom: 0.85rem;">'
            f'<div style="display: flex; align-items: center; gap: 10px;">'
            f'{render_badge("salad", "icon-badge-green")}'
            f'<div>'
            f'<div style="font-size: 1.18rem; font-weight: 800; color: #1C1C1E; letter-spacing: -0.015em; line-height: 1.25;">Personalized Health &amp; Lifestyle Guidance</div>'
            f'<span style="font-size: 0.74rem; font-weight: 600; color: #8E8E93; text-transform: uppercase; letter-spacing: 0.6px;">Apple Health Clinical Strategy • Tailored Metabolic Directive</span>'
            f'</div></div>'
            f'<div><span class="range-badge {rec_data["tier_badge_class"]}">{rec_data["risk_tier"]}</span></div>'
            f'</div>'
            f'<div style="display: flex; flex-wrap: wrap; align-items: center; gap: 6px; margin-bottom: 0.95rem; padding: 0.65rem 0.90rem; background: rgba(255, 255, 255, 0.45); border-radius: 14px; border: 1px solid rgba(255, 255, 255, 0.65);">'
            f'<span style="font-size: 0.74rem; font-weight: 700; color: #8E8E93; text-transform: uppercase; letter-spacing: 0.5px; margin-right: 4px;">Actionable Highlights:</span>'
            f'{highlights_html}</div>'
            f'<div class="recommendation-grid">{rec_cards_html}</div>'
            f'<div style="margin-top: 1.05rem; padding-top: 0.75rem; border-top: 1px solid rgba(0, 0, 0, 0.06); font-size: 0.77rem; color: #8E8E93; display: flex; align-items: center; gap: 8px; line-height: 1.4;">'
            f'{SVG_ICONS["info"]}<span><strong>Clinical Notice:</strong> These recommendations are algorithmic lifestyle suggestions based on clinical screening thresholds and should be discussed with a qualified medical provider.</span>'
            f'</div></div>'
        )

        st.markdown(rec_container_html, unsafe_allow_html=True)

        # ── 4. CENTERED CLINICAL PDF REPORT DOWNLOAD (Apple Pill) ──
        st.download_button(
            label="Download Assessment Report (PDF)",
            data=pdf_bytes,
            file_name=report_filename,
            mime="application/pdf",
            use_container_width=False,
            key="download_assessment_report_pdf"
        )

        # Direct client-side blob download trigger & toast confirmation hook
        components.html(
            f"""
            <script>
                (function() {{
                    const rootDoc = window.parent.document || document;
                    window.parent.__currentPdfBase64 = "{b64_pdf}";
                    window.parent.__currentPdfFilename = "{report_filename}";

                    function attachDownloadHook() {{
                        const dlContainer = rootDoc.querySelector('div[data-testid="stDownloadButton"]');
                        if (!dlContainer) return;
                        const btn = dlContainer.querySelector('button');
                        if (btn && !btn.dataset.appleDownloadBound) {{
                            btn.dataset.appleDownloadBound = "true";
                            btn.addEventListener('click', function(e) {{
                                if (window.parent.triggerDirectBlobDownload) {{
                                    window.parent.triggerDirectBlobDownload();
                                }}
                            }}, false);
                        }}
                        if (window.parent.updateAppleIcons) {{
                            window.parent.updateAppleIcons();
                        }}
                    }}
                    attachDownloadHook();
                    setTimeout(attachDownloadHook, 120);
                    setTimeout(attachDownloadHook, 400);
                }})();
            </script>
            """,
            height=0,
            width=0,
        )


        # Regulatory Disclaimer
        st.markdown(f'''
        <div class="disclaimer">
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 4px;">
                {SVG_ICONS["info"]}
                <strong>Regulatory &amp; Medical Disclaimer</strong>
            </div>
            This clinical decision-support tool is an AI-assisted screening model developed for academic purposes.
            It does not replace professional medical diagnosis. Always consult a licensed healthcare provider
            for clinical evaluation, laboratory diagnostic testing (HbA1c, OGTT), and individualized care.
        </div>
        ''', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
