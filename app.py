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
DATA_PATH  = os.path.join(os.path.dirname(__file__), "diabetes.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "Diabetesmodel.pkl")

ALL_FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
]
TARGET = "Outcome"

FEATURE_META = {
    "Glucose": {
        "label": "Fasting Blood Glucose",
        "min": 0, "max": 300, "default": 120, "step": 1,
        "unit": "mg/dL", "type": int, "primary": True,
        "icon": "droplet", "badge": "icon-badge-rose"
    },
    "BloodPressure": {
        "label": "Blood Pressure (Diastolic)",
        "min": 0, "max": 200, "default": 70, "step": 1,
        "unit": "mm Hg", "type": int, "primary": True,
        "icon": "heart_pulse", "badge": "icon-badge-crimson"
    },
    "BMI": {
        "label": "Body Mass Index (BMI)",
        "min": 10.0, "max": 60.0, "default": 25.0, "step": 0.1,
        "unit": "kg/m²", "type": float, "primary": True,
        "icon": "scale", "badge": "icon-badge-blue"
    },
    "Age": {
        "label": "Patient Age",
        "min": 1, "max": 120, "default": 30, "step": 1,
        "unit": "years", "type": int, "primary": True,
        "icon": "calendar", "badge": "icon-badge-amber"
    },
    "Insulin": {
        "label": "2-Hour Serum Insulin",
        "min": 0, "max": 900, "default": 80, "step": 1,
        "unit": "µU/mL", "type": int, "primary": False,
        "icon": "syringe", "badge": "icon-badge-teal"
    },
    "SkinThickness": {
        "label": "Triceps Skin Fold Thickness",
        "min": 0, "max": 100, "default": 20, "step": 1,
        "unit": "mm", "type": int, "primary": False,
        "icon": "layers", "badge": "icon-badge-amber"
    },
    "Pregnancies": {
        "label": "Number of Pregnancies",
        "min": 0, "max": 20, "default": 1, "step": 1,
        "unit": "count", "type": int, "primary": False,
        "icon": "user", "badge": "icon-badge-purple"
    },
    "DiabetesPedigreeFunction": {
        "label": "Diabetes Pedigree Function",
        "min": 0.0, "max": 2.5, "default": 0.47, "step": 0.01,
        "unit": "score", "type": float, "primary": False,
        "icon": "dna", "badge": "icon-badge-purple"
    },
}

PRESETS = {
    "Custom Profile": None,
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
#  VECTOR SVG ICONS (Apple Minimalist Line Style)
# ─────────────────────────────────────────────
SVG_ICONS = {
    "droplet": '<svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="#FF2D55" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2.69l5.66 5.66a8 8 0 1 1-11.31 0z"/></svg>',
    "heart_pulse": '<svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="#FF3B30" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M19 14c1.49-1.46 3-3.21 3-5.5A5.5 5.5 0 0 0 16.5 3c-1.76 0-3 .5-4.5 2-1.5-1.5-2.74-2-4.5-2A5.5 5.5 0 0 0 2 8.5c0 2.3 1.5 4.05 3 5.5l7 7Z"/><path d="M3.22 12H9.5l.5-1 2 4.5 2-7 1.5 3.5h4.28"/></svg>',
    "scale": '<svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="#007AFF" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m16 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z"/><path d="m2 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z"/><path d="M7 21h10"/><path d="M12 3v18"/><path d="M3 7h2c2 0 5-1 7-2 2 1 5 2 7 2h2"/></svg>',
    "calendar": '<svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="#FF9500" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="18" height="18" x="3" y="4" rx="2" ry="2"/><line x1="16" x2="16" y1="2" y2="6"/><line x1="8" x2="8" y1="2" y2="6"/><line x1="3" x2="21" y1="10" y2="10"/></svg>',
    "syringe": '<svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="#30B0C7" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m18 2 4 4"/><path d="m17 7 3-3"/><path d="M19 9 8.7 19.3c-1 1-2.5 1-3.4 0l-.6-.6c-1-1-1-2.5 0-3.4L15 5"/><path d="m9 11 4 4"/><path d="m5 19-3 3"/><path d="m14 4 6 6"/></svg>',
    "layers": '<svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="#FF9500" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/></svg>',
    "user": '<svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="#5856D6" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>',
    "dna": '<svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="#AF52DE" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M2 15c6.667-6 13.333 0 20-6"/><path d="M9 22c1.798-1.998 2.518-3.995 2.807-5.993"/><path d="M15 2c-1.798 1.998-2.518 3.995-2.807 5.993"/><path d="m17 6-2.5-2.5"/><path d="m14 8-1-1"/><path d="m7 18 2.5 2.5"/><path d="m3.5 14.5.5.5"/><path d="m20 9 .5.5"/><path d="m6.5 12.5 1 1"/><path d="m16.5 10.5 1 1"/><path d="m10 16 1 1"/></svg>',
    "activity": '<svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="#007AFF" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>',
    "shield_check": '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#34C759" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><path d="m9 12 2 2 4-4"/></svg>',
    "alert_triangle": '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#FF3B30" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>',
    "alert_circle": '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#FF9500" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>',
    "file_text": '<svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="#007AFF" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><line x1="10" y1="9" x2="8" y2="9"/></svg>',
    "info": '<svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="#007AFF" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>',
    "check": '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#34C759" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>',
}

def render_badge(icon_name, badge_class="icon-badge-blue"):
    svg_str = SVG_ICONS.get(icon_name, "")
    return f'<div class="icon-badge {badge_class}">{svg_str}</div>'

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AI Diabetes Risk Assessment",
    page_icon="https://raw.githubusercontent.com/feathericons/feather/master/icons/activity.svg",
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
    left: 5%;
    width: 580px;
    height: 580px;
    background: radial-gradient(circle, rgba(0, 122, 255, 0.25) 0%, rgba(88, 86, 214, 0.14) 45%, transparent 70%);
    filter: blur(90px);
    -webkit-filter: blur(90px);
    border-radius: 50%;
    z-index: 0;
    pointer-events: none;
    transform: translateZ(0);
}

.ambient-orb-cyan {
    position: fixed;
    bottom: 5%;
    right: 5%;
    width: 620px;
    height: 620px;
    background: radial-gradient(circle, rgba(50, 199, 235, 0.28) 0%, rgba(52, 199, 89, 0.14) 50%, transparent 70%);
    filter: blur(95px);
    -webkit-filter: blur(95px);
    border-radius: 50%;
    z-index: 0;
    pointer-events: none;
    transform: translateZ(0);
}

/* ── Container Layout: Wide viewport for side-by-side zero scroll ── */
.block-container {
    position: relative !important;
    z-index: 1 !important;
    padding-top: 1.2rem !important;
    padding-bottom: 3.5rem !important;
    max-width: 1320px !important;
}

/* ── Clean Headers ──────────────────────────────────────── */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header[data-testid="stHeader"] {
    background: rgba(242, 242, 247, 0.45) !important;
    backdrop-filter: blur(28px) saturate(210%) !important;
    -webkit-backdrop-filter: blur(28px) saturate(210%) !important;
    border-bottom: 1px solid rgba(255, 255, 255, 0.6) !important;
}

/* ── Global Typography & Contrast ───────────────────────── */
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
    line-height: 1.45;
}

/* ── Hero Banner ────────────────────────────────────────── */
.hero-banner {
    background: linear-gradient(135deg, #007AFF 0%, #5856D6 55%, #AF52DE 100%);
    border-radius: 20px;
    padding: 1.6rem 2.2rem;
    margin-bottom: 1.2rem;
    color: #FFFFFF !important;
    box-shadow: 0 10px 32px rgba(0, 122, 255, 0.22);
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
    font-size: 1.85rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.03em !important;
    margin: 0 0 0.3rem 0 !important;
    color: #FFFFFF !important;
}
.hero-banner p {
    font-size: 0.96rem !important;
    opacity: 0.95;
    max-width: 720px;
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
    padding: 0.22rem 0.8rem;
    font-size: 0.70rem;
    font-weight: 700;
    letter-spacing: 0.6px;
    color: #FFFFFF !important;
    margin-bottom: 0.6rem;
    text-transform: uppercase;
}

/* ── Section Category Labels (Apple Health Small Caps) ──── */
.section-label {
    font-size: 0.74rem !important;
    font-weight: 700 !important;
    color: #8E8E93 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.8px !important;
    margin-top: 0.2rem !important;
    margin-bottom: 0.5rem !important;
    padding-left: 2px !important;
}

/* ── True Apple Liquid Glass Container ──────────────────── */
.apple-glass-card {
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.38) 0%, rgba(255, 255, 255, 0.22) 100%) !important;
    backdrop-filter: blur(28px) saturate(210%) !important;
    -webkit-backdrop-filter: blur(28px) saturate(210%) !important;
    border-radius: 20px !important;
    border: 1px solid rgba(255, 255, 255, 0.45) !important;
    border-top: 1px solid rgba(255, 255, 255, 0.75) !important;
    box-shadow: 0 10px 32px 0 rgba(0, 0, 0, 0.06), inset 0 1px 1px 0 rgba(255, 255, 255, 0.7) !important;
    padding: 1.25rem 1.45rem !important;
    margin-bottom: 0.9rem !important;
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.38) 0%, rgba(255, 255, 255, 0.22) 100%) !important;
    backdrop-filter: blur(28px) saturate(210%) !important;
    -webkit-backdrop-filter: blur(28px) saturate(210%) !important;
    border-radius: 20px !important;
    border: 1px solid rgba(255, 255, 255, 0.45) !important;
    border-top: 1px solid rgba(255, 255, 255, 0.75) !important;
    box-shadow: 0 10px 32px 0 rgba(0, 0, 0, 0.06), inset 0 1px 1px 0 rgba(255, 255, 255, 0.7) !important;
    padding: 1.2rem 1.4rem !important;
    margin-bottom: 0.9rem !important;
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
    margin-bottom: 0.15rem;
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
    font-size: 0.90rem;
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
   1. PIXEL-PERFECT SLIDER STYLING (MATCHING REFERENCE IMAGE)
   ══════════════════════════════════════════════════════════ */
div[data-testid="stSlider"] {
    padding-top: 4px !important;
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
    height: 40px !important;
    position: relative !important;
}

/* Track: Sleek 6px capsule with vivid Apple Blue and neutral gray */
div[data-testid="stSlider"] div[data-baseweb="slider"] > div {
    height: 6px !important;
    border-radius: 9999px !important;
    background-color: #D1D1D6 !important;
}
div[data-testid="stSlider"] div[data-baseweb="slider"] > div > div:first-child {
    height: 6px !important;
    border-radius: 9999px !important;
    background-color: #007AFF !important;
}
div[data-testid="stSlider"] div[class*="efbyxod5"],
div[data-testid="stSlider"] div[data-rac] > div[class*="efbyxod5"] {
    height: 6px !important;
    border-radius: 9999px !important;
    top: calc(50% - 3px) !important;
    width: 100% !important;
}

/* BaseWeb Track Progress: Vivid Apple Blue */
div[data-testid="stSlider"] div[data-baseweb="slider"] > div > div:first-child {
    height: 6px !important;
    border-radius: 9999px !important;
    background-color: #007AFF !important;
}

/* Slider Thumb (Dragging Pill Capsule: 38px x 24px, 13px radius) */
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

/* Remove default inner BaseWeb circle */
div[data-testid="stSlider"] div[role="slider"] > * {
    display: none !important;
}

/* Hover: Subtle scale bump */
div[data-testid="stSlider"] div[role="slider"]:hover {
    transform: scale(1.05) !important;
    cursor: grab !important;
}
div[data-testid="stSlider"] div[class*="efbyxod3"]:hover,
div[data-testid="stSlider"] div[data-rac][style*="position: absolute"]:hover {
    transform: translate(-50%, -50%) scale(1.05) !important;
    cursor: grab !important;
}

/* Active / Dragging: Smooth tactile expansion with blue aura */
div[data-testid="stSlider"] div[role="slider"]:active,
div[data-testid="stSlider"] div[role="slider"]:focus {
    transform: scale(1.10) !important;
    cursor: grabbing !important;
    box-shadow: 0 4px 14px rgba(0, 122, 255, 0.25) !important;
}
div[data-testid="stSlider"] div[class*="efbyxod3"]:active,
div[data-testid="stSlider"] div[class*="efbyxod3"]:focus,
div[data-testid="stSlider"] div[data-rac][style*="position: absolute"]:active,
div[data-testid="stSlider"] div[data-rac][style*="position: absolute"]:focus {
    transform: translate(-50%, -50%) scale(1.10) !important;
    cursor: grabbing !important;
    box-shadow: 0 4px 14px rgba(0, 122, 255, 0.25) !important;
}

/* Hide floating tooltip value above thumb */
div[data-testid="stThumbValue"],
div[data-testid="stSliderThumbValue"] {
    display: none !important;
}

div[data-testid="stSliderTickBarMin"],
div[data-testid="stSliderTickBarMax"],
div[data-testid="stSliderTickBar"] p {
    color: #8E8E93 !important;
    font-size: 0.74rem !important;
    font-weight: 500 !important;
}

/* Cross-Browser Native Range Rules (WebKit & Mozilla) */
input[type="range"] {
    -webkit-appearance: none;
    appearance: none;
    width: 100%;
    height: 6px;
    border-radius: 9999px;
    background: #D1D1D6;
    outline: none;
}
input[type="range"]::-webkit-slider-runnable-track {
    height: 6px;
    border-radius: 9999px;
    background: linear-gradient(to right, #007aff 0%, #007aff var(--progress, 50%), #d1d1d6 var(--progress, 50%), #d1d1d6 100%);
}
input[type="range"]::-moz-range-track {
    height: 6px;
    border-radius: 9999px;
    background: linear-gradient(to right, #007aff 0%, #007aff var(--progress, 50%), #d1d1d6 var(--progress, 50%), #d1d1d6 100%);
}
input[type="range"]::-webkit-slider-thumb {
    -webkit-appearance: none;
    appearance: none;
    width: 38px;
    height: 24px;
    border-radius: 13px;
    background: rgba(235, 235, 240, 0.70);
    backdrop-filter: blur(16px) saturate(180%);
    -webkit-backdrop-filter: blur(16px) saturate(180%);
    border: 1.5px solid rgba(255, 255, 255, 0.95);
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.12);
    cursor: grab;
    margin-top: -9px;
    transition: transform 0.15s cubic-bezier(0.16, 1, 0.3, 1), box-shadow 0.15s ease;
}
input[type="range"]::-webkit-slider-thumb:hover {
    transform: scale(1.05);
    cursor: grab;
}
input[type="range"]::-webkit-slider-thumb:active {
    transform: scale(1.10);
    cursor: grabbing;
    box-shadow: 0 4px 14px rgba(0, 122, 255, 0.25);
}
input[type="range"]::-moz-range-thumb {
    width: 38px;
    height: 24px;
    border-radius: 13px;
    background: rgba(235, 235, 240, 0.70);
    backdrop-filter: blur(16px) saturate(180%);
    border: 1.5px solid rgba(255, 255, 255, 0.95);
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.12);
    cursor: grab;
    transition: transform 0.15s cubic-bezier(0.16, 1, 0.3, 1), box-shadow 0.15s ease;
}
input[type="range"]::-moz-range-thumb:hover {
    transform: scale(1.05);
    cursor: grab;
}
input[type="range"]::-moz-range-thumb:active {
    transform: scale(1.10);
    cursor: grabbing;
    box-shadow: 0 4px 14px rgba(0, 122, 255, 0.25);
}

/* ── Synchronized Number Input Box ──────────────────────── */
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
    height: 40px !important;
}
div[data-testid="stNumberInput"] div[data-baseweb="input"]:focus-within {
    border-color: #007AFF !important;
    box-shadow: 0 0 0 3px rgba(0, 122, 255, 0.22), inset 0 1px 1px 0 rgba(255, 255, 255, 0.9) !important;
}
div[data-testid="stNumberInput"] input {
    font-weight: 700 !important;
    font-size: 0.94rem !important;
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

/* ══════════════════════════════════════════════════════════
   2. CLINICAL REFERENCE RANGE CARDS (PARALLEL BENCHMARKS)
   ══════════════════════════════════════════════════════════ */
.ref-card-row {
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.38) 0%, rgba(255, 255, 255, 0.22) 100%) !important;
    backdrop-filter: blur(28px) saturate(210%) !important;
    -webkit-backdrop-filter: blur(28px) saturate(210%) !important;
    border-radius: 16px !important;
    border: 1px solid rgba(255, 255, 255, 0.45) !important;
    border-top: 1px solid rgba(255, 255, 255, 0.75) !important;
    box-shadow: 0 8px 24px 0 rgba(0, 0, 0, 0.05), inset 0 1px 1px 0 rgba(255, 255, 255, 0.7) !important;
    padding: 0.90rem 1.1rem !important;
    margin-bottom: 0.78rem !important;
    display: flex;
    align-items: center;
    justify-content: space-between;
    transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
    min-height: 76px;
}
.ref-card-row:hover {
    box-shadow: 0 12px 32px 0 rgba(0, 0, 0, 0.08), inset 0 1px 1px rgba(255, 255, 255, 0.9) !important;
    transform: translateY(-1px);
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.45) 0%, rgba(255, 255, 255, 0.28) 100%) !important;
}
.ref-meta-left {
    display: flex;
    align-items: center;
    gap: 10px;
}
.ref-metric-name {
    font-size: 0.88rem !important;
    font-weight: 700 !important;
    color: #1C1C1E !important;
    line-height: 1.2;
}
.ref-metric-unit {
    font-size: 0.70rem !important;
    color: #8E8E93 !important;
    font-weight: 500;
}
.ref-badges-right {
    display: flex;
    flex-direction: column;
    align-items: flex-end;
    gap: 4px;
}
.range-badge {
    display: inline-block;
    padding: 0.20rem 0.65rem;
    border-radius: 999px;
    font-size: 0.70rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.2px;
}
.badge-normal {
    background: rgba(52, 199, 89, 0.16) !important;
    color: #248A3D !important;
}
.badge-danger {
    background: rgba(255, 59, 48, 0.16) !important;
    color: #D70015 !important;
}

/* ── Primary Action Button ──────────────────────────────── */
.stButton > button[kind="primary"] {
    background: #007AFF !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 999px !important;
    font-weight: 700 !important;
    font-size: 1.02rem !important;
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

/* Secondary Action Button (PDF Download) */
.stDownloadButton > button {
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.45) 0%, rgba(255, 255, 255, 0.25) 100%) !important;
    backdrop-filter: blur(28px) saturate(210%) !important;
    -webkit-backdrop-filter: blur(28px) saturate(210%) !important;
    border: 1px solid rgba(0, 122, 255, 0.35) !important;
    border-top: 1px solid rgba(255, 255, 255, 0.85) !important;
    color: #007AFF !important;
    border-radius: 999px !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    padding: 0.65rem 1.6rem !important;
    box-shadow: 0 4px 14px rgba(0, 122, 255, 0.12) !important;
    transition: all 0.18s ease !important;
}
.stDownloadButton > button:hover {
    background: #007AFF !important;
    color: #FFFFFF !important;
    border-color: #007AFF !important;
    box-shadow: 0 6px 20px rgba(0, 122, 255, 0.35) !important;
    transform: translateY(-1px) !important;
}

/* ── Risk Result Glass Panels ───────────────────────────── */
.risk-card {
    border-radius: 20px !important;
    padding: 1.5rem 1.6rem !important;
    margin: 0.6rem 0 !important;
    backdrop-filter: blur(28px) saturate(210%) !important;
    -webkit-backdrop-filter: blur(28px) saturate(210%) !important;
    border-top: 1px solid rgba(255, 255, 255, 0.75) !important;
    box-shadow: 0 10px 32px 0 rgba(0, 0, 0, 0.06), inset 0 1px 1px 0 rgba(255, 255, 255, 0.7) !important;
    transition: all 0.2s ease;
}
.risk-card-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 0.5rem;
}
.risk-card h3 {
    margin: 0 !important;
    font-size: 1.35rem !important;
    font-weight: 700 !important;
}
.risk-high {
    background: linear-gradient(135deg, rgba(255, 238, 238, 0.40) 0%, rgba(255, 238, 238, 0.22) 100%) !important;
    border: 1.5px solid rgba(255, 59, 48, 0.35) !important;
}
.risk-high h3 { color: #D70015 !important; }
.risk-low {
    background: linear-gradient(135deg, rgba(234, 252, 240, 0.40) 0%, rgba(234, 252, 240, 0.22) 100%) !important;
    border: 1.5px solid rgba(52, 199, 89, 0.35) !important;
}
.risk-low h3 { color: #248A3D !important; }
.risk-medium {
    background: linear-gradient(135deg, rgba(255, 248, 230, 0.40) 0%, rgba(255, 248, 230, 0.22) 100%) !important;
    border: 1.5px solid rgba(255, 149, 0, 0.35) !important;
}
.risk-medium h3 { color: #C93400 !important; }

/* ── Metric Summary Tiles ───────────────────────────────── */
.metric-tile {
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.38) 0%, rgba(255, 255, 255, 0.22) 100%) !important;
    backdrop-filter: blur(28px) saturate(210%) !important;
    -webkit-backdrop-filter: blur(28px) saturate(210%) !important;
    border-radius: 18px !important;
    border: 1px solid rgba(255, 255, 255, 0.45) !important;
    border-top: 1px solid rgba(255, 255, 255, 0.75) !important;
    padding: 1.1rem 1.2rem !important;
    text-align: center;
    box-shadow: 0 10px 32px 0 rgba(0, 0, 0, 0.06), inset 0 1px 1px 0 rgba(255, 255, 255, 0.7) !important;
}
.metric-tile .value {
    font-size: 1.9rem !important;
    font-weight: 800 !important;
    font-variant-numeric: tabular-nums;
    letter-spacing: -0.02em;
}
.metric-tile .label {
    font-size: 0.76rem !important;
    color: #8E8E93 !important;
    margin-top: 0.25rem;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.4px;
}

/* ── Regulatory Disclaimer Box ──────────────────────────── */
.disclaimer {
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.38) 0%, rgba(255, 255, 255, 0.22) 100%) !important;
    backdrop-filter: blur(28px) saturate(210%) !important;
    -webkit-backdrop-filter: blur(28px) saturate(210%) !important;
    border: 1px solid rgba(255, 255, 255, 0.45) !important;
    border-left: 5px solid #007AFF !important;
    border-top: 1px solid rgba(255, 255, 255, 0.75) !important;
    border-radius: 18px !important;
    padding: 1.1rem 1.4rem !important;
    font-size: 0.86rem !important;
    color: #3A3A3C !important;
    line-height: 1.5;
    margin-top: 1.2rem !important;
    box-shadow: 0 10px 32px 0 rgba(0, 0, 0, 0.05), inset 0 1px 1px 0 rgba(255, 255, 255, 0.7) !important;
}

/* ── Expanders & Accordions ─────────────────────────────── */
details {
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.38) 0%, rgba(255, 255, 255, 0.22) 100%) !important;
    backdrop-filter: blur(28px) saturate(210%) !important;
    -webkit-backdrop-filter: blur(28px) saturate(210%) !important;
    border-radius: 18px !important;
    border: 1px solid rgba(255, 255, 255, 0.45) !important;
    border-top: 1px solid rgba(255, 255, 255, 0.75) !important;
    box-shadow: 0 10px 32px 0 rgba(0, 0, 0, 0.04), inset 0 1px 1px 0 rgba(255, 255, 255, 0.7) !important;
    margin-top: 0.6rem !important;
    margin-bottom: 0.8rem !important;
    overflow: hidden;
}
summary {
    font-weight: 600 !important;
    color: #1C1C1E !important;
    padding: 0.8rem 1.1rem !important;
    cursor: pointer !important;
}

/* ── Developer Sidebar Drawer ───────────────────────────── */
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

div[data-testid="stMetric"] {
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.38) 0%, rgba(255, 255, 255, 0.22) 100%) !important;
    backdrop-filter: blur(28px) saturate(210%) !important;
    -webkit-backdrop-filter: blur(28px) saturate(210%) !important;
    border-radius: 14px !important;
    border: 1px solid rgba(255, 255, 255, 0.45) !important;
    border-top: 1px solid rgba(255, 255, 255, 0.75) !important;
    padding: 0.8rem 0.95rem !important;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.03), inset 0 1px 1px 0 rgba(255, 255, 255, 0.7) !important;
}

/* ── Print Media Optimization ───────────────────────────── */
@media print {
    header, footer, .ambient-orb-blue, .ambient-orb-cyan, .stButton, details, section[data-testid="stSidebar"], .stDownloadButton {
        display: none !important;
    }
    .block-container {
        max-width: 100% !important;
        padding: 0 !important;
    }
    .apple-glass-card, .risk-card, .metric-tile, .ref-card-row {
        box-shadow: none !important;
        border: 1px solid #D1D1D6 !important;
        background: #FFFFFF !important;
        break-inside: avoid;
    }
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  CLINICAL PDF REPORT BUILDER (ReportLab)
# ─────────────────────────────────────────────
def generate_clinical_pdf(patient_data, prediction, prob_diabetic, prob_non_diabetic, flags, patient_id="PT-2026-8841"):
    """
    Generate a print-ready clinical screening report adhering strictly to
    standard hospital/clinical documentation standards.
    """
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
    meta_data = [
        [
            Paragraph(f"<b>Patient Identifier:</b> {patient_id}", cell_regular),
            Paragraph(f"<b>Generated Timestamp:</b> {curr_time}", cell_regular),
        ],
        [
            Paragraph("<b>Diagnostic Model:</b> ML Random Forest Pipeline (8-Features)", cell_regular),
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
    is_diabetic = (prediction == 1) or (prob_diabetic >= 0.50)
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
    elements.append(Spacer(1, 10))
    
    # ── 4. Patient Input Profile & Benchmark Comparison ──
    elements.append(Paragraph("2. SUBMITTED CLINICAL PARAMETERS VS. BASELINE BENCHMARKS", section_heading))
    param_rows = [
        [Paragraph("Clinical Parameter", cell_bold), Paragraph("Submitted Value", cell_bold), Paragraph("Standard Unit", cell_bold), Paragraph("Baseline Target Norm", cell_bold)]
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
    for k, v in patient_data.items():
        unit, norm = ref_norms.get(k, ("", "—"))
        val_str = f"{v:.2f}" if isinstance(v, float) else f"{v}"
        param_rows.append([
            Paragraph(FEATURE_META.get(k, {}).get("label", k), cell_regular),
            Paragraph(f"<b>{val_str}</b>", cell_bold),
            Paragraph(unit, cell_regular),
            Paragraph(norm, cell_regular),
        ])
    
    param_table = Table(param_rows, colWidths=[160, 110, 120, 150])
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

CHART_BG = "#FFFFFF"
CHART_FG = "#1C1C1E"
GRID_CLR = "#E5E5EA"
PALETTE  = ["#007AFF", "#5856D6", "#34C759", "#FF9500", "#FF3B30", "#AF52DE"]

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
#  SYNCHRONIZED DUAL-INPUT STATE HANDLERS
# ─────────────────────────────────────────────
def sync_from_slider(feat):
    """Update canonical value and numeric box when slider thumb moves."""
    val = st.session_state[f"slider_{feat}"]
    st.session_state[f"input_{feat}"] = val
    st.session_state[f"num_{feat}"] = val

def sync_from_num(feat):
    """Update canonical value and slider thumb when numeric input changes."""
    val = st.session_state[f"num_{feat}"]
    meta = FEATURE_META[feat]
    clamped = max(meta["type"](meta["min"]), min(meta["type"](meta["max"]), val))
    st.session_state[f"input_{feat}"] = clamped
    st.session_state[f"slider_{feat}"] = clamped

def apply_preset(preset_name):
    """Apply clinical preset values to all session states."""
    if preset_name in PRESETS and PRESETS[preset_name] is not None:
        values = PRESETS[preset_name]
        for f, v in values.items():
            if f in FEATURE_META:
                cast_v = FEATURE_META[f]["type"](v)
                st.session_state[f"input_{f}"] = cast_v
                st.session_state[f"slider_{f}"] = cast_v
                st.session_state[f"num_{f}"] = cast_v

# ─────────────────────────────────────────────
#  MAIN APPLICATION
# ─────────────────────────────────────────────
def main():
    # Background Refraction Ambient Orbs
    st.markdown('<div class="ambient-orb-blue"></div><div class="ambient-orb-cyan"></div>', unsafe_allow_html=True)

    # Initialize predict run counter
    if "predict_counter" not in st.session_state:
        st.session_state.predict_counter = 0

    # Load data & machine learning model
    df = load_dataset()
    pipeline, feature_names, _ = load_model()

    # Initialize session states for all features
    for f in ALL_FEATURES:
        meta = FEATURE_META[f]
        if f"input_{f}" not in st.session_state:
            def_val = meta["type"](meta["default"])
            st.session_state[f"input_{f}"] = def_val
            st.session_state[f"slider_{f}"] = def_val
            st.session_state[f"num_{f}"] = def_val

    # ═══════════════════════════════════════════
    #  DEVELOPER DRAWER (SIDEBAR)
    # ═══════════════════════════════════════════
    with st.sidebar:
        st.markdown(f'''
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 0.5rem;">
            {render_badge("activity", "icon-badge-blue")}
            <h2 style="margin: 0; font-size: 1.25rem;">Developer Analytics</h2>
        </div>
        ''', unsafe_allow_html=True)
        st.markdown('<div class="section-label">MODEL TELEMETRY & HEALTH</div>', unsafe_allow_html=True)

        clf = pipeline.named_steps.get("classifier", None)
        clf_name = type(clf).__name__ if clf else "Supervised Pipeline"

        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.metric("Model Architecture", clf_name.replace("Classifier", ""))
        with m_col2:
            st.metric("Feature Dimensions", f"{pipeline.n_features_in_} Inputs")

        st.markdown("")
        metrics_df, preds_dict = compute_benchmark_metrics(df, feature_names)

        st.markdown('<div class="section-label">BENCHMARK EVALUATION (TEST SPLIT)</div>', unsafe_allow_html=True)
        st.dataframe(
            metrics_df.style.format({
                "Accuracy": "{:.3f}", "Precision": "{:.3f}", "Recall": "{:.3f}", "F1": "{:.3f}"
            }),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("")
        st.markdown('<div class="section-label">DIAGNOSTIC CHARTS</div>', unsafe_allow_html=True)

        chart_choice = st.selectbox(
            "Select Diagnostic Telemetry Visualizer",
            ["Model Performance Comparison", "Active Classifier Confusion Matrix", "Feature Correlation Heatmap", "Feature Distributions", "Class Balance"],
            label_visibility="collapsed"
        )

        if chart_choice == "Model Performance Comparison":
            fig_buf = plot_benchmark_bars(metrics_df)
            st.image(fig_buf, use_container_width=True)
        elif chart_choice == "Active Classifier Confusion Matrix":
            active_key = "Random Forest" if "Random Forest" in preds_dict else list(preds_dict.keys())[0]
            y_test, y_pred = preds_dict[active_key]
            fig_buf = plot_confusion_matrix(y_test, y_pred, title=f"Confusion Matrix ({active_key})")
            st.image(fig_buf, use_container_width=True)
        elif chart_choice == "Feature Correlation Heatmap":
            fig_buf = plot_correlation_heatmap(df)
            st.image(fig_buf, use_container_width=True)
        elif chart_choice == "Feature Distributions":
            fig_buf = plot_feature_distributions(df)
            st.image(fig_buf, use_container_width=True)
        elif chart_choice == "Class Balance":
            fig_buf = plot_outcome_distribution(df)
            st.image(fig_buf, use_container_width=True)

    # ═══════════════════════════════════════════
    #  MAIN CONTENT AREA: APPLE HEALTH HERO BANNER
    # ═══════════════════════════════════════════
    st.markdown(f'''
    <div class="hero-banner">
        <div class="hero-badge">
            {SVG_ICONS["activity"]}
            <span>Clinical Decision Support Telemetry</span>
        </div>
        <h1>AI Diabetes Risk Assessment</h1>
        <p>
            Evidence-based clinical risk stratification powered by supervised machine learning.
            Adjust patient biomarkers to analyze individualized metabolic risk and export formal clinical documentation.
        </p>
    </div>
    ''', unsafe_allow_html=True)

    # ── Clinical Preset Selector Row ──
    p_col_label, p_col_select = st.columns([1.6, 3.4])
    with p_col_label:
        st.markdown('<div class="section-label" style="padding-top: 8px;">CLINICAL PRESET PROFILE</div>', unsafe_allow_html=True)
    with p_col_select:
        selected_preset = st.selectbox(
            "Quick-Fill Clinical Preset",
            list(PRESETS.keys()),
            label_visibility="collapsed",
            key="preset_selector",
        )
        if selected_preset != "Custom Profile" and PRESETS[selected_preset] is not None:
            if st.session_state.get("last_applied_preset") != selected_preset:
                apply_preset(selected_preset)
                st.session_state["last_applied_preset"] = selected_preset

    st.markdown("")

    # ═══════════════════════════════════════════════════════════════════
    #  2. SIDE-BY-SIDE DUAL-COLUMN ARCHITECTURE (ZERO VERTICAL SCROLL)
    # ═══════════════════════════════════════════════════════════════════
    col_input_panel, col_ref_panel = st.columns([59, 41], gap="large")

    input_values = {}

    # ─────────────────────────────────────────────────────────────
    #  LEFT COLUMN (59%): PATIENT CLINICAL INPUT PANEL
    # ─────────────────────────────────────────────────────────────
    with col_input_panel:
        st.markdown('<div class="section-label">PATIENT CLINICAL INPUTS</div>', unsafe_allow_html=True)

        primary_feats   = [f for f in feature_names if FEATURE_META[f]["primary"]]
        secondary_feats = [f for f in feature_names if not FEATURE_META[f]["primary"]]

        # ── Primary Input Rows (Glass Container) ──
        with st.container(border=True):
            for feat in primary_feats:
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

                # Control row: Sleek Slider (72%) + Synchronized Number Input (28%)
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

            # ── Secondary Input Rows (Collapsible Glass Expander) ──
            if secondary_feats:
                with st.expander("Advanced Clinical Parameters (Click to expand)", expanded=False):
                    for feat in secondary_feats:
                        meta = FEATURE_META[feat]
                        unit_str = f"({meta['unit']})" if meta["unit"] else ""
                        
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
        <div class="ref-card-row" style="opacity: 0.92;">
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

        # Prepare model input payload
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

        st.markdown('<div class="section-label" style="margin-top: 1.4rem;">RISK STRATIFICATION & FINDINGS</div>', unsafe_allow_html=True)
        st.markdown("### Assessment Findings")

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
        if ins > 166:
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
