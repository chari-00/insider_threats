"""
Configuration file for Insider Threat Detection System
"""

import os
from pathlib import Path

# ============================================
# PATHS
# ============================================

# Root directory
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SYNTHETIC_DATA_DIR = DATA_DIR / "synthetic"
MODELS_DIR = ROOT_DIR / "models"

# Create directories if they don't exist
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, SYNTHETIC_DATA_DIR, MODELS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Data files
DATA_PATH = RAW_DATA_DIR / "lanl_auth_logs.txt"
SYNTHETIC_PATH = SYNTHETIC_DATA_DIR / "synthetic_logs.csv"
FEATURES_PATH = PROCESSED_DATA_DIR / "features.csv"
BASELINES_PATH = PROCESSED_DATA_DIR / "user_baselines.pkl"

# Model files
MODEL_PATH = MODELS_DIR / "isolation_forest.pkl"
SHAP_PATH = MODELS_DIR / "shap_explainer.pkl"

# ============================================
# DATA PROCESSING
# ============================================

# LANL dataset columns
LANL_COLUMNS = [
    'time',
    'source_user',
    'destination_user',
    'source_computer',
    'destination_computer',
    'auth_type',
    'logon_type',
    'auth_orientation',
    'success'
]

# Sample size for faster processing (set to None for full dataset)
LANL_SAMPLE_SIZE = 100000  # Use first 100k rows for demo

# Synthetic data generation
SYNTHETIC_NUM_USERS = 50
SYNTHETIC_NUM_LOGS = 10000
SYNTHETIC_ANOMALY_RATIO = 0.10  # 10% anomalies

# ============================================
# FEATURE ENGINEERING
# ============================================

# Time windows for rolling features (in days)
BASELINE_WINDOW_DAYS = 30
SHORT_WINDOW_DAYS = 7

# After-hours definition
AFTER_HOURS_START = 19  # 7 PM
AFTER_HOURS_END = 7     # 7 AM

# Sensitivity levels for resources
RESOURCE_SENSITIVITY_LEVELS = {
    'low': 1,
    'medium': 3,
    'high': 5
}

# ============================================
# MODEL CONFIGURATION
# ============================================

# Isolation Forest
ISOLATION_FOREST_PARAMS = {
    'n_estimators': 100,
    'contamination': 0.1,  # Expected anomaly ratio
    'random_state': 42,
    'n_jobs': -1  # Use all CPU cores
}

# One-Class SVM
ONE_CLASS_SVM_PARAMS = {
    'kernel': 'rbf',
    'gamma': 'auto',
    'nu': 0.1,  # Upper bound on fraction of outliers
}

# Risk level thresholds
RISK_THRESHOLDS = {
    'low': (0, 40),
    'medium': (40, 70),
    'high': (70, 100)
}

# ============================================
# GRAPH ANALYSIS
# ============================================

# Graph metrics
GRAPH_ANOMALY_THRESHOLD = 0.7  # Threshold for unusual connections
MIN_EDGE_WEIGHT = 2  # Minimum interactions to consider edge

# ============================================
# DASHBOARD
# ============================================

DASHBOARD_PORT = 8050
DASHBOARD_DEBUG = True

# Visualization colors
COLOR_SCHEME = {
    'low_risk': '#27ae60',      # Green
    'medium_risk': '#f39c12',   # Orange
    'high_risk': '#e74c3c',     # Red
    'background': '#ecf0f1',    # Light gray
    'text': '#2c3e50'           # Dark blue-gray
}

# ============================================
# LOGGING
# ============================================

LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
