# 🛡️ Insider Threats Detection System

A comprehensive real-time cyber threat detection dashboard that monitors user behavior to identify insider threats and anomalous activities using advanced machine learning techniques.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Detection Methods](#detection-methods)
- [Data Sources](#data-sources)

## 🎯 Overview

The **Insider Threats Detection System** is an enterprise-grade security solution that provides real-time monitoring and analysis of user activities to detect potential insider threats. The system combines traditional rule-based detection with cutting-edge machine learning algorithms to identify anomalous behavior patterns that may indicate malicious insider activities.

### Key Capabilities

- **Real-time Monitoring**: Live dashboard with real-time threat detection
- **Multi-layered Detection**: 5 different detection algorithms working in parallel
- **User Risk Assessment**: Individual user risk scoring and blocking
- **Interactive Simulation**: Live simulation environment for testing
- **Explainable AI**: SHAP-based model explanations for transparency

## ✨ Features

### 🎮 Live Dashboard
- **Real-time Metrics**: Live event monitoring with instant updates
- **Interactive Controls**: Start/stop simulation with configurable parameters
- **Risk Visualization**: Dynamic charts showing risk distribution and trends
- **User Management**: Real-time user risk assessment and blocking

### 🔍 Advanced Detection
- **Number-based Detection**: Statistical anomaly detection
- **Pattern-based Detection**: Behavioral pattern analysis
- **Relationship-based Detection**: User-resource relationship analysis
- **VAE-based Detection**: Variational Auto-Encoder deep learning
- **Temporal Detection**: LSTM Auto-Encoder for time series analysis

### 📊 Analytics & Reporting
- **Risk Distribution Charts**: Visual representation of threat levels
- **Trend Analysis**: Historical risk score trends
- **User Risk Tables**: Detailed user risk assessments
- **Detection Type Analysis**: Breakdown of detection methods

## 🛠️ Tech Stack

### Frontend
- **Dash**: Python web framework for interactive dashboards
- **Plotly**: Interactive data visualization
- **Bootstrap**: Responsive UI components

### Backend & ML
- **Python 3.13**: Core programming language
- **TensorFlow + Keras**: Deep learning frameworks
- **Scikit-learn**: Traditional ML algorithms
- **Pandas + NumPy**: Data processing and analysis
- **NetworkX**: Graph analysis and relationship modeling

### Data & Visualization
- **SHAP**: Model explainability and interpretability
- **Plotly**: Interactive charts and graphs
- **Faker**: Synthetic data generation for testing

## 🚀 Installation

### Prerequisites
- Python 3.13+
- Windows/Linux/macOS

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/insider-threats.git
   cd insider-threats
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   python app_live.py
   ```

4. **Access the dashboard**
   Open your browser and navigate to: `http://localhost:8050`

### Detailed Installation

1. **Create virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**
   ```bash
   python -c "import dash, tensorflow, pandas; print('✅ All packages installed successfully!')"
   ```

## 📖 Usage

### Starting the Dashboard

```bash
python app_live.py
```

### Simulation Controls

1. **Start Simulation**: Click "Start Simulation" to begin live monitoring
2. **Configure Parameters**:
   - **Simulation Speed**: Adjust the speed of event generation (0.1x to 10x)
   - **Duration**: Set simulation duration in minutes
   - **Risk Threshold**: Configure blocking threshold (50-100)

3. **Monitor Results**: View real-time metrics, charts, and user risk assessments

### Key Dashboard Sections

- **📊 Real-time Metrics**: Live event counts, blocked users, risk scores
- **🔴 Live Event Stream**: Real-time event feed with risk indicators
- **📈 Risk Distribution**: Pie charts showing risk level distribution
- **📊 Risk Trends**: Time-series analysis of risk scores
- **👥 User Risk Assessment**: Detailed user risk tables with filtering
- **🔍 Detection Analysis**: Breakdown of detection method effectiveness

## 📁 Project Structure

```
insider-threats/
├── app_live.py                    # Main dashboard application
├── requirements.txt              # Python dependencies
├── README.md                     # Project documentation
├── .gitignore                   # Git ignore rules
├── assets/
│   └── style.css                # Dashboard styling
├── data/
│   ├── raw/                     # Raw LANL authentication logs
│   ├── processed/               # Processed features and baselines
│   └── synthetic/               # Generated synthetic data
├── models/                      # Trained ML models
└── src/                        # Source code
    ├── __init__.py
    ├── adaptive_ml_detector.py      # Traditional ML detection
    ├── anomaly_detector.py         # Anomaly detection models
    ├── config.py                   # Configuration settings
    ├── data_generator.py           # Synthetic data generation
    ├── data_loader.py              # Data loading utilities
    ├── enhanced_adaptive_ml_detector.py  # Enhanced ML detection
    ├── explainer.py                # SHAP model explanations
    ├── feature_engineering.py      # Feature extraction
    ├── graph_analyzer.py           # Graph-based analysis
    ├── live_simulator.py           # Live simulation engine
    ├── lstm_autoencoder.py         # LSTM temporal detection
    ├── user_blocking_system.py     # User blocking logic
    └── vae_anomaly_detector.py     # VAE-based detection
```

## 🔬 Detection Methods

### 1. Number-based Detection
- **Purpose**: Statistical anomaly detection
- **Method**: Isolation Forest, One-Class SVM
- **Features**: Access frequency, time patterns, resource usage

### 2. Pattern-based Detection
- **Purpose**: Behavioral pattern analysis
- **Method**: User behavior baselines, pattern matching
- **Features**: Login patterns, access sequences, time-based anomalies

### 3. Relationship-based Detection
- **Purpose**: User-resource relationship analysis
- **Method**: Graph analysis, relationship modeling
- **Features**: User-resource connections, access patterns

### 4. VAE-based Detection
- **Purpose**: Deep learning anomaly detection
- **Method**: Variational Auto-Encoder
- **Features**: Learned representations, reconstruction error

### 5. Temporal Detection
- **Purpose**: Time series anomaly detection
- **Method**: LSTM Auto-Encoder
- **Features**: Sequential patterns, temporal dependencies

## 📊 Data Sources

### Real Data
- **LANL Authentication Logs**: Los Alamos National Laboratory dataset
- **Format**: Authentication events with user, resource, and timestamp information
- **Size**: 100,000+ authentication events

### Synthetic Data
- **Generated Users**: 150+ simulated users with realistic behavior patterns
- **Departments**: IT, HR, Finance, Engineering, Marketing
- **Roles**: Admin, Manager, Employee, Contractor
- **Anomaly Ratio**: 10% anomalous activities

## 🔧 Configuration

### Key Settings (`src/config.py`)

```python
# Risk thresholds
RISK_THRESHOLDS = {
    'low': (0, 40),
    'medium': (40, 70),
    'high': (70, 100)
}

# Model parameters
ISOLATION_FOREST_PARAMS = {
    'n_estimators': 100,
    'contamination': 0.1,
    'random_state': 42
}

# Dashboard settings
DASHBOARD_PORT = 8050
DASHBOARD_DEBUG = True
```

## 📈 Performance Metrics

- **Accuracy**: 97.7% (Isolation Forest)
- **F1-Score**: 89.5% (Isolation Forest)
- **Precision**: 88.2% (Isolation Forest)
- **Recall**: 91.0% (Isolation Forest)


### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Run linting
flake8 src/
```


## 🙏 Acknowledgments

- **LANL Dataset**: Los Alamos National Laboratory for providing authentication logs
- **TensorFlow Team**: For the excellent deep learning framework
- **Scikit-learn Team**: For comprehensive ML algorithms
- **Dash Team**: For the powerful dashboard framework

---

**⚠️ Disclaimer**: This system is designed for educational and research purposes. Always ensure compliance with privacy laws and organizational policies when deploying in production environments.
