"""
Anomaly detection models for insider threat detection
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import logging

from .config import (
    ISOLATION_FOREST_PARAMS, ONE_CLASS_SVM_PARAMS, RISK_THRESHOLDS,
    MODEL_PATH
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnomalyDetector:
    """Ensemble anomaly detection system"""
    
    def __init__(self):
        self.isolation_forest = None
        self.one_class_svm = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_trained = False
    
    def train(self, X, feature_names):
        """
        Train both anomaly detection models
        
        Args:
            X (pd.DataFrame or np.array): Feature matrix
            feature_names (list): List of feature names
        """
        logger.info("Training anomaly detection models...")
        
        self.feature_names = feature_names
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train Isolation Forest
        logger.info("Training Isolation Forest...")
        self.isolation_forest = IsolationForest(**ISOLATION_FOREST_PARAMS)
        self.isolation_forest.fit(X_scaled)
        
        # Train One-Class SVM (optional, for ensemble)
        logger.info("Training One-Class SVM...")
        self.one_class_svm = OneClassSVM(**ONE_CLASS_SVM_PARAMS)
        self.one_class_svm.fit(X_scaled)
        
        self.is_trained = True
        logger.info("✅ Model training complete")
    
    def predict(self, X):
        """
        Predict anomaly scores for new data
        
        Args:
            X (pd.DataFrame or np.array): Feature matrix
            
        Returns:
            np.array: Anomaly scores (0-100, higher = more anomalous)
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        X_scaled = self.scaler.transform(X)
        
        # Get scores from both models
        if_scores = self.isolation_forest.score_samples(X_scaled)
        svm_scores = self.one_class_svm.score_samples(X_scaled)
        
        # Normalize to 0-100 scale
        if_normalized = self._normalize_scores(if_scores)
        svm_normalized = self._normalize_scores(svm_scores)
        
        # Ensemble: weighted average
        ensemble_scores = 0.7 * if_normalized + 0.3 * svm_normalized
        
        return ensemble_scores
    
    def _normalize_scores(self, scores):
        """
        Normalize anomaly scores to 0-100 range
        Higher score = more anomalous
        
        Args:
            scores (np.array): Raw anomaly scores
            
        Returns:
            np.array: Normalized scores (0-100)
        """
        # Invert (original: lower = more anomalous)
        scores = -scores
        
        # Normalize to 0-100
        min_score = scores.min()
        max_score = scores.max()
        
        if max_score - min_score > 0:
            normalized = 100 * (scores - min_score) / (max_score - min_score)
        else:
            normalized = np.zeros_like(scores)
        
        return normalized
    
    def categorize_risk(self, scores):
        """
        Categorize risk scores into Low/Medium/High
        
        Args:
            scores (np.array): Anomaly scores (0-100)
            
        Returns:
            np.array: Risk levels ('Low', 'Medium', 'High')
        """
        risk_levels = np.empty(len(scores), dtype=object)
        
        for level, (min_val, max_val) in RISK_THRESHOLDS.items():
            mask = (scores >= min_val) & (scores < max_val)
            risk_levels[mask] = level.capitalize()
        
        # Handle edge case for exactly 100
        risk_levels[scores >= RISK_THRESHOLDS['high'][1]] = 'High'
        
        return risk_levels
    
    def save_models(self, if_path=MODEL_PATH, svm_path=None):
        """Save trained models to disk"""
        if not self.is_trained:
            raise ValueError("No trained models to save")
        
        logger.info(f"Saving models...")
        
        joblib.dump({
            'isolation_forest': self.isolation_forest,
            'one_class_svm': self.one_class_svm,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }, if_path)
        
        logger.info(f"✅ Models saved to {if_path}")
    
    def load_models(self, if_path=MODEL_PATH):
        """Load trained models from disk"""
        logger.info(f"Loading models from {if_path}...")
        
        data = joblib.load(if_path)
        
        self.isolation_forest = data['isolation_forest']
        self.one_class_svm = data['one_class_svm']
        self.scaler = data['scaler']
        self.feature_names = data['feature_names']
        self.is_trained = True
        
        logger.info(f"✅ Models loaded successfully")


def train_and_save_model(features_df):
    """
    Convenience function to train and save models
    
    Args:
        features_df (pd.DataFrame): User features with 'user_id' column
        
    Returns:
        AnomalyDetector: Trained detector
        pd.DataFrame: Features with anomaly scores
    """
    # Prepare feature matrix
    X = features_df.drop(['user_id'], axis=1, errors='ignore')
    X = X.select_dtypes(include=[np.number])
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    feature_names = list(X.columns)
    
    # Train detector
    detector = AnomalyDetector()
    detector.train(X, feature_names)
    
    # Add predictions to features_df
    scores = detector.predict(X)
    risk_levels = detector.categorize_risk(scores)
    
    features_df = features_df.copy()
    features_df['anomaly_score'] = scores
    features_df['risk_level'] = risk_levels
    features_df['is_anomaly'] = (risk_levels == 'High').astype(int)
    
    # Save models
    detector.save_models()
    
    return detector, features_df


from sklearn.ensemble import IsolationForest

def train_isolation_forest(features):
    """Train either sklearn or pure-python isolation forest"""
    clf = IsolationForest(
        n_estimators=100,
        contamination=0.1,
        max_features=1.0
    )
    
    clf.fit(features)
    return clf

def predict_anomalies(model, features):
    """Predict using either implementation"""
    predictions = model.predict(features)
    # isolation-forest uses 0 for inliers, 1 for outliers
    return predictions


if __name__ == "__main__":
    # Test anomaly detection
    from .data_loader import load_all_data
    from .feature_engineering import engineer_features_from_logs
    
    logger.info("Loading data...")
    df = load_all_data(use_lanl=False, use_synthetic=True)
    
    if df is not None:
        logger.info("Engineering features...")
        features_df = engineer_features_from_logs(df)
        
        logger.info("Training anomaly detector...")
        detector, features_with_scores = train_and_save_model(features_df)
        
        print("\n=== Anomaly Detection Results ===")
        print(f"Total users: {len(features_with_scores)}")
        print(f"\nRisk distribution:")
        print(features_with_scores['risk_level'].value_counts())
        
        print(f"\n🚨 High-risk users:")
        high_risk = features_with_scores[features_with_scores['risk_level'] == 'High']
        print(high_risk[['user_id', 'anomaly_score', 'after_hours_ratio', 'avg_sensitivity']].head(10))
        
        print(f"\n✅ Model saved to {MODEL_PATH}")