"""
Explainability module using SHAP for insider threat detection
"""

import pandas as pd
import numpy as np
import shap
import logging
import joblib

from .config import SHAP_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ThreatExplainer:
    """Generate explanations for anomaly detections using SHAP"""
    
    def __init__(self, model, feature_names):
        """
        Initialize explainer with trained model
        
        Args:
            model: Trained Isolation Forest model
            feature_names (list): List of feature names
        """
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        self.shap_values = None
    
    def build_explainer(self, X_train):
        """
        Build SHAP explainer (TreeExplainer for Isolation Forest)
        
        Args:
            X_train (np.array): Training data for background distribution
        """
        logger.info("Building SHAP explainer...")
        
        # Use TreeExplainer for tree-based models
        self.explainer = shap.TreeExplainer(self.model)
        
        logger.info("✅ SHAP explainer ready")
    
    def explain_user(self, user_features, user_id=None, top_n=5):
        """
        Generate explanation for a single user's anomaly score
        
        Args:
            user_features (np.array or pd.Series): User's feature vector
            user_id (str): User identifier for logging
            top_n (int): Number of top contributing features
            
        Returns:
            dict: Explanation with top features and natural language summary
        """
        if self.explainer is None:
            raise ValueError("Must call build_explainer() first")
        
        # Ensure 2D array
        if len(user_features.shape) == 1:
            user_features = user_features.reshape(1, -1)
        
        # Calculate SHAP values
        shap_values = self.explainer.shap_values(user_features)
        
        # Get feature contributions (absolute values for ranking)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        
        contributions = list(zip(self.feature_names, shap_values[0]))
        
        # Sort by absolute contribution
        contributions_sorted = sorted(contributions, key=lambda x: abs(x[1]), reverse=True)
        
        top_features = contributions_sorted[:top_n]
        
        # Generate natural language explanation
        explanation = self._generate_narrative(top_features, user_features[0], user_id)
        
        return {
            'user_id': user_id,
            'top_features': top_features,
            'explanation': explanation,
            'all_contributions': contributions_sorted,
        }
    
    def explain_batch(self, X, user_ids=None, top_n=3):
        """
        Generate explanations for multiple users
        
        Args:
            X (np.array): Feature matrix for multiple users
            user_ids (list): List of user IDs
            top_n (int): Number of top features per user
            
        Returns:
            list: List of explanation dicts
        """
        if self.explainer is None:
            raise ValueError("Must call build_explainer() first")
        
        logger.info(f"Generating explanations for {len(X)} users...")
        
        explanations = []
        
        for i, user_features in enumerate(X):
            user_id = user_ids[i] if user_ids else f"User_{i}"
            explanation = self.explain_user(user_features, user_id, top_n)
            explanations.append(explanation)
        
        return explanations
    
    def _generate_narrative(self, top_features, feature_values, user_id):
        """
        Generate natural language explanation
        
        Args:
            top_features (list): Top contributing features with SHAP values
            feature_values (np.array): Actual feature values
            user_id (str): User identifier
            
        Returns:
            str: Human-readable explanation
        """
        narrative_parts = []
        
        user_label = f"{user_id}" if user_id else "This user"
        
        for feature_name, shap_value in top_features[:3]:  # Top 3 only
            # Get feature index
            feature_idx = self.feature_names.index(feature_name)
            feature_value = feature_values[feature_idx]
            
            # Generate human-readable description
            description = self._describe_feature(feature_name, feature_value, shap_value)
            narrative_parts.append(description)
        
        # Combine into narrative
        if len(narrative_parts) == 0:
            return f"{user_label} shows standard behavioral patterns."
        elif len(narrative_parts) == 1:
            return f"{user_label} flagged due to: {narrative_parts[0]}"
        else:
            main_reasons = ", ".join(narrative_parts[:-1])
            return f"{user_label} flagged due to: {main_reasons}, and {narrative_parts[-1]}"
    
    def _describe_feature(self, feature_name, value, shap_value):
        """
        Generate human-readable description for a feature
        
        Args:
            feature_name (str): Feature name
            value (float): Feature value
            shap_value (float): SHAP contribution
            
        Returns:
            str: Human-readable description
        """
        # Determine if this increases or decreases risk
        direction = "increases" if shap_value > 0 else "decreases"
        
        # Feature-specific descriptions
        if 'after_hours' in feature_name.lower():
            return f"{value*100:.1f}% after-hours activity (baseline: ~15%)"
        
        elif 'sensitivity' in feature_name.lower() and 'avg' in feature_name.lower():
            return f"average resource sensitivity of {value:.2f}/5"
        
        elif 'failure' in feature_name.lower():
            return f"{value*100:.1f}% failed authentication attempts"
        
        elif 'delete' in feature_name.lower():
            return f"{value*100:.1f}% delete operations (unusual)"
        
        elif 'unique_ips' in feature_name.lower():
            return f"{int(value)} different IP addresses used"
        
        elif 'lateral_movement' in feature_name.lower():
            return f"{value*100:.1f}% lateral movement between systems"
        
        elif 'actions_per_day' in feature_name.lower():
            return f"{value:.1f} actions per day (high activity)"
        
        elif 'file_size' in feature_name.lower() and 'total' in feature_name.lower():
            return f"{value:.1f} MB total data accessed"
        
        elif 'new_resource' in feature_name.lower():
            return f"{value*100:.1f}% new resource exploration rate"
        
        elif 'graph_degree' in feature_name.lower():
            return f"{int(value)} resource connections in network"
        
        elif 'graph_avg_sensitivity' in feature_name.lower():
            return f"average network resource sensitivity: {value:.2f}/5"
        
        elif 'activity_change' in feature_name.lower():
            return f"{value:.1f}x activity increase vs baseline"
        
        elif 'weekend' in feature_name.lower():
            return f"{value*100:.1f}% weekend activity"
        
        else:
            # Generic description
            return f"{feature_name.replace('_', ' ')}: {value:.2f}"
    
    def save_explainer(self, file_path=SHAP_PATH):
        """Save SHAP explainer to disk"""
        if self.explainer is None:
            raise ValueError("No explainer to save")
        
        joblib.dump({
            'explainer': self.explainer,
            'feature_names': self.feature_names
        }, file_path)
        
        logger.info(f"✅ SHAP explainer saved to {file_path}")
    
    def load_explainer(self, file_path=SHAP_PATH):
        """Load SHAP explainer from disk"""
        data = joblib.load(file_path)
        
        self.explainer = data['explainer']
        self.feature_names = data['feature_names']
        
        logger.info(f"✅ SHAP explainer loaded from {file_path}")
    
def explain_predictions(model, features):
    # Implement SHAP explanation here
    pass


def generate_explanations(detector, X, user_ids, features_df=None):
    """
    Convenience function to generate explanations for users
    
    Args:
        detector: Trained AnomalyDetector
        X (np.array): Feature matrix
        user_ids (list): User IDs
        features_df (pd.DataFrame): Original features for reference
        
    Returns:
        list: Explanations for each user
    """
    explainer = ThreatExplainer(
        detector.isolation_forest,
        detector.feature_names
    )
    
    explainer.build_explainer(X)
    explanations = explainer.explain_batch(X, user_ids)
    
    return explanations


if __name__ == "__main__":
    # Test explainer
    from .data_loader import load_all_data
    from .feature_engineering import engineer_features_from_logs
    from .anomaly_detector import train_and_save_model
    
    logger.info("Loading data...")
    df = load_all_data(use_lanl=False, use_synthetic=True)
    
    if df is not None:
        logger.info("Engineering features...")
        features_df = engineer_features_from_logs(df)
        
        logger.info("Training model...")
        detector, features_with_scores = train_and_save_model(features_df)
        
        # Get high-risk users
        high_risk = features_with_scores[features_with_scores['risk_level'] == 'High']
        
        if len(high_risk) > 0:
            logger.info(f"Generating explanations for {len(high_risk)} high-risk users...")
            
            X = high_risk.drop(['user_id', 'anomaly_score', 'risk_level', 'is_anomaly'], axis=1)
            X = X.select_dtypes(include=[np.number])
            X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
            
            X_scaled = detector.scaler.transform(X)
            user_ids = high_risk['user_id'].tolist()
            
            explanations = generate_explanations(detector, X_scaled, user_ids, features_with_scores)
            
            print("\n=== Explanations for High-Risk Users ===")
            for exp in explanations[:3]:  # Show first 3
                print(f"\n🚨 {exp['user_id']}")
                print(f"Explanation: {exp['explanation']}")
                print(f"Top contributing features:")
                for feat, shap_val in exp['top_features']:
                    print(f"  - {feat}: {shap_val:.3f}")
