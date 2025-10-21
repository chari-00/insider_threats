"""
Feature engineering for insider threat detection
"""

import pandas as pd
import numpy as np
from datetime import timedelta
import logging
from tqdm import tqdm

from .config import BASELINE_WINDOW_DAYS, SHORT_WINDOW_DAYS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Extract behavioral features from authentication logs"""
    
    def __init__(self, df):
        """
        Initialize with authentication logs
        
        Args:
            df (pd.DataFrame): Authentication logs with columns:
                - timestamp, user_id, action, is_after_hours, etc.
        """
        self.df = df.copy()
        self.features_df = None
    
    def engineer_features(self):
        """
        Extract all behavioral features for each user
        
        Returns:
            pd.DataFrame: User-level features
        """
        logger.info("Engineering features for all users...")
        
        features_list = []
        
        for user in tqdm(self.df['user_id'].unique(), desc="Processing users"):
            user_logs = self.df[self.df['user_id'] == user].copy()
            
            features = self._extract_user_features(user, user_logs)
            features_list.append(features)
        
        self.features_df = pd.DataFrame(features_list)
        
        logger.info(f"Feature engineering complete. Shape: {self.features_df.shape}")
        logger.info(f"Features: {list(self.features_df.columns)}")
        
        return self.features_df
    
    def _extract_user_features(self, user_id, user_logs):
        """
        Extract comprehensive features for a single user
        
        Args:
            user_id (str): User identifier
            user_logs (pd.DataFrame): Logs for this user only
            
        Returns:
            dict: Dictionary of features
        """
        features = {'user_id': user_id}
        
        # ============================================
        # BASIC ACTIVITY FEATURES
        # ============================================
        
        features['total_actions'] = len(user_logs)
        features['unique_resources'] = user_logs['resource_id'].nunique() if 'resource_id' in user_logs else 0
        features['unique_computers'] = user_logs['destination_computer'].nunique() if 'destination_computer' in user_logs else 0
        features['unique_ips'] = user_logs['ip_address'].nunique() if 'ip_address' in user_logs else 0
        
        # Time-based features
        if len(user_logs) > 1:
            time_diff = (user_logs['timestamp'].max() - user_logs['timestamp'].min()).days + 1
            features['days_active'] = time_diff
            features['actions_per_day'] = len(user_logs) / max(time_diff, 1)
        else:
            features['days_active'] = 1
            features['actions_per_day'] = len(user_logs)
        
        # ============================================
        # TEMPORAL BEHAVIOR
        # ============================================
        
        features['after_hours_ratio'] = user_logs['is_after_hours'].mean() if 'is_after_hours' in user_logs else 0
        features['weekend_ratio'] = user_logs['is_weekend'].mean() if 'is_weekend' in user_logs else 0
        
        # Hour distribution entropy (randomness in access times)
        if 'hour' in user_logs.columns:
            hour_counts = user_logs['hour'].value_counts(normalize=True)
            features['hour_entropy'] = -np.sum(hour_counts * np.log2(hour_counts + 1e-10))
        else:
            features['hour_entropy'] = 0
        
        # ============================================
        # ACTION-BASED FEATURES
        # ============================================
        
        if 'action' in user_logs.columns:
            action_counts = user_logs['action'].value_counts(normalize=True)
            features['read_ratio'] = action_counts.get('read', 0)
            features['write_ratio'] = action_counts.get('write', 0)
            features['delete_ratio'] = action_counts.get('delete', 0)
            features['copy_ratio'] = action_counts.get('copy', 0)
        else:
            features['read_ratio'] = features['write_ratio'] = 0
            features['delete_ratio'] = features['copy_ratio'] = 0
        
        # ============================================
        # SECURITY-RELATED FEATURES
        # ============================================
        
        features['failure_ratio'] = user_logs['is_failure'].mean() if 'is_failure' in user_logs else 0
        features['lateral_movement_ratio'] = user_logs['is_lateral_movement'].mean() if 'is_lateral_movement' in user_logs else 0
        
        # Resource sensitivity
        if 'resource_sensitivity' in user_logs.columns:
            features['avg_sensitivity'] = user_logs['resource_sensitivity'].mean()
            features['max_sensitivity'] = user_logs['resource_sensitivity'].max()
            features['high_sensitivity_ratio'] = (user_logs['resource_sensitivity'] >= 4).mean()
        else:
            features['avg_sensitivity'] = features['max_sensitivity'] = 0
            features['high_sensitivity_ratio'] = 0
        
        # File size (data exfiltration indicator)
        if 'file_size_kb' in user_logs.columns:
            features['avg_file_size'] = user_logs['file_size_kb'].mean()
            features['max_file_size'] = user_logs['file_size_kb'].max()
            features['total_data_mb'] = user_logs['file_size_kb'].sum() / 1024
        else:
            features['avg_file_size'] = features['max_file_size'] = 0
            features['total_data_mb'] = 0
        
        # ============================================
        # BEHAVIORAL CONSISTENCY FEATURES
        # ============================================
        
        # Standard deviation of actions per day (consistency)
        if len(user_logs) > 7:
            daily_actions = user_logs.groupby(user_logs['timestamp'].dt.date).size()
            features['action_consistency'] = daily_actions.std() / (daily_actions.mean() + 1)
        else:
            features['action_consistency'] = 0
        
        # New resource access rate (exploring new systems)
        if len(user_logs) > 10 and 'resource_id' in user_logs:
            sorted_logs = user_logs.sort_values('timestamp')
            seen_resources = set()
            new_resource_count = 0
            
            for resource in sorted_logs['resource_id']:
                if resource not in seen_resources:
                    new_resource_count += 1
                    seen_resources.add(resource)
            
            features['new_resource_rate'] = new_resource_count / len(user_logs)
        else:
            features['new_resource_rate'] = 0
        
        # ============================================
        # ROLLING WINDOW FEATURES (Last 7 days vs baseline)
        # ============================================
        
        if len(user_logs) > 14:
            recent_cutoff = user_logs['timestamp'].max() - timedelta(days=SHORT_WINDOW_DAYS)
            recent_logs = user_logs[user_logs['timestamp'] >= recent_cutoff]
            baseline_logs = user_logs[user_logs['timestamp'] < recent_cutoff]
            
            if len(baseline_logs) > 0 and len(recent_logs) > 0:
                # Actions per day comparison
                baseline_actions_per_day = len(baseline_logs) / BASELINE_WINDOW_DAYS
                recent_actions_per_day = len(recent_logs) / SHORT_WINDOW_DAYS
                features['activity_change_ratio'] = recent_actions_per_day / (baseline_actions_per_day + 1)
                
                # After-hours change
                baseline_after_hours = baseline_logs['is_after_hours'].mean() if 'is_after_hours' in baseline_logs else 0
                recent_after_hours = recent_logs['is_after_hours'].mean() if 'is_after_hours' in recent_logs else 0
                features['after_hours_change'] = recent_after_hours - baseline_after_hours
                
                # Sensitivity change
                if 'resource_sensitivity' in baseline_logs.columns:
                    baseline_sensitivity = baseline_logs['resource_sensitivity'].mean()
                    recent_sensitivity = recent_logs['resource_sensitivity'].mean()
                    features['sensitivity_change'] = recent_sensitivity - baseline_sensitivity
                else:
                    features['sensitivity_change'] = 0
            else:
                features['activity_change_ratio'] = 1.0
                features['after_hours_change'] = 0
                features['sensitivity_change'] = 0
        else:
            features['activity_change_ratio'] = 1.0
            features['after_hours_change'] = 0
            features['sensitivity_change'] = 0
        
        return features
    
    def get_feature_matrix(self):
        """
        Get feature matrix for ML (without user_id)
        
        Returns:
            pd.DataFrame: Feature matrix
            list: Feature names
        """
        if self.features_df is None:
            raise ValueError("Must call engineer_features() first")
        
        # Drop user_id and any non-numeric columns
        X = self.features_df.drop(['user_id'], axis=1, errors='ignore')
        
        # Handle any remaining non-numeric columns
        X = X.select_dtypes(include=[np.number])
        
        # Replace inf/nan with 0
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        return X, list(X.columns)


def engineer_features_from_logs(df):
    """
    Convenience function to engineer features from logs
    
    Args:
        df (pd.DataFrame): Authentication logs
        
    Returns:
        pd.DataFrame: User-level features
    """
    engineer = FeatureEngineer(df)
    return engineer.engineer_features()


def extract_features(df):
    # Implement feature extraction logic here
    pass


if __name__ == "__main__":
    # Test feature engineering
    from .data_loader import load_all_data
    
    logger.info("Loading data...")
    df = load_all_data(use_lanl=True, use_synthetic=True)
    
    if df is not None:
        logger.info("Engineering features...")
        engineer = FeatureEngineer(df)
        features_df = engineer.engineer_features()
        
        print("\n=== Feature Summary ===")
        print(f"Users: {len(features_df)}")
        print(f"Features: {len(features_df.columns) - 1}")  # Exclude user_id
        print(f"\nFeature columns:")
        print(list(features_df.columns))
        print(f"\nSample features:")
        print(features_df.head())
        
        # Show statistics
        print(f"\n=== Feature Statistics ===")
        print(features_df.describe())