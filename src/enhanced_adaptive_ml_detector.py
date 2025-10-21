"""
Enhanced Adaptive ML-based Risk Detection System with Auto-Encoders
Implements 5 detection types: Number-based, Pattern-based, Relationship-based, VAE-based, and Temporal-based
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging
from collections import defaultdict, deque
import json
from pathlib import Path

from .config import RISK_THRESHOLDS
from .vae_anomaly_detector import VAEAnomalyDetector
from .lstm_autoencoder import TemporalAnomalyDetector
from .anomaly_detector import AnomalyDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EnhancedDetectionResult:
    """Enhanced result of adaptive ML detection with auto-encoders"""
    user_id: str
    risk_score: float
    risk_level: str
    detection_types: List[str]
    number_based_score: float
    pattern_based_score: float
    relationship_based_score: float
    vae_based_score: float
    temporal_based_score: float
    features: Dict[str, float]
    explanation: str
    timestamp: datetime
    should_block: bool = False


class EnhancedAdaptiveMLDetector:
    """
    Enhanced adaptive ML-based threat detection system that combines
    traditional rule-based detection with deep learning auto-encoders
    """
    
    def __init__(self, enable_vae: bool = True, enable_temporal: bool = True):
        # Traditional detection components
        self.user_baselines = {}  # Learned user behavior baselines
        self.user_history = defaultdict(deque)  # Recent user activity history
        self.global_patterns = {}  # Global behavioral patterns
        self.resource_relationships = defaultdict(set)  # User-resource relationships
        self.computer_relationships = defaultdict(set)  # User-computer relationships
        self.ip_relationships = defaultdict(set)  # User-IP relationships
        
        # Auto-encoder components
        self.enable_vae = enable_vae
        self.enable_temporal = enable_temporal
        self.vae_detector = None
        self.temporal_detector = None
        
        # Detection weights (enhanced with auto-encoders)
        self.weights = {
            'number_based': 0.25,
            'pattern_based': 0.20,
            'relationship_based': 0.15,
            'vae_based': 0.25,
            'temporal_based': 0.15
        }
        
        # Risk thresholds
        self.BLOCK_THRESHOLD = 80.0
        self.HIGH_RISK_THRESHOLD = 70.0
        self.MEDIUM_RISK_THRESHOLD = 40.0
        
        # History window
        self.HISTORY_WINDOW = 100
        
        # Feature collection for auto-encoders
        self.feature_matrix = []
        self.user_feature_history = defaultdict(list)
        
        logger.info(f"Enhanced ML Detector initialized: VAE={enable_vae}, Temporal={enable_temporal}")
    
    def initialize_auto_encoders(self, feature_dim: int = 15):
        """Initialize auto-encoder models"""
        
        if self.enable_vae:
            logger.info("Initializing VAE detector...")
            self.vae_detector = VAEAnomalyDetector(
                input_dim=feature_dim,
                latent_dim=max(4, feature_dim // 3),
                hidden_dims=[max(8, feature_dim // 2), max(6, feature_dim // 3)]
            )
        
        if self.enable_temporal:
            logger.info("Initializing Temporal detector...")
            self.temporal_detector = TemporalAnomalyDetector(
                sequence_length=20,
                feature_dim=feature_dim
            )
        
        logger.info("Auto-encoders initialized successfully")
    
    def initialize_detectors(self):
        """Initialize other detectors"""
        
        self.anomaly_detector = AnomalyDetector(contamination=0.1)
        
        logger.info("Detectors initialized successfully")
    
    def train_auto_encoders(self, training_data: np.ndarray = None):
        """Train auto-encoder models"""
        
        if training_data is None:
            # Generate training data from user history
            training_data = self._generate_training_data()
        
        if training_data is None or len(training_data) < 100:
            logger.warning("Insufficient training data for auto-encoders")
            return
        
        logger.info(f"Training auto-encoders with {len(training_data)} samples...")
        
        if self.enable_vae and self.vae_detector:
            try:
                self.vae_detector.train(training_data, epochs=50, batch_size=32, verbose=0)
                logger.info("VAE training completed")
            except Exception as e:
                logger.error(f"VAE training failed: {e}")
                self.enable_vae = False
        
        if self.enable_temporal and self.temporal_detector:
            try:
                self.temporal_detector.lstm_autoencoder.train(training_data, epochs=50, batch_size=32, verbose=0)
                logger.info("LSTM Auto-Encoder training completed")
            except Exception as e:
                logger.error(f"LSTM training failed: {e}")
                self.enable_temporal = False
    
    def _generate_training_data(self) -> Optional[np.ndarray]:
        """Generate training data from user history"""
        
        if not self.user_feature_history:
            return None
        
        # Collect features from all users
        all_features = []
        for user_features in self.user_feature_history.values():
            all_features.extend(user_features)
        
        if len(all_features) < 50:
            return None
        
        return np.array(all_features)
    
    def _extract_features_for_autoencoders(self, user_id: str, event_data: Dict) -> np.ndarray:
        """Extract features for auto-encoder models"""
        
        features = []
        
        # Basic event features
        features.extend([
            event_data.get('file_size_kb', 0),
            event_data.get('resource_sensitivity', 1),
            event_data.get('hour', 0),
            1.0 if event_data.get('is_after_hours', False) else 0.0,
            1.0 if event_data.get('is_weekend', False) else 0.0,
            1.0 if event_data.get('is_failure', False) else 0.0,
            1.0 if event_data.get('is_lateral_movement', False) else 0.0,
        ])
        
        # User baseline features
        if user_id in self.user_baselines:
            baseline = self.user_baselines[user_id]
            features.extend([
                baseline['avg_file_size'],
                baseline['failure_rate'],
                baseline['after_hours_rate'],
                baseline['weekend_rate'],
                baseline['sensitivity_preference'],
            len(self.ip_relationships[user_id]) / 5.0,  # Normalized
            len(self.resource_relationships[user_id]) / 50.0,  # Normalized
        ])
        
        # Pad or truncate to fixed size
        target_size = 15
        if len(features) < target_size:
            features.extend([0.0] * (target_size - len(features)))
        else:
            features = features[:target_size]
        
        return np.array(features)
    
    def update_user_baseline(self, user_id: str, event_data: Dict):
        """Update user behavioral baseline with new event data"""
        
        if user_id not in self.user_baselines:
            self.user_baselines[user_id] = {
                'total_events': 0,
                'avg_file_size': 0.0,
                'typical_hours': set(),
                'typical_resources': defaultdict(int),
                'typical_actions': defaultdict(int),
                'typical_computers': defaultdict(int),
                'typical_ips': defaultdict(int),
                'avg_actions_per_day': 0.0,
                'failure_rate': 0.0,
                'after_hours_rate': 0.0,
                'weekend_rate': 0.0,
                'sensitivity_preference': 0.0,
                'last_updated': datetime.now()
            }
        
        baseline = self.user_baselines[user_id]
        baseline['total_events'] += 1
        
        # Update file size average
        current_avg = baseline['avg_file_size']
        new_size = event_data.get('file_size_kb', 0)
        baseline['avg_file_size'] = (current_avg * (baseline['total_events'] - 1) + new_size) / baseline['total_events']
        
        # Update typical hours
        hour = event_data.get('hour', 0)
        baseline['typical_hours'].add(hour)
        
        # Update resource preferences
        resource = event_data.get('resource_id', '')
        baseline['typical_resources'][resource] += 1
        
        # Update action preferences
        action = event_data.get('action', '')
        baseline['typical_actions'][action] += 1
        
        # Update computer preferences
        computer = event_data.get('destination_computer', '')
        baseline['typical_computers'][computer] += 1
        
        # Update IP preferences
        ip = event_data.get('ip_address', '')
        baseline['typical_ips'][ip] += 1
        
        # Update rates
        total_events = baseline['total_events']
        baseline['failure_rate'] = (baseline['failure_rate'] * (total_events - 1) + 
                                   (1 if event_data.get('is_failure', False) else 0)) / total_events
        baseline['after_hours_rate'] = (baseline['after_hours_rate'] * (total_events - 1) + 
                                       (1 if event_data.get('is_after_hours', False) else 0)) / total_events
        baseline['weekend_rate'] = (baseline['weekend_rate'] * (total_events - 1) + 
                                   (1 if event_data.get('is_weekend', False) else 0)) / total_events
        
        # Update sensitivity preference
        sensitivity = event_data.get('resource_sensitivity', 1)
        baseline['sensitivity_preference'] = (baseline['sensitivity_preference'] * (total_events - 1) + 
                                            sensitivity) / total_events
        
        baseline['last_updated'] = datetime.now()
    
    def add_event_to_history(self, user_id: str, event_data: Dict):
        """Add event to user's recent history"""
        
        # Add to history
        self.user_history[user_id].append(event_data)
        
        # Maintain history window
        if len(self.user_history[user_id]) > self.HISTORY_WINDOW:
            self.user_history[user_id].popleft()
        
        # Update relationships
        resource = event_data.get('resource_id', '')
        computer = event_data.get('destination_computer', '')
        ip = event_data.get('ip_address', '')
        
        self.resource_relationships[user_id].add(resource)
        self.computer_relationships[user_id].add(computer)
        self.ip_relationships[user_id].add(ip)
        
        # Extract features for auto-encoders
        features = self._extract_features_for_autoencoders(user_id, event_data)
        self.user_feature_history[user_id].append(features)
        
        # Maintain feature history window
        if len(self.user_feature_history[user_id]) > self.HISTORY_WINDOW:
            self.user_feature_history[user_id].pop(0)
        
        # Add to temporal detector
        if self.enable_temporal and self.temporal_detector and hasattr(self.temporal_detector, 'add_user_event'):
            self.temporal_detector.add_user_event(user_id, features)
    
    def detect_vae_anomalies(self, user_id: str, event_data: Dict) -> Tuple[float, List[str]]:
        """Detect anomalies using VAE"""
        
        if not self.enable_vae or not self.vae_detector or not self.vae_detector.is_trained:
            return 0.0, []
        
        try:
            features = self._extract_features_for_autoencoders(user_id, event_data)
            features_batch = features.reshape(1, -1)
            
            scores = self.vae_detector.predict_anomaly_scores(features_batch)
            score = scores[0]
            
            detections = []
            if score > 70:
                detections.append("vae_high_reconstruction_error")
            elif score > 50:
                detections.append("vae_moderate_reconstruction_error")
            
            return score, detections
            
        except Exception as e:
            logger.error(f"VAE detection error: {e}")
            return 0.0, []
    
    def detect_temporal_anomalies(self, user_id: str, event_data: Dict) -> Tuple[float, List[str]]:
        """Detect temporal anomalies using LSTM Auto-Encoder"""
        
        if not self.enable_temporal or not self.temporal_detector or not hasattr(self.temporal_detector, 'detect_temporal_anomaly'):
            return 0.0, []
        
        try:
            score, is_anomaly = self.temporal_detector.detect_temporal_anomaly(user_id)
            
            detections = []
            if is_anomaly:
                detections.append("temporal_pattern_deviation")
            elif score > 50:
                detections.append("temporal_suspicious_pattern")
            
            return score, detections
            
        except Exception as e:
            logger.error(f"Temporal detection error: {e}")
            return 0.0, []
    
    def detect_number_based_anomalies(self, user_id: str, event_data: Dict) -> Tuple[float, List[str]]:
        """Number-based detection: Unusual file counts, login times, frequencies"""
        
        if user_id not in self.user_baselines:
            return 0.0, []
        
        baseline = self.user_baselines[user_id]
        risk_score = 0.0
        detections = []
        
        # 1. Unusual file size
        current_size = event_data.get('file_size_kb', 0)
        avg_size = baseline['avg_file_size']
        
        if avg_size > 0:
            size_ratio = current_size / avg_size
            if size_ratio > 5.0:
                risk_score += 20.0
                detections.append("unusually_large_file")
            elif size_ratio > 2.0:
                risk_score += 10.0
                detections.append("large_file_access")
        
        # 2. Unusual access time
        hour = event_data.get('hour', 0)
        if hour not in baseline['typical_hours']:
            risk_score += 15.0
            detections.append("unusual_access_time")
        
        # 3. High frequency access
        recent_events = len(self.user_history[user_id])
        if recent_events > 20:
            risk_score += 12.0
            detections.append("high_frequency_access")
        
        # 4. After hours access
        if event_data.get('is_after_hours', False) and baseline['after_hours_rate'] < 0.1:
            risk_score += 18.0
            detections.append("unusual_after_hours_access")
        
        # 5. Weekend access
        if event_data.get('is_weekend', False) and baseline['weekend_rate'] < 0.05:
            risk_score += 15.0
            detections.append("unusual_weekend_access")
        
        # 6. High failure rate
        current_failure_rate = baseline['failure_rate']
        if current_failure_rate > 0.2:
            risk_score += 10.0
            detections.append("high_failure_rate")
        
        return min(risk_score, 50.0), detections
    
    def detect_pattern_based_anomalies(self, user_id: str, event_data: Dict) -> Tuple[float, List[str]]:
        """Pattern-based detection: Behavior doesn't match learned patterns"""
        
        if user_id not in self.user_baselines:
            return 0.0, []
        
        baseline = self.user_baselines[user_id]
        risk_score = 0.0
        detections = []
        
        # 1. Unusual resource access
        resource = event_data.get('resource_id', '')
        if resource not in baseline['typical_resources']:
            risk_score += 25.0
            detections.append("unusual_resource_access")
        else:
            total_resource_access = sum(baseline['typical_resources'].values())
            resource_frequency = baseline['typical_resources'][resource] / total_resource_access
            if resource_frequency < 0.01:
                risk_score += 15.0
                detections.append("rare_resource_access")
        
        # 2. Unusual action
        action = event_data.get('action', '')
        if action not in baseline['typical_actions']:
            risk_score += 20.0
            detections.append("unusual_action")
        
        # 3. Unusual sensitivity level
        sensitivity = event_data.get('resource_sensitivity', 1)
        avg_sensitivity = baseline['sensitivity_preference']
        
        if sensitivity > avg_sensitivity + 2:
            risk_score += 18.0
            detections.append("unusual_high_sensitivity_access")
        
        # 4. Destructive actions
        if action in ['delete', 'copy'] and action not in baseline['typical_actions']:
            risk_score += 22.0
            detections.append("destructive_action")
        
        return min(risk_score, 50.0), detections
    
    def detect_relationship_based_anomalies(self, user_id: str, event_data: Dict) -> Tuple[float, List[str]]:
        """Relationship-based detection: Unusual connections to resources, computers, IPs"""
        
        risk_score = 0.0
        detections = []
        
        # 1. Lateral movement
        computer = event_data.get('destination_computer', '')
        user_computers = self.computer_relationships[user_id]
        
        if len(user_computers) > 10:
            risk_score += 20.0
            detections.append("excessive_computer_access")
        
        # 2. Multiple IP usage
        ip = event_data.get('ip_address', '')
        user_ips = self.ip_relationships[user_id]
        
        if len(user_ips) > 5:
            risk_score += 25.0
            detections.append("multiple_ip_usage")
        
        # 3. Unusual resource relationships
        resource = event_data.get('resource_id', '')
        user_resources = self.resource_relationships[user_id]
        
        if len(user_resources) > 50:
            risk_score += 15.0
            detections.append("excessive_resource_access")
        
        return min(risk_score, 50.0), detections
    
    def detect_anomaly(self, user_id: str, event_data: Dict) -> EnhancedDetectionResult:
        """Main detection function that combines all detection types"""
        
        # Update user baseline (adaptive learning)
        self.update_user_baseline(user_id, event_data)
        
        # Add to history
        self.add_event_to_history(user_id, event_data)
        
        # Run all detection types
        number_score, number_detections = self.detect_number_based_anomalies(user_id, event_data)
        pattern_score, pattern_detections = self.detect_pattern_based_anomalies(user_id, event_data)
        relationship_score, relationship_detections = self.detect_relationship_based_anomalies(user_id, event_data)
        vae_score, vae_detections = self.detect_vae_anomalies(user_id, event_data)
        temporal_score, temporal_detections = self.detect_temporal_anomalies(user_id, event_data)
        
        # Calculate weighted total score
        total_score = (
            number_score * self.weights['number_based'] +
            pattern_score * self.weights['pattern_based'] +
            relationship_score * self.weights['relationship_based'] +
            vae_score * self.weights['vae_based'] +
            temporal_score * self.weights['temporal_based']
        )
        
        # Combine all detections
        all_detections = (number_detections + pattern_detections + 
                         relationship_detections + vae_detections + temporal_detections)
        
        # Determine risk level
        if total_score >= self.HIGH_RISK_THRESHOLD:
            risk_level = "High"
        elif total_score >= self.MEDIUM_RISK_THRESHOLD:
            risk_level = "Medium"
        else:
            risk_level = "Low"
        
        # Determine if user should be blocked
        should_block = total_score >= self.BLOCK_THRESHOLD
        
        # Create explanation
        explanation = self._create_enhanced_explanation(
            total_score, number_score, pattern_score, relationship_score,
            vae_score, temporal_score, all_detections
        )
        
        # Extract features for ML model
        features = self._extract_features_for_autoencoders(user_id, event_data)
        features_dict = {f"feature_{i}": float(features[i]) for i in range(len(features))}
        
        return EnhancedDetectionResult(
            user_id=user_id,
            risk_score=total_score,
            risk_level=risk_level,
            detection_types=all_detections,
            number_based_score=number_score,
            pattern_based_score=pattern_score,
            relationship_based_score=relationship_score,
            vae_based_score=vae_score,
            temporal_based_score=temporal_score,
            features=features_dict,
            explanation=explanation,
            timestamp=datetime.now(),
            should_block=should_block
        )
    
    def _create_enhanced_explanation(self, total_score: float, number_score: float, 
                                   pattern_score: float, relationship_score: float,
                                   vae_score: float, temporal_score: float,
                                   detections: List[str]) -> str:
        """Create enhanced human-readable explanation"""
        
        if total_score < 20:
            return "Normal user behavior detected."
        
        explanations = []
        
        if number_score > 15:
            explanations.append(f"Unusual numerical patterns detected (score: {number_score:.1f})")
        
        if pattern_score > 15:
            explanations.append(f"Behavioral pattern deviations detected (score: {pattern_score:.1f})")
        
        if relationship_score > 15:
            explanations.append(f"Unusual relationship patterns detected (score: {relationship_score:.1f})")
        
        if vae_score > 15:
            explanations.append(f"Deep learning anomaly detected (score: {vae_score:.1f})")
        
        if temporal_score > 15:
            explanations.append(f"Temporal pattern anomaly detected (score: {temporal_score:.1f})")
        
        if detections:
            explanations.append(f"Specific anomalies: {', '.join(detections[:3])}")
        
        return ". ".join(explanations) + "."
    
    def get_enhanced_detection_statistics(self) -> Dict:
        """Get enhanced detection statistics"""
        
        total_users = len(self.user_baselines)
        total_events = sum(baseline['total_events'] for baseline in self.user_baselines.values())
        
        stats = {
            "total_users": total_users,
            "total_events": total_events,
            "avg_events_per_user": total_events / max(1, total_users),
            "detection_weights": self.weights,
            "risk_thresholds": {
                "block": self.BLOCK_THRESHOLD,
                "high": self.HIGH_RISK_THRESHOLD,
                "medium": self.MEDIUM_RISK_THRESHOLD
            },
            "auto_encoders": {
                "vae_enabled": self.enable_vae,
                "temporal_enabled": self.enable_temporal,
                "vae_trained": self.vae_detector.is_trained if self.vae_detector else False,
                "temporal_trained": self.temporal_detector.lstm_autoencoder.is_trained if self.temporal_detector else False
            }
        }
        
        return stats
    
    def save_enhanced_model(self, filepath: str):
        """Save the enhanced model including auto-encoders"""
        
        model_data = {
            'user_baselines': self.user_baselines,
            'user_history': {k: list(v) for k, v in self.user_history.items()},
            'resource_relationships': {k: list(v) for k, v in self.resource_relationships.items()},
            'computer_relationships': {k: list(v) for k, v in self.computer_relationships.items()},
            'ip_relationships': {k: list(v) for k, v in self.ip_relationships.items()},
            'weights': self.weights,
            'thresholds': {
                'block': self.BLOCK_THRESHOLD,
                'high': self.HIGH_RISK_THRESHOLD,
                'medium': self.MEDIUM_RISK_THRESHOLD
            },
            'enable_vae': self.enable_vae,
            'enable_temporal': self.enable_temporal
        }
        
        with open(filepath, 'w') as f:
            json.dump(model_data, f, indent=2, default=str)
        
        # Save auto-encoders separately
        if self.enable_vae and self.vae_detector and self.vae_detector.is_trained:
            vae_path = filepath.replace('.json', '_vae.pkl')
            self.vae_detector.save_model(vae_path)
        
        if self.enable_temporal and self.temporal_detector and self.temporal_detector.lstm_autoencoder.is_trained:
            temporal_path = filepath.replace('.json', '_temporal.pkl')
            self.temporal_detector.lstm_autoencoder.save_model(temporal_path)
        
        logger.info(f"💾 Enhanced model saved to {filepath}")


if __name__ == "__main__":
    # Test enhanced detector
    logger.info("Testing Enhanced Adaptive ML Detector...")
    
    detector = EnhancedAdaptiveMLDetector(enable_vae=True, enable_temporal=True)
    detector.initialize_auto_encoders()
    
    # Generate some training data
    np.random.seed(42)
    training_data = np.random.normal(0, 1, (200, 15))
    detector.train_auto_encoders(training_data)
    
    # Test detection
    test_event = {
        'user_id': 'TEST_USER',
        'resource_id': 'financial_data_001',
        'action': 'read',
        'file_size_kb': 5000,
        'resource_sensitivity': 5,
        'destination_computer': 'WS-001',
        'ip_address': '192.168.1.100',
        'hour': 22,
        'is_after_hours': True,
        'is_weekend': False,
        'is_failure': False,
        'is_lateral_movement': True
    }
    
    result = detector.detect_anomaly('TEST_USER', test_event)
    logger.info(f"Enhanced detection result: Risk {result.risk_score:.1f} ({result.risk_level})")
    logger.info(f"Detection breakdown: Number={result.number_based_score:.1f}, "
                f"Pattern={result.pattern_based_score:.1f}, Relationship={result.relationship_based_score:.1f}, "
                f"VAE={result.vae_based_score:.1f}, Temporal={result.temporal_based_score:.1f}")
    logger.info(f"Explanation: {result.explanation}")
    
    logger.info("✅ Enhanced detector test completed successfully!")
