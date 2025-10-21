"""
Adaptive ML-based Risk Detection System
Implements 3 detection types: Number-based, Pattern-based, and Relationship-based
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging
from collections import defaultdict, deque
import json

from .config import RISK_THRESHOLDS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """Result of adaptive ML detection"""
    user_id: str
    risk_score: float
    risk_level: str
    detection_types: List[str]
    number_based_score: float
    pattern_based_score: float
    relationship_based_score: float
    features: Dict[str, float]
    explanation: str
    timestamp: datetime
    should_block: bool = False


class AdaptiveMLDetector:
    """
    Adaptive ML-based threat detection system that learns user behavior
    and detects anomalies using 3 detection types
    """
    
    def __init__(self):
        self.user_baselines = {}  # Learned user behavior baselines
        self.user_history = defaultdict(deque)  # Recent user activity history
        self.global_patterns = {}  # Global behavioral patterns
        self.resource_relationships = defaultdict(set)  # User-resource relationships
        self.computer_relationships = defaultdict(set)  # User-computer relationships
        self.ip_relationships = defaultdict(set)  # User-IP relationships
        
        # Detection weights (can be adjusted based on performance)
        self.weights = {
            'number_based': 0.4,
            'pattern_based': 0.35,
            'relationship_based': 0.25
        }
        
        # Risk thresholds
        self.BLOCK_THRESHOLD = 80.0
        self.HIGH_RISK_THRESHOLD = 70.0
        self.MEDIUM_RISK_THRESHOLD = 40.0
        
        # History window (number of events to keep per user)
        self.HISTORY_WINDOW = 100
        
    def update_user_baseline(self, user_id: str, event_data: Dict):
        """
        Update user behavioral baseline with new event data
        This is the adaptive learning component
        """
        
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
    
    def detect_number_based_anomalies(self, user_id: str, event_data: Dict) -> Tuple[float, List[str]]:
        """
        Number-based detection: Unusual file counts, login times, frequencies
        """
        
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
            if size_ratio > 5.0:  # 5x larger than usual
                risk_score += 20.0
                detections.append("unusually_large_file")
            elif size_ratio > 2.0:  # 2x larger than usual
                risk_score += 10.0
                detections.append("large_file_access")
        
        # 2. Unusual access time
        hour = event_data.get('hour', 0)
        if hour not in baseline['typical_hours']:
            risk_score += 15.0
            detections.append("unusual_access_time")
        
        # 3. High frequency access (burst activity)
        recent_events = len(self.user_history[user_id])
        if recent_events > 20:  # More than 20 events in recent history
            risk_score += 12.0
            detections.append("high_frequency_access")
        
        # 4. After hours access (if not typical)
        if event_data.get('is_after_hours', False) and baseline['after_hours_rate'] < 0.1:
            risk_score += 18.0
            detections.append("unusual_after_hours_access")
        
        # 5. Weekend access (if not typical)
        if event_data.get('is_weekend', False) and baseline['weekend_rate'] < 0.05:
            risk_score += 15.0
            detections.append("unusual_weekend_access")
        
        # 6. High failure rate
        current_failure_rate = baseline['failure_rate']
        if current_failure_rate > 0.2:  # More than 20% failures
            risk_score += 10.0
            detections.append("high_failure_rate")
        
        return min(risk_score, 50.0), detections  # Cap at 50 for this component
    
    def detect_pattern_based_anomalies(self, user_id: str, event_data: Dict) -> Tuple[float, List[str]]:
        """
        Pattern-based detection: Behavior doesn't match learned patterns
        """
        
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
            # Check if accessing resource much more than usual
            total_resource_access = sum(baseline['typical_resources'].values())
            resource_frequency = baseline['typical_resources'][resource] / total_resource_access
            if resource_frequency < 0.01:  # Less than 1% of total access
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
        
        if sensitivity > avg_sensitivity + 2:  # Much higher sensitivity
            risk_score += 18.0
            detections.append("unusual_high_sensitivity_access")
        
        # 4. Destructive actions (delete, copy)
        if action in ['delete', 'copy'] and action not in baseline['typical_actions']:
            risk_score += 22.0
            detections.append("destructive_action")
        
        # 5. Pattern deviation in recent behavior
        recent_events = list(self.user_history[user_id])
        if len(recent_events) >= 10:
            # Check if recent behavior is very different from baseline
            recent_resources = set(e.get('resource_id', '') for e in recent_events[-10:])
            typical_resources = set(baseline['typical_resources'].keys())
            
            overlap = len(recent_resources.intersection(typical_resources))
            if overlap < len(recent_resources) * 0.3:  # Less than 30% overlap
                risk_score += 15.0
                detections.append("behavioral_pattern_deviation")
        
        return min(risk_score, 50.0), detections  # Cap at 50 for this component
    
    def detect_relationship_based_anomalies(self, user_id: str, event_data: Dict) -> Tuple[float, List[str]]:
        """
        Relationship-based detection: Unusual connections to resources, computers, IPs
        """
        
        risk_score = 0.0
        detections = []
        
        # 1. Lateral movement (accessing many different computers)
        computer = event_data.get('destination_computer', '')
        user_computers = self.computer_relationships[user_id]
        
        if len(user_computers) > 10:  # Accessing too many computers
            risk_score += 20.0
            detections.append("excessive_computer_access")
        
        # 2. Multiple IP usage (potential account sharing or compromised account)
        ip = event_data.get('ip_address', '')
        user_ips = self.ip_relationships[user_id]
        
        if len(user_ips) > 5:  # Using too many IP addresses
            risk_score += 25.0
            detections.append("multiple_ip_usage")
        
        # 3. Unusual resource relationships
        resource = event_data.get('resource_id', '')
        user_resources = self.resource_relationships[user_id]
        
        if len(user_resources) > 50:  # Accessing too many different resources
            risk_score += 15.0
            detections.append("excessive_resource_access")
        
        # 4. Rapid relationship expansion
        recent_events = list(self.user_history[user_id])
        if len(recent_events) >= 20:
            # Check if user is rapidly expanding their relationships
            recent_computers = set(e.get('destination_computer', '') for e in recent_events[-20:])
            recent_ips = set(e.get('ip_address', '') for e in recent_events[-20:])
            recent_resources = set(e.get('resource_id', '') for e in recent_events[-20:])
            
            if len(recent_computers) > 8:  # Many new computers in short time
                risk_score += 18.0
                detections.append("rapid_computer_expansion")
            
            if len(recent_ips) > 4:  # Many new IPs in short time
                risk_score += 20.0
                detections.append("rapid_ip_expansion")
            
            if len(recent_resources) > 15:  # Many new resources in short time
                risk_score += 12.0
                detections.append("rapid_resource_expansion")
        
        # 5. Unusual geographic patterns (if IP geolocation available)
        # This would require IP geolocation service
        # For now, we'll use IP diversity as a proxy
        
        # 6. Privilege escalation patterns
        if user_id in self.user_baselines:
            baseline = self.user_baselines[user_id]
            current_sensitivity = event_data.get('resource_sensitivity', 1)
            avg_sensitivity = baseline['sensitivity_preference']
            
            if current_sensitivity > avg_sensitivity + 1.5:
                risk_score += 16.0
                detections.append("privilege_escalation")
        
        return min(risk_score, 50.0), detections  # Cap at 50 for this component
    
    def detect_anomaly(self, user_id: str, event_data: Dict) -> DetectionResult:
        """
        Main detection function that combines all 3 detection types
        """
        
        # Update user baseline (adaptive learning)
        self.update_user_baseline(user_id, event_data)
        
        # Add to history
        self.add_event_to_history(user_id, event_data)
        
        # Run all detection types
        number_score, number_detections = self.detect_number_based_anomalies(user_id, event_data)
        pattern_score, pattern_detections = self.detect_pattern_based_anomalies(user_id, event_data)
        relationship_score, relationship_detections = self.detect_relationship_based_anomalies(user_id, event_data)
        
        # Calculate weighted total score
        total_score = (
            number_score * self.weights['number_based'] +
            pattern_score * self.weights['pattern_based'] +
            relationship_score * self.weights['relationship_based']
        )
        
        # Combine all detections
        all_detections = number_detections + pattern_detections + relationship_detections
        
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
        explanation = self._create_explanation(
            total_score, number_score, pattern_score, relationship_score,
            all_detections
        )
        
        # Extract features for ML model
        features = self._extract_features(user_id, event_data)
        
        return DetectionResult(
            user_id=user_id,
            risk_score=total_score,
            risk_level=risk_level,
            detection_types=all_detections,
            number_based_score=number_score,
            pattern_based_score=pattern_score,
            relationship_based_score=relationship_score,
            features=features,
            explanation=explanation,
            timestamp=datetime.now(),
            should_block=should_block
        )
    
    def _create_explanation(self, total_score: float, number_score: float, 
                          pattern_score: float, relationship_score: float,
                          detections: List[str]) -> str:
        """Create human-readable explanation of the detection"""
        
        if total_score < 20:
            return "Normal user behavior detected."
        
        explanations = []
        
        if number_score > 15:
            explanations.append(f"Unusual numerical patterns detected (score: {number_score:.1f})")
        
        if pattern_score > 15:
            explanations.append(f"Behavioral pattern deviations detected (score: {pattern_score:.1f})")
        
        if relationship_score > 15:
            explanations.append(f"Unusual relationship patterns detected (score: {relationship_score:.1f})")
        
        if detections:
            explanations.append(f"Specific anomalies: {', '.join(detections[:3])}")
        
        return ". ".join(explanations) + "."
    
    def _extract_features(self, user_id: str, event_data: Dict) -> Dict[str, float]:
        """Extract features for ML model training"""
        
        features = {}
        
        # Basic event features
        features['file_size_kb'] = event_data.get('file_size_kb', 0)
        features['resource_sensitivity'] = event_data.get('resource_sensitivity', 1)
        features['hour'] = event_data.get('hour', 0)
        features['is_after_hours'] = 1.0 if event_data.get('is_after_hours', False) else 0.0
        features['is_weekend'] = 1.0 if event_data.get('is_weekend', False) else 0.0
        features['is_failure'] = 1.0 if event_data.get('is_failure', False) else 0.0
        features['is_lateral_movement'] = 1.0 if event_data.get('is_lateral_movement', False) else 0.0
        
        # User baseline features
        if user_id in self.user_baselines:
            baseline = self.user_baselines[user_id]
            features['baseline_avg_file_size'] = baseline['avg_file_size']
            features['baseline_failure_rate'] = baseline['failure_rate']
            features['baseline_after_hours_rate'] = baseline['after_hours_rate']
            features['baseline_weekend_rate'] = baseline['weekend_rate']
            features['baseline_sensitivity_preference'] = baseline['sensitivity_preference']
            features['total_historical_events'] = baseline['total_events']
        else:
            features['baseline_avg_file_size'] = 0.0
            features['baseline_failure_rate'] = 0.0
            features['baseline_after_hours_rate'] = 0.0
            features['baseline_weekend_rate'] = 0.0
            features['baseline_sensitivity_preference'] = 1.0
            features['total_historical_events'] = 0
        
        # Relationship features
        features['unique_computers'] = len(self.computer_relationships[user_id])
        features['unique_ips'] = len(self.ip_relationships[user_id])
        features['unique_resources'] = len(self.resource_relationships[user_id])
        
        # Recent activity features
        recent_events = list(self.user_history[user_id])
        features['recent_events_count'] = len(recent_events)
        
        if recent_events:
            recent_sizes = [e.get('file_size_kb', 0) for e in recent_events]
            features['recent_avg_file_size'] = np.mean(recent_sizes)
            features['recent_max_file_size'] = np.max(recent_sizes)
            
            recent_sensitivities = [e.get('resource_sensitivity', 1) for e in recent_events]
            features['recent_avg_sensitivity'] = np.mean(recent_sensitivities)
            features['recent_max_sensitivity'] = np.max(recent_sensitivities)
        else:
            features['recent_avg_file_size'] = 0.0
            features['recent_max_file_size'] = 0.0
            features['recent_avg_sensitivity'] = 1.0
            features['recent_max_sensitivity'] = 1.0
        
        return features
    
    def get_user_risk_profile(self, user_id: str) -> Dict:
        """Get comprehensive risk profile for a user"""
        
        if user_id not in self.user_baselines:
            return {"error": "User not found"}
        
        baseline = self.user_baselines[user_id]
        recent_events = list(self.user_history[user_id])
        
        return {
            "user_id": user_id,
            "total_events": baseline['total_events'],
            "avg_file_size": baseline['avg_file_size'],
            "typical_hours": sorted(list(baseline['typical_hours'])),
            "top_resources": dict(sorted(baseline['typical_resources'].items(), 
                                       key=lambda x: x[1], reverse=True)[:10]),
            "top_actions": dict(sorted(baseline['typical_actions'].items(), 
                                     key=lambda x: x[1], reverse=True)[:5]),
            "failure_rate": baseline['failure_rate'],
            "after_hours_rate": baseline['after_hours_rate'],
            "weekend_rate": baseline['weekend_rate'],
            "sensitivity_preference": baseline['sensitivity_preference'],
            "unique_computers": len(self.computer_relationships[user_id]),
            "unique_ips": len(self.ip_relationships[user_id]),
            "unique_resources": len(self.resource_relationships[user_id]),
            "recent_events": len(recent_events),
            "last_updated": baseline['last_updated'].isoformat()
        }
    
    def get_detection_statistics(self) -> Dict:
        """Get overall detection statistics"""
        
        total_users = len(self.user_baselines)
        total_events = sum(baseline['total_events'] for baseline in self.user_baselines.values())
        
        return {
            "total_users": total_users,
            "total_events": total_events,
            "avg_events_per_user": total_events / max(1, total_users),
            "detection_weights": self.weights,
            "risk_thresholds": {
                "block": self.BLOCK_THRESHOLD,
                "high": self.HIGH_RISK_THRESHOLD,
                "medium": self.MEDIUM_RISK_THRESHOLD
            }
        }
    
    def save_model(self, filepath: str):
        """Save the adaptive ML model to disk"""
        
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
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(model_data, f, indent=2, default=str)
        
        logger.info(f"💾 Saved adaptive ML model to {filepath}")
    
    def load_model(self, filepath: str):
        """Load the adaptive ML model from disk"""
        
        with open(filepath, 'r') as f:
            model_data = json.load(f)
        
        self.user_baselines = model_data['user_baselines']
        self.user_history = {k: deque(v, maxlen=self.HISTORY_WINDOW) 
                           for k, v in model_data['user_history'].items()}
        self.resource_relationships = {k: set(v) for k, v in model_data['resource_relationships'].items()}
        self.computer_relationships = {k: set(v) for k, v in model_data['computer_relationships'].items()}
        self.ip_relationships = {k: set(v) for k, v in model_data['ip_relationships'].items()}
        self.weights = model_data['weights']
        
        thresholds = model_data['thresholds']
        self.BLOCK_THRESHOLD = thresholds['block']
        self.HIGH_RISK_THRESHOLD = thresholds['high']
        self.MEDIUM_RISK_THRESHOLD = thresholds['medium']
        
        logger.info(f"📁 Loaded adaptive ML model from {filepath}")


if __name__ == "__main__":
    # Test the adaptive ML detector
    logger.info("Testing Adaptive ML Detector...")
    
    detector = AdaptiveMLDetector()
    
    # Simulate some events
    test_events = [
        {
            'user_id': 'USER0001',
            'resource_id': 'financial_data_001',
            'action': 'read',
            'file_size_kb': 100,
            'resource_sensitivity': 4,
            'destination_computer': 'WS-001',
            'ip_address': '192.168.1.100',
            'hour': 10,
            'is_after_hours': False,
            'is_weekend': False,
            'is_failure': False,
            'is_lateral_movement': False
        },
        {
            'user_id': 'USER0001',
            'resource_id': 'hr_records_005',
            'action': 'write',
            'file_size_kb': 5000,
            'resource_sensitivity': 5,
            'destination_computer': 'WS-002',
            'ip_address': '192.168.1.101',
            'hour': 22,
            'is_after_hours': True,
            'is_weekend': False,
            'is_failure': False,
            'is_lateral_movement': True
        }
    ]
    
    for event in test_events:
        result = detector.detect_anomaly(event['user_id'], event)
        logger.info(f"Detection result: {result.risk_score:.1f} ({result.risk_level}) - {result.explanation}")
    
    # Get user profile
    profile = detector.get_user_risk_profile('USER0001')
    logger.info(f"User profile: {profile}")
    
    logger.info("✅ Test completed successfully!")
