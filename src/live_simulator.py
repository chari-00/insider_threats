"""
Live Simulation System for Real-time Cyber Threat Detection
Generates continuous authentication events and manages user states
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging
import threading
import time
import random
from pathlib import Path
from collections import defaultdict

from .config import SYNTHETIC_DATA_DIR, SYNTHETIC_NUM_USERS, AFTER_HOURS_START, AFTER_HOURS_END
from .data_generator import SyntheticDataGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class User:
    """User state for simulation"""
    user_id: str
    department: str
    role: str
    current_risk_score: float = 0.0
    is_blocked: bool = False
    access_count: int = 0
    last_access: Optional[datetime] = None
    typical_hours: set = None
    typical_resources: set = None
    typical_computers: set = None
    
    def __post_init__(self):
        if self.typical_hours is None:
            self.typical_hours = set()
        if self.typical_resources is None:
            self.typical_resources = set()
        if self.typical_computers is None:
            self.typical_computers = set()


@dataclass
class Event:
    """Authentication event for simulation"""
    timestamp: datetime
    user_id: str
    resource_id: str
    action: str
    destination_computer: str
    source_computer: str
    ip_address: str
    auth_type: str
    success: str
    file_size_kb: int
    resource_sensitivity: int
    risk_score: float
    detection_type: str
    detection_types: List[str] = None  # Enhanced detection types
    is_after_hours: bool = False
    is_weekend: bool = False
    is_failure: bool = False
    is_lateral_movement: bool = False


class LiveSimulator:
    """
    Real-time simulation system that generates continuous authentication events
    and manages user states for live threat detection
    """
    
    def __init__(self, num_users: int = 150, simulation_speed: float = 1.0):
        self.num_users = num_users
        self.simulation_speed = simulation_speed
        self.BLOCK_THRESHOLD = 80.0
        
        # Simulation state
        self.is_running = False
        self.start_time = None
        self.duration_minutes = 60
        
        # Data storage
        self.events: List[Event] = []
        self.users: Dict[str, User] = {}
        
        # Event generation pools
        self.computers = [f"COMP{i:04d}" for i in range(50)]
        self.resources = [f"FILE_{i:05d}" for i in range(200)]
        self.ip_addresses = [f"192.168.1.{i}" for i in range(1, 31)]
        self.departments = ["IT", "HR", "Finance", "Marketing", "Sales", "Operations", "Legal"]
        self.roles = ["Manager", "Employee", "Admin", "Contractor", "Intern"]
        self.actions = ["read", "write", "execute", "delete", "copy"]
        self.auth_types = ["Kerberos", "NTLM", "Basic"]
        
        # Simulation thread
        self.simulation_thread = None
        
        # Initialize users
        self._initialize_users()
        
        logger.info(f"🚀 LiveSimulator initialized with {num_users} users")
    
    def _initialize_users(self):
        """Initialize user pool with realistic profiles"""
        
        for i in range(self.num_users):
            user_id = f"USER{i:04d}"
            
            # Create user with realistic profile
            user = User(
                user_id=user_id,
                department=random.choice(self.departments),
                role=random.choice(self.roles)
            )
            
            # Set typical behavior patterns
            user.typical_hours = set(random.sample(range(8, 18), k=random.randint(3, 8)))
            user.typical_resources = set(random.sample(self.resources[:100], k=random.randint(5, 20)))
            user.typical_computers = set(random.sample(self.computers[:20], k=random.randint(2, 8)))
            
            self.users[user_id] = user
        
        logger.info(f"✅ Initialized {len(self.users)} users")
    
    def start_simulation(self, duration_minutes: int = 60):
        """Start the live simulation"""
        
        if self.is_running:
            logger.warning("Simulation is already running")
            return
        
        self.duration_minutes = duration_minutes
        self.is_running = True
        self.start_time = datetime.now()
        
        # Start simulation thread
        self.simulation_thread = threading.Thread(target=self._simulation_loop, daemon=True)
        self.simulation_thread.start()
        
        logger.info(f"🎮 Started simulation for {duration_minutes} minutes")
    
    def stop_simulation(self):
        """Stop the live simulation"""
        
        if not self.is_running:
            logger.warning("Simulation is not running")
            return
        
        self.is_running = False
        
        if self.simulation_thread:
            self.simulation_thread.join(timeout=1)
        
        logger.info("🛑 Simulation stopped")
    
    def _simulation_loop(self):
        """Main simulation loop that generates events"""
        
        end_time = self.start_time + timedelta(minutes=self.duration_minutes)
        
        while self.is_running and datetime.now() < end_time:
            # Generate event
            event = self._generate_event()
            
            if event:
                self.events.append(event)
                
                # Update user state
                self._update_user_state(event)
            
            # Wait based on simulation speed
            sleep_time = 1.0 / self.simulation_speed
            time.sleep(sleep_time)
        
        self.is_running = False
        logger.info(f"⏰ Simulation completed. Generated {len(self.events)} events")
    
    def _generate_event(self) -> Optional[Event]:
        """Generate a single authentication event"""
        
        # Select random user
        user_id = random.choice(list(self.users.keys()))
        user = self.users[user_id]
        
        # Skip if user is blocked
        if user.is_blocked:
            return None
        
        # Generate event timestamp
        timestamp = datetime.now()
        
        # Determine if this is an anomalous event (10% chance)
        is_anomalous = random.random() < 0.1
        
        if is_anomalous:
            return self._generate_anomalous_event(user, timestamp)
        else:
            return self._generate_normal_event(user, timestamp)
    
    def _generate_normal_event(self, user: User, timestamp: datetime) -> Event:
        """Generate normal authentication event"""
        
        # Normal working hours
        hour = random.choice(list(user.typical_hours)) if user.typical_hours else random.randint(8, 17)
        
        # Use typical resources and computers
        resource_id = random.choice(list(user.typical_resources)) if user.typical_resources else random.choice(self.resources[:100])
        destination_computer = random.choice(list(user.typical_computers)) if user.typical_computers else random.choice(self.computers[:20])
        
        # Normal actions
        action = random.choice(["read", "write", "execute"])
        
        # Low-medium sensitivity
        resource_sensitivity = random.randint(1, 3)
        
        # Calculate risk score (low for normal events)
        risk_score = random.uniform(5, 25)
        
        return Event(
            timestamp=timestamp,
            user_id=user.user_id,
            resource_id=resource_id,
            action=action,
            destination_computer=destination_computer,
            source_computer=random.choice(self.computers),
            ip_address=random.choice(self.ip_addresses[:5]),  # Consistent IPs
            auth_type=random.choice(["Kerberos", "NTLM"]),
            success="Success",
            file_size_kb=int(np.random.lognormal(5, 1.5)),
            resource_sensitivity=resource_sensitivity,
            risk_score=risk_score,
            detection_type="normal",
            is_after_hours=False,
            is_weekend=False,
            is_failure=False,
            is_lateral_movement=False
        )
    
    def _generate_anomalous_event(self, user: User, timestamp: datetime) -> Event:
        """Generate anomalous authentication event"""
        
        # Choose anomaly type
        anomaly_types = [
            "after_hours", "sensitive_access", "lateral_movement", 
            "failed_auth", "data_exfil", "unusual_resource", "high_frequency"
        ]
        anomaly_type = random.choice(anomaly_types)
        
        if anomaly_type == "after_hours":
            # After-hours access
            hour = random.choice([22, 23, 0, 1, 2, 3, 4, 5])
            resource_sensitivity = random.randint(3, 5)
            risk_score = random.uniform(60, 85)
            detection_type = "after_hours_access"
            is_after_hours = True
            
        elif anomaly_type == "sensitive_access":
            # High-sensitivity resource access
            hour = random.randint(8, 18)
            resource_sensitivity = 5
            risk_score = random.uniform(70, 90)
            detection_type = "sensitive_resource_access"
            is_after_hours = False
            
        elif anomaly_type == "lateral_movement":
            # Unusual computer access
            hour = random.randint(8, 20)
            resource_sensitivity = random.randint(2, 4)
            risk_score = random.uniform(65, 85)
            detection_type = "lateral_movement"
            is_after_hours = hour >= AFTER_HOURS_START
            
        elif anomaly_type == "failed_auth":
            # Failed authentication
            hour = random.randint(0, 23)
            resource_sensitivity = random.randint(1, 3)
            risk_score = random.uniform(50, 75)
            detection_type = "failed_authentication"
            is_after_hours = hour < AFTER_HOURS_END or hour >= AFTER_HOURS_START
            
        elif anomaly_type == "data_exfil":
            # Large file transfer
            hour = random.choice([20, 21, 22, 23])
            resource_sensitivity = 4
            risk_score = random.uniform(75, 95)
            detection_type = "data_exfiltration"
            is_after_hours = True
            
        elif anomaly_type == "unusual_resource":
            # Accessing unusual resources
            hour = random.randint(8, 18)
            resource_sensitivity = random.randint(3, 5)
            risk_score = random.uniform(55, 80)
            detection_type = "unusual_resource_access"
            is_after_hours = False
            
        else:  # high_frequency
            # High frequency access
            hour = random.randint(8, 18)
            resource_sensitivity = random.randint(2, 4)
            risk_score = random.uniform(60, 85)
            detection_type = "high_frequency_access"
            is_after_hours = False
        
        # Generate event data
        resource_id = random.choice(self.resources)
        destination_computer = random.choice(self.computers)
        action = random.choice(self.actions)
        
        return Event(
            timestamp=timestamp,
            user_id=user.user_id,
            resource_id=resource_id,
            action=action,
            destination_computer=destination_computer,
            source_computer=random.choice(self.computers),
            ip_address=random.choice(self.ip_addresses),
            auth_type=random.choice(self.auth_types),
            success="Success" if anomaly_type != "failed_auth" else "Failure",
            file_size_kb=int(np.random.lognormal(8, 2)) if anomaly_type == "data_exfil" else int(np.random.lognormal(5, 1.5)),
            resource_sensitivity=resource_sensitivity,
            risk_score=risk_score,
            detection_type=detection_type,
            is_after_hours=is_after_hours,
            is_weekend=timestamp.weekday() >= 5,
            is_failure=anomaly_type == "failed_auth",
            is_lateral_movement=anomaly_type == "lateral_movement"
        )
    
    def _update_user_state(self, event: Event):
        """Update user state based on event"""
        
        user = self.users[event.user_id]
        
        # Update access count
        user.access_count += 1
        user.last_access = event.timestamp
        
        # Update risk score (rolling average)
        if user.current_risk_score == 0:
            user.current_risk_score = event.risk_score
        else:
            # Weighted average with recent events having more weight
            weight = 0.3
            user.current_risk_score = (1 - weight) * user.current_risk_score + weight * event.risk_score
        
        # Block user if risk score exceeds threshold
        if user.current_risk_score >= self.BLOCK_THRESHOLD and not user.is_blocked:
            user.is_blocked = True
            logger.warning(f"🚫 User {user.user_id} blocked due to high risk score: {user.current_risk_score:.1f}")
    
    def get_simulation_stats(self) -> Dict:
        """Get current simulation statistics"""
        
        if not self.events:
            return {
                'total_events': 0,
                'blocked_users': 0,
                'high_risk_events': 0,
                'simulation_time': '0:00:00',
                'events_per_minute': 0,
                'risk_distribution': {}
            }
        
        # Calculate statistics
        total_events = len(self.events)
        blocked_users = sum(1 for user in self.users.values() if user.is_blocked)
        high_risk_events = sum(1 for event in self.events if event.risk_score >= 70)
        
        # Simulation time
        if self.start_time:
            elapsed = datetime.now() - self.start_time
            simulation_time = str(elapsed).split('.')[0]  # Remove microseconds
        else:
            simulation_time = '0:00:00'
        
        # Events per minute
        if self.start_time:
            elapsed_minutes = (datetime.now() - self.start_time).total_seconds() / 60
            events_per_minute = total_events / max(elapsed_minutes, 1)
        else:
            events_per_minute = 0
        
        # Risk distribution
        risk_levels = {'Low': 0, 'Medium': 0, 'High': 0}
        for event in self.events:
            if event.risk_score < 40:
                risk_levels['Low'] += 1
            elif event.risk_score < 70:
                risk_levels['Medium'] += 1
            else:
                risk_levels['High'] += 1
        
        return {
            'total_events': total_events,
            'blocked_users': blocked_users,
            'high_risk_events': high_risk_events,
            'simulation_time': simulation_time,
            'events_per_minute': events_per_minute,
            'risk_distribution': risk_levels
        }
    
    def process_event_with_ml(self, event: Event, detector=None) -> Event:
        """
        Process event through ML detection system
        
        Args:
            event: Authentication event
            detector: Enhanced ML detector
            
        Returns:
            Event with updated risk score and detection types
        """
        if detector is None:
            return event
        
        try:
            # Convert event to dict for ML detector
            event_data = {
                'user_id': event.user_id,
                'resource_id': event.resource_id,
                'action': event.action,
                'file_size_kb': event.file_size_kb,
                'resource_sensitivity': event.resource_sensitivity,
                'destination_computer': event.destination_computer,
                'ip_address': event.ip_address,
                'hour': event.timestamp.hour,
                'is_after_hours': event.is_after_hours,
                'is_weekend': event.is_weekend,
                'is_failure': event.is_failure,
                'is_lateral_movement': event.is_lateral_movement
            }
            
            # Run enhanced ML detection
            detection_result = detector.detect_anomaly(event.user_id, event_data)
            
            # Update event with ML results
            event.risk_score = detection_result.risk_score
            event.detection_types = detection_result.detection_types
            event.detection_type = detection_result.detection_types[0] if detection_result.detection_types else "normal"
            
            return event
            
        except Exception as e:
            logger.error(f"ML processing error: {e}")
            return event


def create_simulation_data(num_users: int = 150):
    """
    Create initial simulation data using the existing data generator
    """
    
    logger.info(f"📊 Creating simulation data for {num_users} users...")
    
    # Create synthetic data directory
    synthetic_dir = Path("data/synthetic")
    synthetic_dir.mkdir(parents=True, exist_ok=True)
    
    # Use existing data generator
    generator = SyntheticDataGenerator(num_users=num_users, num_logs=num_users * 50)
    df = generator.generate()
    
    # Save to CSV
    output_path = synthetic_dir / "simulation_data.csv"
    df.to_csv(output_path, index=False)
    
    logger.info(f"✅ Simulation data created: {output_path}")
    logger.info(f"   - Total logs: {len(df)}")
    logger.info(f"   - Unique users: {df['user_id'].nunique()}")
    logger.info(f"   - Anomalous users: {df['is_anomalous_user'].sum()}")
    
    return output_path


if __name__ == "__main__":
    # Test the live simulator
    logger.info("Testing LiveSimulator...")
    
    simulator = LiveSimulator(num_users=10, simulation_speed=2.0)
    
    # Start simulation for 30 seconds
    simulator.start_simulation(duration_minutes=0.5)
    
    # Wait for simulation to complete
    time.sleep(35)
    
    # Print statistics
    stats = simulator.get_simulation_stats()
    logger.info(f"Simulation stats: {stats}")
    
    user_stats = simulator.get_user_stats()
    logger.info(f"User stats: {user_stats}")
    
    logger.info("✅ Test completed successfully!")
