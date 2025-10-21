"""
Generate synthetic authentication logs using Faker for demo/testing
"""

import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import logging
from tqdm import tqdm

from .config import (
    SYNTHETIC_PATH, SYNTHETIC_NUM_USERS, SYNTHETIC_NUM_LOGS,
    SYNTHETIC_ANOMALY_RATIO, AFTER_HOURS_START, AFTER_HOURS_END
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

fake = Faker()
Faker.seed(42)
np.random.seed(42)


class SyntheticDataGenerator:
    """Generate realistic synthetic authentication logs with anomalies"""
    
    def __init__(self, num_users=SYNTHETIC_NUM_USERS, num_logs=SYNTHETIC_NUM_LOGS):
        self.num_users = num_users
        self.num_logs = num_logs
        self.anomaly_ratio = SYNTHETIC_ANOMALY_RATIO
        
        # Generate consistent user/resource pools
        self.users = [f"USER{i:04d}" for i in range(num_users)]
        self.computers = [f"COMP{i:04d}" for i in range(50)]
        self.resources = [f"FILE_{i:05d}" for i in range(200)]
        self.ip_addresses = [fake.ipv4_private() for _ in range(30)]
        
        # Identify anomalous users
        num_anomalies = int(num_users * self.anomaly_ratio)
        self.anomalous_users = np.random.choice(self.users, num_anomalies, replace=False)
        
        logger.info(f"Anomalous users: {self.anomalous_users}")
    
    def generate(self):
        """
        Generate synthetic authentication logs
        
        Returns:
            pd.DataFrame: Synthetic logs with normal and anomalous behavior
        """
        logger.info(f"Generating {self.num_logs} synthetic authentication logs...")
        
        logs = []
        
        for _ in tqdm(range(self.num_logs), desc="Generating logs"):
            user = np.random.choice(self.users)
            
            if user in self.anomalous_users:
                log = self._generate_anomalous_log(user)
            else:
                log = self._generate_normal_log(user)
            
            logs.append(log)
        
        df = pd.DataFrame(logs)
        
        # Add ground truth labels
        df['is_anomalous_user'] = df['user_id'].isin(self.anomalous_users).astype(int)
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        logger.info(f"Generated {len(df)} logs ({len(self.anomalous_users)} anomalous users)")
        
        return df
    
    def _generate_normal_log(self, user):
        """Generate normal authentication log"""
        # Normal working hours (8 AM - 6 PM, weekdays)
        base_time = datetime.now() - timedelta(days=np.random.randint(1, 30))
        hour = np.random.choice(range(8, 18), p=self._normal_hour_distribution())
        timestamp = base_time.replace(hour=hour, minute=np.random.randint(0, 60))
        
        # Skip weekends mostly
        if timestamp.weekday() >= 5:
            if np.random.random() > 0.1:  # 90% chance to regenerate
                timestamp -= timedelta(days=np.random.randint(1, 3))
        
        return {
            'timestamp': timestamp,
            'user_id': user,
            'source_computer': np.random.choice(self.computers[:20]),  # Limited pool
            'destination_computer': np.random.choice(self.computers[:20]),
            'resource_id': np.random.choice(self.resources[:100]),  # Non-sensitive
            'action': np.random.choice(['read', 'write', 'execute'], p=[0.6, 0.3, 0.1]),
            'ip_address': np.random.choice(self.ip_addresses[:5]),  # Consistent IPs
            'auth_type': np.random.choice(['Kerberos', 'NTLM'], p=[0.9, 0.1]),
            'success': 'Success',
            'is_failure': 0,
            'file_size_kb': int(np.random.lognormal(5, 1.5)),
            'resource_sensitivity': np.random.randint(1, 3),  # Low-medium
            'is_after_hours': 0,
            'is_weekend': 0,
            'is_lateral_movement': 0,
        }
    
    def _generate_anomalous_log(self, user):
        """Generate anomalous authentication log with suspicious patterns"""
        base_time = datetime.now() - timedelta(days=np.random.randint(1, 30))
        
        # Anomaly patterns
        anomaly_type = np.random.choice([
            'after_hours',
            'sensitive_access',
            'lateral_movement',
            'failed_auth',
            'data_exfil'
        ])
        
        if anomaly_type == 'after_hours':
            # After-hours access
            hour = np.random.choice([22, 23, 0, 1, 2, 3, 4, 5])
            timestamp = base_time.replace(hour=hour, minute=np.random.randint(0, 60))
            is_after_hours = 1
            resource_sensitivity = np.random.randint(3, 6)  # Higher sensitivity
            
        elif anomaly_type == 'sensitive_access':
            # Access to high-sensitivity resources
            hour = np.random.randint(8, 18)
            timestamp = base_time.replace(hour=hour, minute=np.random.randint(0, 60))
            is_after_hours = 0
            resource_sensitivity = 5  # Max sensitivity
            
        elif anomaly_type == 'lateral_movement':
            # Unusual computer access
            hour = np.random.randint(8, 20)
            timestamp = base_time.replace(hour=hour, minute=np.random.randint(0, 60))
            is_after_hours = 1 if hour >= AFTER_HOURS_START else 0
            resource_sensitivity = np.random.randint(2, 5)
            
        elif anomaly_type == 'failed_auth':
            # Failed authentication attempts
            hour = np.random.randint(0, 24)
            timestamp = base_time.replace(hour=hour, minute=np.random.randint(0, 60))
            is_after_hours = 1 if (hour < AFTER_HOURS_END or hour >= AFTER_HOURS_START) else 0
            resource_sensitivity = np.random.randint(1, 4)
            
        else:  # data_exfil
            # Large file transfers after hours
            hour = np.random.choice([20, 21, 22, 23])
            timestamp = base_time.replace(hour=hour, minute=np.random.randint(0, 60))
            is_after_hours = 1
            resource_sensitivity = 4
        
        return {
            'timestamp': timestamp,
            'user_id': user,
            'source_computer': np.random.choice(self.computers),  # Any computer
            'destination_computer': np.random.choice(self.computers),
            'resource_id': np.random.choice(self.resources),  # Any resource
            'action': np.random.choice(['read', 'write', 'delete', 'copy'], p=[0.4, 0.3, 0.2, 0.1]),
            'ip_address': np.random.choice(self.ip_addresses),  # Varied IPs
            'auth_type': np.random.choice(['Kerberos', 'NTLM', 'Basic'], p=[0.6, 0.3, 0.1]),
            'success': 'Success' if anomaly_type != 'failed_auth' else 'Failure',
            'is_failure': 1 if anomaly_type == 'failed_auth' else 0,
            'file_size_kb': int(np.random.lognormal(8, 2)) if anomaly_type == 'data_exfil' else int(np.random.lognormal(5, 1.5)),
            'resource_sensitivity': resource_sensitivity,
            'is_after_hours': is_after_hours,
            'is_weekend': 1 if timestamp.weekday() >= 5 else 0,
            'is_lateral_movement': 1 if anomaly_type == 'lateral_movement' else 0,
        }
    
    def _normal_hour_distribution(self):
        """Probability distribution for normal working hours"""
        # Peak at 10 AM and 2 PM
        hours = list(range(8, 18))
        probs = [0.05, 0.10, 0.15, 0.15, 0.10, 0.08, 0.12, 0.12, 0.08, 0.05]
        return probs
    
    def save(self, df, file_path=SYNTHETIC_PATH):
        """Save synthetic data to CSV"""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(file_path, index=False)
        logger.info(f"Saved synthetic data to {file_path}")


def generate_synthetic_logs(output_path):
    # Implement synthetic log generation using Faker here
    pass


def main():
    """Generate and save synthetic data"""
    generator = SyntheticDataGenerator()
    df = generator.generate()
    
    # Display statistics
    print("\n=== Synthetic Data Summary ===")
    print(f"Total logs: {len(df)}")
    print(f"Unique users: {df['user_id'].nunique()}")
    print(f"Anomalous users: {df['is_anomalous_user'].sum()}")
    print(f"After-hours logs: {df['is_after_hours'].sum()} ({100*df['is_after_hours'].mean():.1f}%)")
    print(f"Failed authentications: {df['is_failure'].sum()}")
    print(f"\nSample logs:")
    print(df.head(10))
    
    # Save
    generator.save(df)
    print(f"\n[OK] Synthetic data generated at: {SYNTHETIC_PATH}")


if __name__ == "__main__":
    main()
