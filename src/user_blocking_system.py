"""
User Blocking and Revocation System for High-Risk Users
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import smtplib
    from email.mime.text import MimeText
    from email.mime.multipart import MimeMultipart
    EMAIL_AVAILABLE = True
except ImportError:
    EMAIL_AVAILABLE = False
    logger.warning("Email functionality not available - install required packages for email notifications")


class BlockReason(Enum):
    """Reasons for blocking a user"""
    HIGH_RISK_SCORE = "high_risk_score"
    SUSPICIOUS_BEHAVIOR = "suspicious_behavior"
    MULTIPLE_ANOMALIES = "multiple_anomalies"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    DATA_EXFILTRATION = "data_exfiltration"
    LATERAL_MOVEMENT = "lateral_movement"
    AFTER_HOURS_ACCESS = "after_hours_access"
    UNUSUAL_RESOURCE_ACCESS = "unusual_resource_access"
    MANUAL_BLOCK = "manual_block"


class BlockStatus(Enum):
    """Status of user blocking"""
    ACTIVE = "active"
    BLOCKED = "blocked"
    REVOKED = "revoked"
    PENDING_REVIEW = "pending_review"
    WHITELISTED = "whitelisted"


@dataclass
class BlockRecord:
    """Record of a user block action"""
    user_id: str
    timestamp: datetime
    reason: BlockReason
    risk_score: float
    detection_types: List[str]
    blocked_by: str  # System or admin
    duration_minutes: Optional[int] = None  # None = permanent
    auto_unblock_time: Optional[datetime] = None
    status: BlockStatus = BlockStatus.BLOCKED
    notes: str = ""
    reviewed_by: Optional[str] = None
    review_timestamp: Optional[datetime] = None


@dataclass
class SecurityAlert:
    """Security alert for blocked users"""
    alert_id: str
    user_id: str
    severity: str  # low, medium, high, critical
    title: str
    description: str
    timestamp: datetime
    risk_score: float
    detection_types: List[str]
    recommended_actions: List[str]
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None


class UserBlockingSystem:
    """
    Comprehensive user blocking and revocation system
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the blocking system
        
        Args:
            config_path: Path to configuration file
        """
        self.blocked_users: Dict[str, BlockRecord] = {}
        self.security_alerts: List[SecurityAlert] = []
        self.whitelist: set = set()
        self.block_history: List[BlockRecord] = []
        
        # Configuration
        self.config = {
            'auto_block_threshold': 80.0,
            'alert_threshold': 70.0,
            'auto_unblock_after_hours': 24,  # Auto-unblock after 24 hours if no review
            'email_notifications': True,
            'email_config': {
                'smtp_server': 'smtp.gmail.com',
                'smtp_port': 587,
                'username': '',
                'password': '',
                'from_email': 'security@company.com',
                'to_emails': ['admin@company.com', 'security@company.com']
            },
            'log_file': 'logs/blocking_system.log'
        }
        
        # Load configuration if provided
        if config_path and Path(config_path).exists():
            self.load_config(config_path)
        
        # Create log directory
        log_dir = Path(self.config['log_file']).parent
        log_dir.mkdir(parents=True, exist_ok=True)
    
    def load_config(self, config_path: str):
        """Load configuration from file"""
        with open(config_path, 'r') as f:
            self.config.update(json.load(f))
        logger.info(f"📁 Loaded configuration from {config_path}")
    
    def save_config(self, config_path: str):
        """Save configuration to file"""
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2, default=str)
        logger.info(f"💾 Saved configuration to {config_path}")
    
    def should_block_user(self, user_id: str, risk_score: float, 
                         detection_types: List[str]) -> Tuple[bool, BlockReason]:
        """
        Determine if a user should be blocked
        
        Args:
            user_id: User identifier
            risk_score: Current risk score
            detection_types: List of detection types
            
        Returns:
            Tuple[bool, BlockReason]: Should block and reason
        """
        
        # Check if user is whitelisted
        if user_id in self.whitelist:
            return False, BlockReason.MANUAL_BLOCK
        
        # Check if already blocked
        if user_id in self.blocked_users:
            return False, BlockReason.MANUAL_BLOCK
        
        # High risk score
        if risk_score >= self.config['auto_block_threshold']:
            return True, BlockReason.HIGH_RISK_SCORE
        
        # Specific dangerous behaviors
        dangerous_behaviors = [
            'data_exfiltration', 'lateral_movement', 'privilege_escalation'
        ]
        
        if any(behavior in detection_types for behavior in dangerous_behaviors):
            return True, BlockReason.SUSPICIOUS_BEHAVIOR
        
        # Multiple anomalies in short time
        if len(detection_types) >= 3:
            return True, BlockReason.MULTIPLE_ANOMALIES
        
        # Specific detection types that warrant blocking
        blocking_detections = [
            'unusual_large_file', 'destructive_action', 'excessive_computer_access',
            'multiple_ip_usage', 'rapid_computer_expansion'
        ]
        
        if any(detection in detection_types for detection in blocking_detections):
            return True, BlockReason.SUSPICIOUS_BEHAVIOR
        
        return False, BlockReason.MANUAL_BLOCK
    
    def block_user(self, user_id: str, risk_score: float, detection_types: List[str],
                  reason: BlockReason, duration_minutes: Optional[int] = None,
                  notes: str = "") -> BlockRecord:
        """
        Block a user and create security alert
        
        Args:
            user_id: User identifier
            risk_score: Risk score that triggered the block
            detection_types: Detection types
            reason: Reason for blocking
            duration_minutes: Duration of block (None = permanent)
            notes: Additional notes
            
        Returns:
            BlockRecord: Block record
        """
        
        # Create block record
        block_record = BlockRecord(
            user_id=user_id,
            timestamp=datetime.now(),
            reason=reason,
            risk_score=risk_score,
            detection_types=detection_types,
            blocked_by="system",
            duration_minutes=duration_minutes,
            notes=notes
        )
        
        # Set auto-unblock time if duration specified
        if duration_minutes:
            block_record.auto_unblock_time = datetime.now() + timedelta(minutes=duration_minutes)
        
        # Add to blocked users
        self.blocked_users[user_id] = block_record
        self.block_history.append(block_record)
        
        # Create security alert
        alert = self._create_security_alert(user_id, risk_score, detection_types, reason)
        self.security_alerts.append(alert)
        
        # Log the block
        self._log_block_action(block_record)
        
        # Send notifications
        if self.config['email_notifications']:
            self._send_block_notification(block_record, alert)
        
        logger.warning(f"🚨 BLOCKED USER: {user_id} (Risk: {risk_score:.1f}, Reason: {reason.value})")
        
        return block_record
    
    def unblock_user(self, user_id: str, unblocked_by: str = "admin", 
                    notes: str = "") -> bool:
        """
        Unblock a user
        
        Args:
            user_id: User identifier
            unblocked_by: Who unblocked the user
            notes: Unblock notes
            
        Returns:
            bool: Success status
        """
        
        if user_id not in self.blocked_users:
            logger.warning(f"User {user_id} is not blocked")
            return False
        
        block_record = self.blocked_users[user_id]
        block_record.status = BlockStatus.ACTIVE
        block_record.reviewed_by = unblocked_by
        block_record.review_timestamp = datetime.now()
        block_record.notes += f" | Unblocked: {notes}"
        
        # Remove from blocked users
        del self.blocked_users[user_id]
        
        # Log the unblock
        self._log_unblock_action(block_record, unblocked_by, notes)
        
        logger.info(f"✅ UNBLOCKED USER: {user_id} by {unblocked_by}")
        
        return True
    
    def revoke_user_access(self, user_id: str, revoked_by: str = "admin",
                          notes: str = "") -> bool:
        """
        Permanently revoke user access
        
        Args:
            user_id: User identifier
            revoked_by: Who revoked access
            notes: Revocation notes
            
        Returns:
            bool: Success status
        """
        
        # Create revocation record
        revoke_record = BlockRecord(
            user_id=user_id,
            timestamp=datetime.now(),
            reason=BlockReason.MANUAL_BLOCK,
            risk_score=100.0,  # Maximum risk for revocation
            detection_types=["manual_revocation"],
            blocked_by=revoked_by,
            status=BlockStatus.REVOKED,
            notes=notes
        )
        
        # Add to blocked users (permanent)
        self.blocked_users[user_id] = revoke_record
        self.block_history.append(revoke_record)
        
        # Create critical security alert
        alert = SecurityAlert(
            alert_id=f"REVOKE_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            user_id=user_id,
            severity="critical",
            title=f"User Access Revoked: {user_id}",
            description=f"User access has been permanently revoked by {revoked_by}. Reason: {notes}",
            timestamp=datetime.now(),
            risk_score=100.0,
            detection_types=["manual_revocation"],
            recommended_actions=[
                "Immediately disable all user accounts",
                "Change all shared passwords",
                "Review user's recent activities",
                "Notify security team",
                "Conduct forensic analysis"
            ]
        )
        
        self.security_alerts.append(alert)
        
        # Log the revocation
        self._log_revoke_action(revoke_record)
        
        # Send notifications
        if self.config['email_notifications']:
            self._send_revoke_notification(revoke_record, alert)
        
        logger.critical(f"🔴 REVOKED USER: {user_id} by {revoked_by}")
        
        return True
    
    def whitelist_user(self, user_id: str, whitelisted_by: str = "admin",
                      notes: str = "") -> bool:
        """
        Add user to whitelist (exempt from automatic blocking)
        
        Args:
            user_id: User identifier
            whitelisted_by: Who whitelisted the user
            notes: Whitelist notes
            
        Returns:
            bool: Success status
        """
        
        self.whitelist.add(user_id)
        
        # If user is currently blocked, unblock them
        if user_id in self.blocked_users:
            self.unblock_user(user_id, whitelisted_by, f"Whitelisted: {notes}")
        
        logger.info(f"✅ WHITELISTED USER: {user_id} by {whitelisted_by}")
        
        return True
    
    def remove_from_whitelist(self, user_id: str, removed_by: str = "admin",
                             notes: str = "") -> bool:
        """
        Remove user from whitelist
        
        Args:
            user_id: User identifier
            removed_by: Who removed from whitelist
            notes: Removal notes
            
        Returns:
            bool: Success status
        """
        
        if user_id in self.whitelist:
            self.whitelist.remove(user_id)
            logger.info(f"❌ REMOVED FROM WHITELIST: {user_id} by {removed_by}")
            return True
        
        return False
    
    def check_auto_unblock(self):
        """Check for users that should be auto-unblocked"""
        
        current_time = datetime.now()
        users_to_unblock = []
        
        for user_id, block_record in self.blocked_users.items():
            if (block_record.auto_unblock_time and 
                current_time >= block_record.auto_unblock_time):
                users_to_unblock.append(user_id)
        
        for user_id in users_to_unblock:
            self.unblock_user(user_id, "system", "Auto-unblocked after duration")
            logger.info(f"⏰ AUTO-UNBLOCKED USER: {user_id}")
    
    def get_blocked_users(self) -> List[Dict]:
        """Get list of currently blocked users"""
        
        blocked_list = []
        for user_id, block_record in self.blocked_users.items():
            blocked_list.append({
                'user_id': user_id,
                'timestamp': block_record.timestamp.isoformat(),
                'reason': block_record.reason.value,
                'risk_score': block_record.risk_score,
                'detection_types': block_record.detection_types,
                'status': block_record.status.value,
                'duration_minutes': block_record.duration_minutes,
                'auto_unblock_time': block_record.auto_unblock_time.isoformat() if block_record.auto_unblock_time else None,
                'notes': block_record.notes
            })
        
        return blocked_list
    
    def get_security_alerts(self, severity: Optional[str] = None,
                           acknowledged: Optional[bool] = None) -> List[Dict]:
        """Get security alerts with optional filtering"""
        
        alerts = []
        for alert in self.security_alerts:
            # Apply filters
            if severity and alert.severity != severity:
                continue
            if acknowledged is not None and alert.acknowledged != acknowledged:
                continue
            
            alerts.append({
                'alert_id': alert.alert_id,
                'user_id': alert.user_id,
                'severity': alert.severity,
                'title': alert.title,
                'description': alert.description,
                'timestamp': alert.timestamp.isoformat(),
                'risk_score': alert.risk_score,
                'detection_types': alert.detection_types,
                'recommended_actions': alert.recommended_actions,
                'acknowledged': alert.acknowledged,
                'acknowledged_by': alert.acknowledged_by,
                'acknowledged_at': alert.acknowledged_at.isoformat() if alert.acknowledged_at else None
            })
        
        return alerts
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge a security alert"""
        
        for alert in self.security_alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                alert.acknowledged_by = acknowledged_by
                alert.acknowledged_at = datetime.now()
                logger.info(f"✅ ACKNOWLEDGED ALERT: {alert_id} by {acknowledged_by}")
                return True
        
        return False
    
    def _create_security_alert(self, user_id: str, risk_score: float,
                              detection_types: List[str], reason: BlockReason) -> SecurityAlert:
        """Create a security alert for blocked user"""
        
        # Determine severity
        if risk_score >= 90:
            severity = "critical"
        elif risk_score >= 80:
            severity = "high"
        elif risk_score >= 70:
            severity = "medium"
        else:
            severity = "low"
        
        # Create alert
        alert = SecurityAlert(
            alert_id=f"BLOCK_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            user_id=user_id,
            severity=severity,
            title=f"User Blocked: {user_id}",
            description=f"User {user_id} has been blocked due to {reason.value}. Risk score: {risk_score:.1f}",
            timestamp=datetime.now(),
            risk_score=risk_score,
            detection_types=detection_types,
            recommended_actions=self._get_recommended_actions(reason, detection_types)
        )
        
        return alert
    
    def _get_recommended_actions(self, reason: BlockReason, 
                                detection_types: List[str]) -> List[str]:
        """Get recommended actions based on block reason and detection types"""
        
        actions = [
            "Review user's recent activities",
            "Check for data exfiltration",
            "Verify user identity",
            "Notify security team"
        ]
        
        if reason == BlockReason.HIGH_RISK_SCORE:
            actions.extend([
                "Conduct detailed risk assessment",
                "Review user's access permissions",
                "Check for compromised credentials"
            ])
        
        if 'lateral_movement' in detection_types:
            actions.extend([
                "Check all systems user accessed",
                "Review network logs",
                "Scan for malware"
            ])
        
        if 'data_exfiltration' in detection_types:
            actions.extend([
                "Immediately secure sensitive data",
                "Check data transfer logs",
                "Notify data protection officer"
            ])
        
        if 'privilege_escalation' in detection_types:
            actions.extend([
                "Review user's permissions",
                "Check for privilege abuse",
                "Audit system access logs"
            ])
        
        return actions
    
    def _log_block_action(self, block_record: BlockRecord):
        """Log block action to file"""
        
        log_entry = {
            'action': 'BLOCK',
            'timestamp': block_record.timestamp.isoformat(),
            'user_id': block_record.user_id,
            'reason': block_record.reason.value,
            'risk_score': block_record.risk_score,
            'detection_types': block_record.detection_types,
            'blocked_by': block_record.blocked_by,
            'notes': block_record.notes
        }
        
        self._write_log_entry(log_entry)
    
    def _log_unblock_action(self, block_record: BlockRecord, unblocked_by: str, notes: str):
        """Log unblock action to file"""
        
        log_entry = {
            'action': 'UNBLOCK',
            'timestamp': datetime.now().isoformat(),
            'user_id': block_record.user_id,
            'original_reason': block_record.reason.value,
            'unblocked_by': unblocked_by,
            'notes': notes
        }
        
        self._write_log_entry(log_entry)
    
    def _log_revoke_action(self, revoke_record: BlockRecord):
        """Log revocation action to file"""
        
        log_entry = {
            'action': 'REVOKE',
            'timestamp': revoke_record.timestamp.isoformat(),
            'user_id': revoke_record.user_id,
            'revoked_by': revoke_record.blocked_by,
            'notes': revoke_record.notes
        }
        
        self._write_log_entry(log_entry)
    
    def _write_log_entry(self, log_entry: Dict):
        """Write log entry to file"""
        
        log_file = Path(self.config['log_file'])
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def _send_block_notification(self, block_record: BlockRecord, alert: SecurityAlert):
        """Send email notification for blocked user"""
        
        if not self.config['email_notifications'] or not EMAIL_AVAILABLE:
            return
        
        try:
            # Create email
            msg = MimeMultipart()
            msg['From'] = self.config['email_config']['from_email']
            msg['To'] = ', '.join(self.config['email_config']['to_emails'])
            msg['Subject'] = f"SECURITY ALERT: User Blocked - {block_record.user_id}"
            
            # Email body
            body = f"""
            SECURITY ALERT: User Blocked
            
            User ID: {block_record.user_id}
            Timestamp: {block_record.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
            Risk Score: {block_record.risk_score:.1f}
            Reason: {block_record.reason.value}
            Detection Types: {', '.join(block_record.detection_types)}
            
            Recommended Actions:
            {chr(10).join(f"- {action}" for action in alert.recommended_actions)}
            
            Please review and take appropriate action.
            """
            
            msg.attach(MimeText(body, 'plain'))
            
            # Send email
            server = smtplib.SMTP(self.config['email_config']['smtp_server'], 
                                self.config['email_config']['smtp_port'])
            server.starttls()
            server.login(self.config['email_config']['username'], 
                        self.config['email_config']['password'])
            server.send_message(msg)
            server.quit()
            
            logger.info(f"📧 Sent block notification for user {block_record.user_id}")
            
        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")
    
    def _send_revoke_notification(self, revoke_record: BlockRecord, alert: SecurityAlert):
        """Send email notification for revoked user"""
        
        if not self.config['email_notifications'] or not EMAIL_AVAILABLE:
            return
        
        try:
            # Create email
            msg = MimeMultipart()
            msg['From'] = self.config['email_config']['from_email']
            msg['To'] = ', '.join(self.config['email_config']['to_emails'])
            msg['Subject'] = f"CRITICAL: User Access Revoked - {revoke_record.user_id}"
            
            # Email body
            body = f"""
            CRITICAL SECURITY ALERT: User Access Revoked
            
            User ID: {revoke_record.user_id}
            Timestamp: {revoke_record.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
            Revoked By: {revoke_record.blocked_by}
            Notes: {revoke_record.notes}
            
            IMMEDIATE ACTIONS REQUIRED:
            {chr(10).join(f"- {action}" for action in alert.recommended_actions)}
            
            This is a critical security incident requiring immediate attention.
            """
            
            msg.attach(MimeText(body, 'plain'))
            
            # Send email
            server = smtplib.SMTP(self.config['email_config']['smtp_server'], 
                                self.config['email_config']['smtp_port'])
            server.starttls()
            server.login(self.config['email_config']['username'], 
                        self.config['email_config']['password'])
            server.send_message(msg)
            server.quit()
            
            logger.info(f"📧 Sent revocation notification for user {revoke_record.user_id}")
            
        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")
    
    def export_blocking_data(self, filepath: str):
        """Export blocking system data to JSON file"""
        
        data = {
            'blocked_users': {k: asdict(v) for k, v in self.blocked_users.items()},
            'security_alerts': [asdict(alert) for alert in self.security_alerts],
            'whitelist': list(self.whitelist),
            'block_history': [asdict(record) for record in self.block_history],
            'config': self.config
        }
        
        # Convert datetime objects to strings
        def convert_datetime(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, dict):
                return {k: convert_datetime(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_datetime(item) for item in obj]
            else:
                return obj
        
        data = convert_datetime(data)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"💾 Exported blocking data to {filepath}")
    
    def load_blocking_data(self, filepath: str):
        """Load blocking system data from JSON file"""
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Reconstruct blocked users
        self.blocked_users = {}
        for user_id, record_data in data['blocked_users'].items():
            record_data['timestamp'] = datetime.fromisoformat(record_data['timestamp'])
            if record_data['auto_unblock_time']:
                record_data['auto_unblock_time'] = datetime.fromisoformat(record_data['auto_unblock_time'])
            if record_data['review_timestamp']:
                record_data['review_timestamp'] = datetime.fromisoformat(record_data['review_timestamp'])
            record_data['reason'] = BlockReason(record_data['reason'])
            record_data['status'] = BlockStatus(record_data['status'])
            self.blocked_users[user_id] = BlockRecord(**record_data)
        
        # Reconstruct security alerts
        self.security_alerts = []
        for alert_data in data['security_alerts']:
            alert_data['timestamp'] = datetime.fromisoformat(alert_data['timestamp'])
            if alert_data['acknowledged_at']:
                alert_data['acknowledged_at'] = datetime.fromisoformat(alert_data['acknowledged_at'])
            self.security_alerts.append(SecurityAlert(**alert_data))
        
        # Reconstruct other data
        self.whitelist = set(data['whitelist'])
        self.config.update(data['config'])
        
        logger.info(f"📁 Loaded blocking data from {filepath}")


if __name__ == "__main__":
    # Test the blocking system
    logger.info("Testing User Blocking System...")
    
    # Create blocking system
    blocking_system = UserBlockingSystem()
    
    # Test blocking a user
    test_user_id = "USER0001"
    test_risk_score = 85.0
    test_detection_types = ["unusual_large_file", "after_hours_access", "lateral_movement"]
    
    should_block, reason = blocking_system.should_block_user(
        test_user_id, test_risk_score, test_detection_types
    )
    
    if should_block:
        block_record = blocking_system.block_user(
            test_user_id, test_risk_score, test_detection_types, reason
        )
        logger.info(f"Blocked user: {block_record.user_id}")
    
    # Test getting blocked users
    blocked_users = blocking_system.get_blocked_users()
    logger.info(f"Blocked users: {len(blocked_users)}")
    
    # Test getting security alerts
    alerts = blocking_system.get_security_alerts()
    logger.info(f"Security alerts: {len(alerts)}")
    
    # Test unblocking
    blocking_system.unblock_user(test_user_id, "admin", "Test unblock")
    
    logger.info("✅ Test completed successfully!")
