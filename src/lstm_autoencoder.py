"""
LSTM Auto-Encoder for temporal pattern learning in cyber threat detection
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
import logging
from typing import Tuple, Dict, List, Optional
import joblib
from pathlib import Path
from collections import deque

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LSTMAutoEncoder:
    """
    LSTM Auto-Encoder for detecting temporal anomalies in user behavior sequences
    """
    
    def __init__(self, sequence_length: int = 20, feature_dim: int = 10, 
                 lstm_units: List[int] = [64, 32], dropout_rate: float = 0.2):
        """
        Initialize LSTM Auto-Encoder
        
        Args:
            sequence_length: Length of input sequences
            feature_dim: Number of features per timestep
            lstm_units: LSTM layer units for encoder/decoder
            dropout_rate: Dropout rate for regularization
        """
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        
        self.scaler = StandardScaler()
        self.autoencoder = None
        self.encoder = None
        self.decoder = None
        self.is_trained = False
        
        # Anomaly threshold
        self.anomaly_threshold = 95.0
        
        logger.info(f"LSTM Auto-Encoder initialized: seq_len={sequence_length}, features={feature_dim}")
    
    def _build_autoencoder(self):
        """Build LSTM Auto-Encoder architecture"""
        
        # Input layer
        inputs = keras.Input(shape=(self.sequence_length, self.feature_dim))
        
        # Encoder
        x = inputs
        for i, units in enumerate(self.lstm_units):
            return_sequences = i < len(self.lstm_units) - 1
            x = layers.LSTM(units, return_sequences=return_sequences, 
                           dropout=self.dropout_rate, recurrent_dropout=self.dropout_rate)(x)
        
        # Encoded representation (bottleneck)
        encoded = x
        
        # Decoder
        x = encoded
        for i, units in enumerate(reversed(self.lstm_units)):
            if i == 0:
                # First decoder layer needs to repeat the encoded vector
                x = layers.RepeatVector(self.sequence_length)(x)
            return_sequences = i < len(self.lstm_units) - 1
            x = layers.LSTM(units, return_sequences=return_sequences,
                           dropout=self.dropout_rate, recurrent_dropout=self.dropout_rate)(x)
        
        # Output layer - ensure we have the right shape
        if len(x.shape) == 2:  # If we don't have time dimension
            x = layers.RepeatVector(self.sequence_length)(x)
        outputs = layers.TimeDistributed(layers.Dense(self.feature_dim, activation='linear'))(x)
        
        # Auto-encoder model
        self.autoencoder = keras.Model(inputs, outputs, name='lstm_autoencoder')
        
        # Encoder model
        self.encoder = keras.Model(inputs, encoded, name='lstm_encoder')
        
        # Decoder model
        encoded_input = keras.Input(shape=(self.lstm_units[-1],))
        x = layers.RepeatVector(self.sequence_length)(encoded_input)
        for i, units in enumerate(reversed(self.lstm_units)):
            return_sequences = i < len(self.lstm_units) - 1
            x = layers.LSTM(units, return_sequences=return_sequences,
                           dropout=self.dropout_rate, recurrent_dropout=self.dropout_rate)(x)
        
        # Ensure proper shape for TimeDistributed
        if len(x.shape) == 2:
            x = layers.RepeatVector(self.sequence_length)(x)
        decoder_outputs = layers.TimeDistributed(layers.Dense(self.feature_dim, activation='linear'))(x)
        self.decoder = keras.Model(encoded_input, decoder_outputs, name='lstm_decoder')
        
        # Compile model
        self.autoencoder.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        logger.info("LSTM Auto-Encoder architecture built successfully")
    
    def _create_sequences(self, data: np.ndarray) -> np.ndarray:
        """
        Create sequences from time series data
        
        Args:
            data: Time series data (samples, features)
            
        Returns:
            Sequences (samples, sequence_length, features)
        """
        sequences = []
        for i in range(len(data) - self.sequence_length + 1):
            sequences.append(data[i:i + self.sequence_length])
        
        return np.array(sequences)
    
    def train(self, X: np.ndarray, epochs: int = 100, batch_size: int = 32, 
              validation_split: float = 0.2, verbose: int = 1):
        """
        Train LSTM Auto-Encoder
        
        Args:
            X: Training data (samples, features)
            epochs: Number of training epochs
            batch_size: Batch size
            validation_split: Validation split ratio
            verbose: Verbosity level
        """
        logger.info(f"Training LSTM Auto-Encoder with {len(X)} samples...")
        
        # Scale data
        X_scaled = self.scaler.fit_transform(X)
        
        # Create sequences
        X_sequences = self._create_sequences(X_scaled)
        
        logger.info(f"Created {len(X_sequences)} sequences of length {self.sequence_length}")
        
        # Build model if not already built
        if self.autoencoder is None:
            self._build_autoencoder()
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=15, restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=8, min_lr=1e-6
            )
        ]
        
        # Train model
        history = self.autoencoder.fit(
            X_sequences, X_sequences,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=verbose
        )
        
        self.is_trained = True
        logger.info("LSTM Auto-Encoder training completed")
        
        return history
    
    def predict_anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomaly scores using reconstruction error
        
        Args:
            X: Input data (samples, features)
            
        Returns:
            Anomaly scores (higher = more anomalous)
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        X_scaled = self.scaler.transform(X)
        X_sequences = self._create_sequences(X_scaled)
        
        # Get reconstructions
        reconstructions = self.autoencoder.predict(X_sequences, verbose=0)
        
        # Calculate reconstruction error (MSE) for each sequence
        reconstruction_errors = np.mean(np.square(X_sequences - reconstructions), axis=(1, 2))
        
        # Normalize to 0-100 scale
        scores = self._normalize_scores(reconstruction_errors)
        
        return scores
    
    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """Normalize scores to 0-100 range"""
        min_score = np.min(scores)
        max_score = np.max(scores)
        
        if max_score - min_score > 0:
            normalized = 100 * (scores - min_score) / (max_score - min_score)
        else:
            normalized = np.zeros_like(scores)
        
        return normalized
    
    def detect_anomalies(self, X: np.ndarray, threshold_percentile: float = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalies using reconstruction error threshold
        
        Args:
            X: Input data
            threshold_percentile: Percentile for anomaly threshold
            
        Returns:
            Tuple of (anomaly_scores, is_anomaly)
        """
        scores = self.predict_anomaly_scores(X)
        
        if threshold_percentile is None:
            threshold_percentile = self.anomaly_threshold
        
        threshold = np.percentile(scores, threshold_percentile)
        is_anomaly = scores > threshold
        
        return scores, is_anomaly
    
    def get_encoded_representation(self, X: np.ndarray) -> np.ndarray:
        """Get encoded representation of sequences"""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        X_scaled = self.scaler.transform(X)
        X_sequences = self._create_sequences(X_scaled)
        
        encoded = self.encoder.predict(X_sequences, verbose=0)
        return encoded
    
    def reconstruct_sequences(self, X: np.ndarray) -> np.ndarray:
        """Reconstruct input sequences"""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        X_scaled = self.scaler.transform(X)
        X_sequences = self._create_sequences(X_scaled)
        
        reconstructions = self.autoencoder.predict(X_sequences, verbose=0)
        
        return self.scaler.inverse_transform(reconstructions.reshape(-1, self.feature_dim)).reshape(
            reconstructions.shape
        )
    
    def predict_next_steps(self, X: np.ndarray, steps: int = 5) -> np.ndarray:
        """
        Predict future behavior based on recent patterns
        
        Args:
            X: Recent behavior data
            steps: Number of future steps to predict
            
        Returns:
            Predicted future behavior
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        X_scaled = self.scaler.transform(X)
        
        # Get the last sequence
        if len(X_scaled) >= self.sequence_length:
            last_sequence = X_scaled[-self.sequence_length:].reshape(1, self.sequence_length, self.feature_dim)
        else:
            # Pad with zeros if not enough data
            padded = np.zeros((self.sequence_length, self.feature_dim))
            padded[-len(X_scaled):] = X_scaled
            last_sequence = padded.reshape(1, self.sequence_length, self.feature_dim)
        
        # Get encoded representation
        encoded = self.encoder.predict(last_sequence, verbose=0)
        
        # Generate future sequences
        predictions = []
        current_encoded = encoded
        
        for _ in range(steps):
            # Decode to get next sequence
            decoded = self.decoder.predict(current_encoded, verbose=0)
            predictions.append(decoded[0, -1, :])  # Get last timestep
            
            # Update encoded representation (simple approach)
            # In practice, you might want a more sophisticated prediction mechanism
            current_encoded = encoded  # Keep using the same encoded representation
        
        predictions = np.array(predictions)
        
        # Inverse transform predictions
        predictions_scaled = self.scaler.inverse_transform(predictions)
        
        return predictions_scaled
    
    def save_model(self, filepath: str):
        """Save LSTM Auto-Encoder model and scaler"""
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        model_data = {
            'autoencoder': self.autoencoder,
            'encoder': self.encoder,
            'decoder': self.decoder,
            'scaler': self.scaler,
            'sequence_length': self.sequence_length,
            'feature_dim': self.feature_dim,
            'lstm_units': self.lstm_units,
            'dropout_rate': self.dropout_rate,
            'anomaly_threshold': self.anomaly_threshold
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"LSTM Auto-Encoder model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load LSTM Auto-Encoder model and scaler"""
        model_data = joblib.load(filepath)
        
        self.autoencoder = model_data['autoencoder']
        self.encoder = model_data['encoder']
        self.decoder = model_data['decoder']
        self.scaler = model_data['scaler']
        self.sequence_length = model_data['sequence_length']
        self.feature_dim = model_data['feature_dim']
        self.lstm_units = model_data['lstm_units']
        self.dropout_rate = model_data['dropout_rate']
        self.anomaly_threshold = model_data['anomaly_threshold']
        self.is_trained = True
        
        logger.info(f"LSTM Auto-Encoder model loaded from {filepath}")
    
    def get_model_summary(self) -> Dict:
        """Get model architecture summary"""
        if self.autoencoder is None:
            return {"error": "Model not built"}
        
        return {
            "sequence_length": self.sequence_length,
            "feature_dim": self.feature_dim,
            "lstm_units": self.lstm_units,
            "dropout_rate": self.dropout_rate,
            "total_params": self.autoencoder.count_params(),
            "is_trained": self.is_trained
        }


class TemporalAnomalyDetector:
    """
    High-level temporal anomaly detector that uses LSTM Auto-Encoder
    for user behavior sequence analysis
    """
    
    def __init__(self, sequence_length: int = 20, feature_dim: int = 10):
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.lstm_autoencoder = LSTMAutoEncoder(sequence_length, feature_dim)
        self.user_sequences = {}  # Store user behavior sequences
        
    def add_user_event(self, user_id: str, event_features: np.ndarray):
        """Add event to user's behavior sequence"""
        if user_id not in self.user_sequences:
            self.user_sequences[user_id] = deque(maxlen=self.sequence_length * 2)
        
        self.user_sequences[user_id].append(event_features)
    
    def get_user_sequence(self, user_id: str) -> Optional[np.ndarray]:
        """Get user's recent behavior sequence"""
        if user_id not in self.user_sequences or len(self.user_sequences[user_id]) < self.sequence_length:
            return None
        
        sequence = list(self.user_sequences[user_id])[-self.sequence_length:]
        return np.array(sequence)
    
    def detect_temporal_anomaly(self, user_id: str) -> Tuple[float, bool]:
        """
        Detect temporal anomalies in user behavior
        
        Args:
            user_id: User identifier
            
        Returns:
            Tuple of (anomaly_score, is_anomaly)
        """
        sequence = self.get_user_sequence(user_id)
        if sequence is None:
            return 0.0, False
        
        if not self.lstm_autoencoder.is_trained:
            return 0.0, False
        
        # Reshape for prediction (add batch dimension)
        sequence_batch = sequence.reshape(1, -1)
        
        scores = self.lstm_autoencoder.predict_anomaly_scores(sequence_batch)
        score = scores[0]
        
        # Simple threshold-based anomaly detection
        is_anomaly = score > 70.0  # Threshold for temporal anomalies
        
        return score, is_anomaly


if __name__ == "__main__":
    # Test LSTM Auto-Encoder
    logger.info("Testing LSTM Auto-Encoder...")
    
    # Generate synthetic temporal data
    np.random.seed(42)
    n_samples = 1000
    n_features = 8
    
    # Create time series with some patterns
    time_series = []
    for i in range(n_samples):
        # Normal pattern with some noise
        base_pattern = np.sin(i * 0.1) + np.cos(i * 0.05)
        noise = np.random.normal(0, 0.1, n_features)
        
        # Add some anomalies
        if i % 100 == 0:  # Periodic anomalies
            noise += np.random.normal(0, 0.5, n_features)
        
        sample = base_pattern + noise
        time_series.append(sample)
    
    X = np.array(time_series)
    
    # Create and train LSTM Auto-Encoder
    lstm_ae = LSTMAutoEncoder(sequence_length=15, feature_dim=n_features, lstm_units=[32, 16])
    history = lstm_ae.train(X, epochs=50, batch_size=32, verbose=1)
    
    # Test anomaly detection
    scores, is_anomaly = lstm_ae.detect_anomalies(X)
    
    logger.info(f"Temporal anomaly detection results:")
    logger.info(f"  Total sequences: {len(scores)}")
    logger.info(f"  Detected anomalies: {np.sum(is_anomaly)}")
    logger.info(f"  Anomaly rate: {np.mean(is_anomaly) * 100:.1f}%")
    logger.info(f"  Score range: {np.min(scores):.2f} - {np.max(scores):.2f}")
    
    # Test temporal detector
    temporal_detector = TemporalAnomalyDetector(sequence_length=15, feature_dim=n_features)
    temporal_detector.lstm_autoencoder = lstm_ae  # Use trained model
    
    # Simulate user events
    for i in range(50):
        user_id = f"USER_{i % 10}"  # 10 different users
        event_features = X[i] if i < len(X) else np.random.normal(0, 1, n_features)
        temporal_detector.add_user_event(user_id, event_features)
    
    # Test temporal anomaly detection
    for user_id in list(temporal_detector.user_sequences.keys())[:3]:
        score, is_anomaly = temporal_detector.detect_temporal_anomaly(user_id)
        logger.info(f"User {user_id}: score={score:.2f}, anomaly={is_anomaly}")
    
    logger.info("✅ LSTM Auto-Encoder test completed successfully!")
