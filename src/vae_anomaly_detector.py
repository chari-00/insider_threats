"""
Variational Auto-Encoder (VAE) based anomaly detection for cyber threat detection
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
import logging
from typing import Tuple, Dict, List
import joblib
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VAELossLayer(layers.Layer):
    """Custom layer for VAE loss calculation"""
    
    def __init__(self, input_dim, beta=1.0, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.beta = beta
    
    def call(self, inputs):
        x, x_decoded_mean, z_mean, z_log_var = inputs
        
        # Reconstruction loss
        reconstruction_loss = keras.losses.mse(x, x_decoded_mean)
        reconstruction_loss *= self.input_dim
        
        # KL divergence loss
        kl_loss = 1 + z_log_var - keras.ops.square(z_mean) - keras.ops.exp(z_log_var)
        kl_loss = keras.ops.mean(kl_loss, axis=-1)
        kl_loss *= -0.5
        
        # Total loss
        vae_loss = keras.ops.mean(reconstruction_loss + self.beta * kl_loss)
        
        self.add_loss(vae_loss)
        return x_decoded_mean


class VAEAnomalyDetector:
    """
    Variational Auto-Encoder for anomaly detection in user behavior patterns
    """
    
    def __init__(self, input_dim: int = 20, latent_dim: int = 8, 
                 hidden_dims: List[int] = [16, 12], beta: float = 1.0):
        """
        Initialize VAE anomaly detector
        
        Args:
            input_dim: Number of input features
            latent_dim: Dimension of latent space
            hidden_dims: Hidden layer dimensions
            beta: KL divergence weight (beta-VAE)
        """
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.beta = beta
        
        self.scaler = StandardScaler()
        self.vae_model = None
        self.encoder = None
        self.decoder = None
        self.is_trained = False
        
        # Anomaly threshold (percentile of reconstruction error)
        self.anomaly_threshold = 95.0
        
        logger.info(f"VAE initialized: input_dim={input_dim}, latent_dim={latent_dim}")
    
    def _build_vae(self):
        """Build VAE architecture"""
        
        # Input layer
        inputs = keras.Input(shape=(self.input_dim,))
        
        # Encoder
        x = inputs
        for dim in self.hidden_dims:
            x = layers.Dense(dim, activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.2)(x)
        
        # Latent space (mean and log variance)
        z_mean = layers.Dense(self.latent_dim, name='z_mean')(x)
        z_log_var = layers.Dense(self.latent_dim, name='z_log_var')(x)
        
        # Reparameterization trick
        def sampling(args):
            z_mean, z_log_var = args
            batch = keras.ops.shape(z_mean)[0]
            dim = keras.ops.shape(z_mean)[1]
            epsilon = keras.random.normal(shape=(batch, dim))
            return z_mean + keras.ops.exp(0.5 * z_log_var) * epsilon
        
        z = layers.Lambda(sampling, output_shape=(self.latent_dim,), name='z')([z_mean, z_log_var])
        
        # Decoder
        x = z
        for dim in reversed(self.hidden_dims):
            x = layers.Dense(dim, activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.2)(x)
        
        outputs = layers.Dense(self.input_dim, activation='linear')(x)
        
        # Add VAE loss layer
        vae_outputs = VAELossLayer(self.input_dim, self.beta)([inputs, outputs, z_mean, z_log_var])
        
        # VAE model
        self.vae_model = keras.Model(inputs, vae_outputs, name='vae')
        
        # Encoder model
        self.encoder = keras.Model(inputs, [z_mean, z_log_var, z], name='encoder')
        
        # Decoder model
        latent_inputs = keras.Input(shape=(self.latent_dim,))
        x = latent_inputs
        for dim in reversed(self.hidden_dims):
            x = layers.Dense(dim, activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.2)(x)
        decoder_outputs = layers.Dense(self.input_dim, activation='linear')(x)
        self.decoder = keras.Model(latent_inputs, decoder_outputs, name='decoder')
        
        # Compile model
        self.vae_model.compile(optimizer='adam')
        
        logger.info("VAE architecture built successfully")
    
    def train(self, X: np.ndarray, epochs: int = 100, batch_size: int = 32, 
              validation_split: float = 0.2, verbose: int = 1):
        """
        Train VAE model
        
        Args:
            X: Training data
            epochs: Number of training epochs
            batch_size: Batch size
            validation_split: Validation split ratio
            verbose: Verbosity level
        """
        logger.info(f"Training VAE with {len(X)} samples...")
        
        # Scale data
        X_scaled = self.scaler.fit_transform(X)
        
        # Build model if not already built
        if self.vae_model is None:
            self._build_vae()
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=10, restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6
            )
        ]
        
        # Train model
        history = self.vae_model.fit(
            X_scaled, X_scaled,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=verbose
        )
        
        self.is_trained = True
        logger.info("VAE training completed")
        
        return history
    
    def predict_anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomaly scores using reconstruction error
        
        Args:
            X: Input data
            
        Returns:
            Anomaly scores (higher = more anomalous)
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        X_scaled = self.scaler.transform(X)
        
        # Get reconstructions
        reconstructions = self.vae_model.predict(X_scaled, verbose=0)
        
        # Calculate reconstruction error (MSE)
        reconstruction_errors = np.mean(np.square(X_scaled - reconstructions), axis=1)
        
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
    
    def get_latent_representation(self, X: np.ndarray) -> np.ndarray:
        """Get latent space representation of data"""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        X_scaled = self.scaler.transform(X)
        z_mean, z_log_var, z = self.encoder.predict(X_scaled, verbose=0)
        
        return z_mean
    
    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        """Reconstruct input data"""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        X_scaled = self.scaler.transform(X)
        reconstructions = self.vae_model.predict(X_scaled, verbose=0)
        
        return self.scaler.inverse_transform(reconstructions)
    
    def save_model(self, filepath: str):
        """Save VAE model and scaler"""
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        model_data = {
            'vae_model': self.vae_model,
            'encoder': self.encoder,
            'decoder': self.decoder,
            'scaler': self.scaler,
            'input_dim': self.input_dim,
            'latent_dim': self.latent_dim,
            'hidden_dims': self.hidden_dims,
            'beta': self.beta,
            'anomaly_threshold': self.anomaly_threshold
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"VAE model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load VAE model and scaler"""
        model_data = joblib.load(filepath)
        
        self.vae_model = model_data['vae_model']
        self.encoder = model_data['encoder']
        self.decoder = model_data['decoder']
        self.scaler = model_data['scaler']
        self.input_dim = model_data['input_dim']
        self.latent_dim = model_data['latent_dim']
        self.hidden_dims = model_data['hidden_dims']
        self.beta = model_data['beta']
        self.anomaly_threshold = model_data['anomaly_threshold']
        self.is_trained = True
        
        logger.info(f"VAE model loaded from {filepath}")
    
    def get_model_summary(self) -> Dict:
        """Get model architecture summary"""
        if self.vae_model is None:
            return {"error": "Model not built"}
        
        return {
            "input_dim": self.input_dim,
            "latent_dim": self.latent_dim,
            "hidden_dims": self.hidden_dims,
            "beta": self.beta,
            "total_params": self.vae_model.count_params(),
            "is_trained": self.is_trained
        }


if __name__ == "__main__":
    # Test VAE anomaly detector
    logger.info("Testing VAE Anomaly Detector...")
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_features = 15
    
    # Normal data
    normal_data = np.random.multivariate_normal(
        mean=np.zeros(n_features),
        cov=np.eye(n_features),
        size=int(n_samples * 0.9)
    )
    
    # Anomalous data
    anomaly_data = np.random.multivariate_normal(
        mean=np.ones(n_features) * 2,
        cov=np.eye(n_features) * 0.5,
        size=int(n_samples * 0.1)
    )
    
    X = np.vstack([normal_data, anomaly_data])
    np.random.shuffle(X)
    
    # Create and train VAE
    vae = VAEAnomalyDetector(input_dim=n_features, latent_dim=6, hidden_dims=[12, 8])
    history = vae.train(X, epochs=50, batch_size=32, verbose=1)
    
    # Test anomaly detection
    scores, is_anomaly = vae.detect_anomalies(X)
    
    logger.info(f"Anomaly detection results:")
    logger.info(f"  Total samples: {len(X)}")
    logger.info(f"  Detected anomalies: {np.sum(is_anomaly)}")
    logger.info(f"  Anomaly rate: {np.mean(is_anomaly) * 100:.1f}%")
    logger.info(f"  Score range: {np.min(scores):.2f} - {np.max(scores):.2f}")
    
    logger.info("✅ VAE test completed successfully!")
