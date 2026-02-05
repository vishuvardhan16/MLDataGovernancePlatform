"""
Anomaly Detection Engine for Data Governance.

Implements deep autoencoder-based anomaly detection for identifying
irregularities in data processing, access patterns, and transformations.

Reference: Section 3.3 of the paper - Anomaly Detection Engine
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest

from .config import AnomalyConfig, AnomalyType


logger = logging.getLogger(__name__)


@dataclass
class AnomalyEvent:
    """Represents a detected anomaly event."""
    
    event_id: str
    timestamp: float
    anomaly_type: AnomalyType
    anomaly_score: float
    reconstruction_error: float
    feature_vector: np.ndarray
    metadata: Dict
    
    @property
    def is_critical(self) -> bool:
        """Check if anomaly is critical (score > 0.9)."""
        return self.anomaly_score > 0.9


class DeepAutoencoder(nn.Module):
    """
    Deep Autoencoder for anomaly detection.
    
    Architecture from Section 3.3:
    - Encoder: 256 → 128 → 64 → 32 (latent)
    - Decoder: 32 → 64 → 128 → 256
    
    Anomalies are detected when reconstruction error exceeds threshold τ.
    """
    
    def __init__(
        self,
        input_dim: int = 256,
        encoder_dims: List[int] = None,
        latent_dim: int = 32,
        decoder_dims: List[int] = None,
        dropout: float = 0.2
    ):
        """
        Initialize Deep Autoencoder.
        
        Args:
            input_dim: Input feature dimension
            encoder_dims: Encoder hidden layer dimensions
            latent_dim: Latent space dimension
            decoder_dims: Decoder hidden layer dimensions
            dropout: Dropout rate
        """
        super().__init__()
        
        encoder_dims = encoder_dims or [256, 128, 64]
        decoder_dims = decoder_dims or [64, 128, 256]
        
        # Build encoder
        encoder_layers = []
        prev_dim = input_dim
        
        for dim in encoder_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = dim
        
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Build decoder
        decoder_layers = []
        prev_dim = latent_dim
        
        for dim in decoder_dims:
            decoder_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = dim
        
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space."""
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to reconstruction."""
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through autoencoder.
        
        Args:
            x: Input tensor (batch, input_dim)
            
        Returns:
            Tuple of (reconstruction, latent_representation)
        """
        z = self.encode(x)
        x_reconstructed = self.decode(z)
        return x_reconstructed, z
    
    def compute_reconstruction_error(
        self, 
        x: torch.Tensor, 
        x_reconstructed: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute reconstruction error (MSE).
        
        L_recon = ||x - x̂||²
        
        Args:
            x: Original input
            x_reconstructed: Reconstructed output
            
        Returns:
            Per-sample reconstruction error
        """
        return torch.mean((x - x_reconstructed) ** 2, dim=1)


class AnomalyDetectionEngine:
    """
    Engine for detecting anomalies in data pipeline operations.
    
    Combines deep autoencoder with statistical methods for robust
    anomaly detection across different types of irregularities.
    """
    
    def __init__(self, config: AnomalyConfig, device: str = 'cpu'):
        """
        Initialize anomaly detection engine.
        
        Args:
            config: Anomaly detection configuration
            device: Computation device
        """
        self.config = config
        self.device = device
        
        # Initialize autoencoder
        self.autoencoder = DeepAutoencoder(
            input_dim=config.input_dim,
            encoder_dims=config.encoder_dims,
            latent_dim=config.latent_dim,
            decoder_dims=config.decoder_dims,
            dropout=config.dropout
        ).to(device)
        
        # Feature scaler
        self.scaler = StandardScaler()
        
        # Anomaly threshold (set during training)
        self.threshold: Optional[float] = None
        
        # Training state
        self._trained = False
        self._last_train_time: Optional[float] = None
        
        # Detected anomalies history
        self.anomaly_history: List[AnomalyEvent] = []
    
    def extract_pipeline_features(
        self,
        pipeline_state: Dict
    ) -> np.ndarray:
        """
        Extract feature vector from pipeline state.
        
        Args:
            pipeline_state: Dictionary containing pipeline metrics
            
        Returns:
            Feature vector of shape (input_dim,)
        """
        features = np.zeros(self.config.input_dim)
        idx = 0
        
        # Temporal features
        if 'execution_time' in pipeline_state:
            features[idx] = np.log1p(pipeline_state['execution_time'])
            idx += 1
        
        if 'start_hour' in pipeline_state:
            # Cyclical encoding for hour
            hour = pipeline_state['start_hour']
            features[idx] = np.sin(2 * np.pi * hour / 24)
            features[idx + 1] = np.cos(2 * np.pi * hour / 24)
            idx += 2
        
        if 'day_of_week' in pipeline_state:
            # Cyclical encoding for day
            day = pipeline_state['day_of_week']
            features[idx] = np.sin(2 * np.pi * day / 7)
            features[idx + 1] = np.cos(2 * np.pi * day / 7)
            idx += 2
        
        # Volume features
        if 'rows_processed' in pipeline_state:
            features[idx] = np.log1p(pipeline_state['rows_processed'])
            idx += 1
        
        if 'bytes_processed' in pipeline_state:
            features[idx] = np.log1p(pipeline_state['bytes_processed'])
            idx += 1
        
        # Schema features
        if 'num_columns' in pipeline_state:
            features[idx] = pipeline_state['num_columns'] / 100
            idx += 1
        
        if 'schema_hash' in pipeline_state:
            # Use hash as categorical feature
            features[idx] = hash(pipeline_state['schema_hash']) % 1000 / 1000
            idx += 1
        
        # Access features
        if 'num_reads' in pipeline_state:
            features[idx] = np.log1p(pipeline_state['num_reads'])
            idx += 1
        
        if 'num_writes' in pipeline_state:
            features[idx] = np.log1p(pipeline_state['num_writes'])
            idx += 1
        
        if 'unique_users' in pipeline_state:
            features[idx] = np.log1p(pipeline_state['unique_users'])
            idx += 1
        
        # Error features
        if 'error_count' in pipeline_state:
            features[idx] = np.log1p(pipeline_state['error_count'])
            idx += 1
        
        if 'warning_count' in pipeline_state:
            features[idx] = np.log1p(pipeline_state['warning_count'])
            idx += 1
        
        # Fill remaining dimensions with noise
        remaining = self.config.input_dim - idx
        if remaining > 0:
            features[idx:] = np.random.randn(remaining) * 0.01
        
        return features.astype(np.float32)
    
    def train(
        self,
        training_data: np.ndarray,
        validation_data: Optional[np.ndarray] = None,
        num_epochs: Optional[int] = None,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Train the autoencoder on normal behavior data.
        
        Args:
            training_data: Normal behavior samples (n_samples, input_dim)
            validation_data: Optional validation set
            num_epochs: Number of training epochs
            verbose: Whether to print progress
            
        Returns:
            Training history
        """
        num_epochs = num_epochs or self.config.num_epochs
        
        # Fit scaler
        self.scaler.fit(training_data)
        scaled_data = self.scaler.transform(training_data)
        
        # Create data loader
        tensor_data = torch.tensor(scaled_data, dtype=torch.float32)
        dataset = TensorDataset(tensor_data)
        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )
        
        # Optimizer
        optimizer = torch.optim.Adam(
            self.autoencoder.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.l2_lambda
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )
        
        history = {'train_loss': [], 'val_loss': []}
        
        self.autoencoder.train()
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch in loader:
                x = batch[0].to(self.device)
                
                optimizer.zero_grad()
                
                x_recon, z = self.autoencoder(x)
                
                # Reconstruction loss
                recon_loss = F.mse_loss(x_recon, x)
                
                # L2 regularization on latent space
                reg_loss = self.config.l2_lambda * torch.mean(z ** 2)
                
                loss = recon_loss + reg_loss
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            history['train_loss'].append(avg_loss)
            
            # Validation
            if validation_data is not None:
                val_loss = self._compute_validation_loss(validation_data)
                history['val_loss'].append(val_loss)
                scheduler.step(val_loss)
            else:
                scheduler.step(avg_loss)
            
            if verbose and (epoch + 1) % 20 == 0:
                msg = f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.6f}"
                if validation_data is not None:
                    msg += f", Val Loss: {history['val_loss'][-1]:.6f}"
                logger.info(msg)
        
        # Set anomaly threshold based on training data reconstruction errors
        self._set_threshold(scaled_data)
        
        self._trained = True
        self._last_train_time = time.time()
        
        logger.info(f"Training complete. Anomaly threshold: {self.threshold:.6f}")
        
        return history
    
    def _compute_validation_loss(self, validation_data: np.ndarray) -> float:
        """Compute validation loss."""
        self.autoencoder.eval()
        
        scaled_data = self.scaler.transform(validation_data)
        x = torch.tensor(scaled_data, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            x_recon, _ = self.autoencoder(x)
            loss = F.mse_loss(x_recon, x)
        
        self.autoencoder.train()
        return loss.item()
    
    def _set_threshold(self, training_data: np.ndarray) -> None:
        """
        Set anomaly threshold based on training data.
        
        Threshold τ is set at the specified percentile of reconstruction errors.
        """
        self.autoencoder.eval()
        
        x = torch.tensor(training_data, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            x_recon, _ = self.autoencoder(x)
            errors = self.autoencoder.compute_reconstruction_error(x, x_recon)
            errors = errors.cpu().numpy()
        
        self.threshold = np.percentile(errors, self.config.threshold_percentile)
    
    def detect(
        self,
        pipeline_state: Dict,
        event_id: Optional[str] = None
    ) -> Optional[AnomalyEvent]:
        """
        Detect anomaly in a single pipeline state.
        
        Args:
            pipeline_state: Pipeline state dictionary
            event_id: Optional event identifier
            
        Returns:
            AnomalyEvent if anomaly detected, None otherwise
        """
        if not self._trained:
            raise RuntimeError("Model must be trained before detection")
        
        # Extract and scale features
        features = self.extract_pipeline_features(pipeline_state)
        scaled_features = self.scaler.transform(features.reshape(1, -1))
        
        # Compute reconstruction error
        self.autoencoder.eval()
        x = torch.tensor(scaled_features, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            x_recon, _ = self.autoencoder(x)
            error = self.autoencoder.compute_reconstruction_error(x, x_recon)
            error = error.cpu().numpy()[0]
        
        # Check threshold
        if error > self.threshold:
            # Compute anomaly score (normalized)
            anomaly_score = min(1.0, error / (2 * self.threshold))
            
            # Determine anomaly type
            anomaly_type = self._classify_anomaly_type(pipeline_state, features)
            
            event = AnomalyEvent(
                event_id=event_id or f"anomaly_{int(time.time() * 1000)}",
                timestamp=time.time(),
                anomaly_type=anomaly_type,
                anomaly_score=anomaly_score,
                reconstruction_error=error,
                feature_vector=features,
                metadata=pipeline_state
            )
            
            self.anomaly_history.append(event)
            
            logger.warning(
                f"Anomaly detected: {anomaly_type.value}, "
                f"score={anomaly_score:.4f}, error={error:.6f}"
            )
            
            return event
        
        return None
    
    def detect_batch(
        self,
        pipeline_states: List[Dict]
    ) -> List[AnomalyEvent]:
        """
        Detect anomalies in a batch of pipeline states.
        
        Args:
            pipeline_states: List of pipeline state dictionaries
            
        Returns:
            List of detected anomaly events
        """
        if not self._trained:
            raise RuntimeError("Model must be trained before detection")
        
        # Extract features
        features = np.array([
            self.extract_pipeline_features(state)
            for state in pipeline_states
        ])
        
        scaled_features = self.scaler.transform(features)
        
        # Compute reconstruction errors
        self.autoencoder.eval()
        x = torch.tensor(scaled_features, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            x_recon, _ = self.autoencoder(x)
            errors = self.autoencoder.compute_reconstruction_error(x, x_recon)
            errors = errors.cpu().numpy()
        
        # Detect anomalies
        anomalies = []
        
        for i, (error, state) in enumerate(zip(errors, pipeline_states)):
            if error > self.threshold:
                anomaly_score = min(1.0, error / (2 * self.threshold))
                anomaly_type = self._classify_anomaly_type(state, features[i])
                
                event = AnomalyEvent(
                    event_id=f"anomaly_{int(time.time() * 1000)}_{i}",
                    timestamp=time.time(),
                    anomaly_type=anomaly_type,
                    anomaly_score=anomaly_score,
                    reconstruction_error=error,
                    feature_vector=features[i],
                    metadata=state
                )
                
                anomalies.append(event)
                self.anomaly_history.append(event)
        
        if anomalies:
            logger.warning(f"Detected {len(anomalies)} anomalies in batch")
        
        return anomalies
    
    def _classify_anomaly_type(
        self,
        pipeline_state: Dict,
        features: np.ndarray
    ) -> AnomalyType:
        """
        Classify the type of anomaly based on pipeline state.
        
        Args:
            pipeline_state: Pipeline state dictionary
            features: Extracted feature vector
            
        Returns:
            Classified anomaly type
        """
        # Schema drift detection
        if pipeline_state.get('schema_changed', False):
            return AnomalyType.SCHEMA_DRIFT
        
        # Unauthorized access detection
        if pipeline_state.get('unauthorized_user', False):
            return AnomalyType.UNAUTHORIZED_ACCESS
        
        # Unexpected schedule detection
        if pipeline_state.get('off_schedule', False):
            return AnomalyType.UNEXPECTED_SCHEDULE
        
        # Data quality issues
        if pipeline_state.get('null_ratio', 0) > 0.5:
            return AnomalyType.DATA_QUALITY
        
        # Retention violation
        if pipeline_state.get('retention_exceeded', False):
            return AnomalyType.RETENTION_VIOLATION
        
        # PII exposure
        if pipeline_state.get('pii_detected', False):
            return AnomalyType.PII_EXPOSURE
        
        # Default to data quality
        return AnomalyType.DATA_QUALITY
    
    def get_anomaly_statistics(self) -> Dict:
        """Get statistics about detected anomalies."""
        if not self.anomaly_history:
            return {'total_anomalies': 0}
        
        scores = [a.anomaly_score for a in self.anomaly_history]
        types = [a.anomaly_type.value for a in self.anomaly_history]
        
        type_counts = {}
        for t in types:
            type_counts[t] = type_counts.get(t, 0) + 1
        
        return {
            'total_anomalies': len(self.anomaly_history),
            'critical_anomalies': sum(1 for a in self.anomaly_history if a.is_critical),
            'mean_score': np.mean(scores),
            'max_score': np.max(scores),
            'type_distribution': type_counts
        }
