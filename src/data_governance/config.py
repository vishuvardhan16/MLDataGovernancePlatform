"""
Configuration module for Data Governance Automation Framework.

Contains hyperparameters and settings for lineage tracking, anomaly detection,
and policy enforcement components.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum
import torch


class AnomalyType(Enum):
    """Types of anomalies detected by the system."""
    SCHEMA_DRIFT = "schema_drift"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    UNEXPECTED_SCHEDULE = "unexpected_schedule"
    DATA_QUALITY = "data_quality"
    RETENTION_VIOLATION = "retention_violation"
    PII_EXPOSURE = "pii_exposure"


class PolicyAction(Enum):
    """Actions taken upon policy violation."""
    ALERT = "alert"
    LOG = "log"
    BLOCK = "block"
    QUARANTINE = "quarantine"
    NOTIFY_STEWARD = "notify_steward"


@dataclass
class LineageConfig:
    """Configuration for lineage inference engine."""
    
    # Graph Neural Network parameters
    node_embedding_dim: int = 128
    hidden_dim: int = 256
    output_dim: int = 128
    num_gnn_layers: int = 3
    dropout: float = 0.1
    
    # Edge weight threshold for lineage path extraction
    edge_weight_threshold: float = 0.5
    
    # Metadata similarity parameters
    semantic_similarity_weight: float = 0.6
    structural_similarity_weight: float = 0.4
    
    # Graph construction
    max_neighbors: int = 50
    use_attention: bool = True
    attention_heads: int = 4


@dataclass
class AnomalyConfig:
    """Configuration for anomaly detection engine."""
    
    # Autoencoder architecture
    input_dim: int = 256
    encoder_dims: List[int] = field(default_factory=lambda: [256, 128, 64])
    latent_dim: int = 32
    decoder_dims: List[int] = field(default_factory=lambda: [64, 128, 256])
    
    # Training parameters
    learning_rate: float = 0.001
    batch_size: int = 128
    num_epochs: int = 100
    dropout: float = 0.2
    
    # Regularization
    l2_lambda: float = 1e-5
    
    # Anomaly threshold (percentile of reconstruction error)
    threshold_percentile: float = 95.0
    
    # Retraining schedule (days)
    retrain_interval_days: int = 7


@dataclass
class PolicyConfig:
    """Configuration for policy enforcement engine."""
    
    # Evaluation interval (seconds)
    evaluation_interval: int = 300  # 5 minutes
    
    # Alert thresholds
    anomaly_score_threshold: float = 0.8
    lineage_completeness_threshold: float = 0.7
    
    # False positive reduction
    min_confidence_for_action: float = 0.85
    
    # Priority levels
    default_priority: int = 5
    max_priority: int = 10


@dataclass
class FrameworkConfig:
    """Master configuration for the governance automation framework."""
    
    lineage: LineageConfig = field(default_factory=LineageConfig)
    anomaly: AnomalyConfig = field(default_factory=AnomalyConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    
    # General settings
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    log_level: str = "INFO"
    
    # Data paths
    data_dir: str = "data"
    output_dir: str = "outputs"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    
    # Experiment tracking
    experiment_name: str = "governance_automation_v1"
