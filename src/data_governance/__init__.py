"""
Automating Data Governance: A Machine Learning Approach for Lineage Tracking,
Anomaly Detection, and Policy Enforcement in Enterprise Data Platforms.

This package implements an ML-driven framework for automated data governance
combining graph-based lineage inference, deep learning anomaly detection,
and adaptive policy enforcement.

Author: Vishnuvardhan Reddy Kaithapuram
License: MIT
"""

__version__ = "1.0.0"
__author__ = "Vishnuvardhan Reddy Kaithapuram"

from .config import FrameworkConfig, LineageConfig, AnomalyConfig, PolicyConfig
from .lineage import LineageInferenceEngine, LineageGraph
from .anomaly import AnomalyDetectionEngine, DeepAutoencoder
from .policy import PolicyEnforcementEngine, GovernancePolicy
from .framework import GovernanceAutomationFramework
from .synthetic_data import SyntheticDataGenerator
from .evaluation import FrameworkEvaluator

__all__ = [
    # Configuration
    "FrameworkConfig",
    "LineageConfig",
    "AnomalyConfig",
    "PolicyConfig",
    # Core engines
    "LineageInferenceEngine",
    "LineageGraph",
    "AnomalyDetectionEngine", 
    "DeepAutoencoder",
    "PolicyEnforcementEngine",
    "GovernancePolicy",
    # Framework
    "GovernanceAutomationFramework",
    # Utilities
    "SyntheticDataGenerator",
    "FrameworkEvaluator",
]
