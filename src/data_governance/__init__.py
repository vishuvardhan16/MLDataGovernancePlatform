"""
Automating Data Governance: A Machine Learning Approach for Lineage Tracking,
Anomaly Detection, and Policy Enforcement in Enterprise Data Platforms.

This package implements an ML-driven framework for automated data governance
combining graph-based lineage inference, deep learning anomaly detection,
and adaptive policy enforcement.

Author: Data Governance Research Team
License: MIT
"""

__version__ = "1.0.0"
__author__ = "Data Governance Research Team"

from .lineage import LineageInferenceEngine, LineageGraph
from .anomaly import AnomalyDetectionEngine, DeepAutoencoder
from .policy import PolicyEnforcementEngine, GovernancePolicy
from .framework import GovernanceAutomationFramework

__all__ = [
    "LineageInferenceEngine",
    "LineageGraph",
    "AnomalyDetectionEngine", 
    "DeepAutoencoder",
    "PolicyEnforcementEngine",
    "GovernancePolicy",
    "GovernanceAutomationFramework",
]
