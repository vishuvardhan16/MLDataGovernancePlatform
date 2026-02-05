"""
Integrated Governance Automation Framework.

Combines lineage tracking, anomaly detection, and policy enforcement
into a unified system for enterprise data governance.

Reference: Section 4 - System Architecture and Implementation
"""

import logging
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import torch

from .config import FrameworkConfig, AnomalyType
from .lineage import LineageInferenceEngine, LineageGraph, LineageNode
from .anomaly import AnomalyDetectionEngine, AnomalyEvent
from .policy import PolicyEnforcementEngine, PolicyViolation, GovernancePolicy


logger = logging.getLogger(__name__)


@dataclass
class GovernanceEvent:
    """Represents a governance event processed by the framework."""
    
    event_id: str
    event_type: str  # 'data_operation', 'access', 'transformation', 'schema_change'
    timestamp: float
    source_system: str
    metadata: Dict
    
    # Processing results
    lineage_updated: bool = False
    anomaly_detected: bool = False
    violations: List[str] = None
    
    def __post_init__(self):
        if self.violations is None:
            self.violations = []


class GovernanceAutomationFramework:
    """
    Integrated framework for automated data governance.
    
    Combines:
    - Lineage Inference Engine (Section 3.2)
    - Anomaly Detection Engine (Section 3.3)
    - Policy Enforcement Engine (Section 3.4)
    
    Provides real-time governance monitoring and enforcement.
    """
    
    def __init__(self, config: Optional[FrameworkConfig] = None):
        """
        Initialize governance automation framework.
        
        Args:
            config: Framework configuration (uses defaults if None)
        """
        self.config = config or FrameworkConfig()
        
        # Set random seed for reproducibility
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        
        # Initialize components
        self.lineage_engine = LineageInferenceEngine(
            config=self.config.lineage,
            device=self.config.device
        )
        
        self.anomaly_engine = AnomalyDetectionEngine(
            config=self.config.anomaly,
            device=self.config.device
        )
        
        self.policy_engine = PolicyEnforcementEngine(
            config=self.config.policy
        )
        
        # Event processing history
        self.processed_events: List[GovernanceEvent] = []
        
        # Framework state
        self._initialized = False
        self._running = False
        
        # Create output directories
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.log_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info("Governance Automation Framework initialized")
    
    def initialize(
        self,
        training_data: np.ndarray,
        validation_data: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Initialize framework with training data.
        
        Args:
            training_data: Normal behavior samples for anomaly detection
            validation_data: Optional validation set
            
        Returns:
            Initialization results
        """
        logger.info("Initializing framework components...")
        
        results = {}
        
        # Train anomaly detection model
        logger.info("Training anomaly detection model...")
        anomaly_history = self.anomaly_engine.train(
            training_data=training_data,
            validation_data=validation_data,
            verbose=True
        )
        results['anomaly_training'] = {
            'final_loss': anomaly_history['train_loss'][-1],
            'threshold': self.anomaly_engine.threshold
        }
        
        self._initialized = True
        logger.info("Framework initialization complete")
        
        return results
    
    def process_event(
        self,
        event_type: str,
        source_system: str,
        metadata: Dict,
        event_id: Optional[str] = None
    ) -> GovernanceEvent:
        """
        Process a single governance event.
        
        Args:
            event_type: Type of event
            source_system: Source system identifier
            metadata: Event metadata
            event_id: Optional event identifier
            
        Returns:
            Processed governance event
        """
        if not self._initialized:
            raise RuntimeError("Framework must be initialized before processing events")
        
        event = GovernanceEvent(
            event_id=event_id or f"evt_{int(time.time() * 1000)}",
            event_type=event_type,
            timestamp=time.time(),
            source_system=source_system,
            metadata=metadata
        )
        
        # Step 1: Update lineage if applicable
        if event_type in ['data_operation', 'transformation']:
            self._update_lineage(event)
        
        # Step 2: Check for anomalies
        anomaly = self._detect_anomaly(event)
        
        # Step 3: Evaluate policies
        violations = self._evaluate_policies(event, anomaly)
        
        # Record results
        event.anomaly_detected = anomaly is not None
        event.violations = [v.violation_id for v in violations]
        
        self.processed_events.append(event)
        
        return event
    
    def _update_lineage(self, event: GovernanceEvent) -> None:
        """Update lineage graph based on event."""
        metadata = event.metadata
        
        # Add dataset if new
        if 'dataset_id' in metadata:
            dataset_id = metadata['dataset_id']
            
            if dataset_id not in self.lineage_engine.graph.nodes:
                self.lineage_engine.add_dataset(
                    dataset_id=dataset_id,
                    name=metadata.get('dataset_name', dataset_id),
                    metadata=metadata
                )
                event.lineage_updated = True
        
        # Add transformation if applicable
        if event.event_type == 'transformation' and 'transform_id' in metadata:
            self.lineage_engine.add_transformation(
                transform_id=metadata['transform_id'],
                name=metadata.get('transform_name', 'Unknown'),
                input_datasets=metadata.get('input_datasets', []),
                output_datasets=metadata.get('output_datasets', []),
                metadata=metadata
            )
            event.lineage_updated = True
    
    def _detect_anomaly(
        self, 
        event: GovernanceEvent
    ) -> Optional[AnomalyEvent]:
        """Detect anomalies in event."""
        # Build pipeline state from event metadata
        pipeline_state = {
            'execution_time': event.metadata.get('execution_time', 0),
            'start_hour': event.metadata.get('start_hour', 12),
            'day_of_week': event.metadata.get('day_of_week', 0),
            'rows_processed': event.metadata.get('rows_processed', 0),
            'bytes_processed': event.metadata.get('bytes_processed', 0),
            'num_columns': event.metadata.get('num_columns', 0),
            'schema_hash': event.metadata.get('schema_hash', ''),
            'num_reads': event.metadata.get('num_reads', 0),
            'num_writes': event.metadata.get('num_writes', 0),
            'unique_users': event.metadata.get('unique_users', 1),
            'error_count': event.metadata.get('error_count', 0),
            'warning_count': event.metadata.get('warning_count', 0),
            # Anomaly type indicators
            'schema_changed': event.metadata.get('schema_changed', False),
            'unauthorized_user': event.metadata.get('unauthorized_user', False),
            'off_schedule': event.metadata.get('off_schedule', False),
            'null_ratio': event.metadata.get('null_ratio', 0),
            'retention_exceeded': event.metadata.get('retention_exceeded', False),
            'pii_detected': event.metadata.get('pii_detected', False),
        }
        
        return self.anomaly_engine.detect(
            pipeline_state=pipeline_state,
            event_id=event.event_id
        )
    
    def _evaluate_policies(
        self,
        event: GovernanceEvent,
        anomaly: Optional[AnomalyEvent]
    ) -> List[PolicyViolation]:
        """Evaluate policies for event."""
        # Build context for policy evaluation
        context = {
            'event_id': event.event_id,
            'event_type': event.event_type,
            'source_system': event.source_system,
            **event.metadata
        }
        
        # Add lineage completeness if dataset
        if 'dataset_id' in event.metadata:
            dataset_id = event.metadata['dataset_id']
            context['lineage_completeness'] = \
                self.lineage_engine.graph.compute_lineage_completeness(dataset_id)
        
        return self.policy_engine.evaluate_policies(
            context=context,
            anomaly_event=anomaly
        )
    
    def process_batch(
        self,
        events: List[Dict]
    ) -> List[GovernanceEvent]:
        """
        Process a batch of governance events.
        
        Args:
            events: List of event dictionaries
            
        Returns:
            List of processed governance events
        """
        processed = []
        
        for event_data in events:
            event = self.process_event(
                event_type=event_data.get('event_type', 'data_operation'),
                source_system=event_data.get('source_system', 'unknown'),
                metadata=event_data.get('metadata', {}),
                event_id=event_data.get('event_id')
            )
            processed.append(event)
        
        return processed
    
    def infer_lineage(self) -> int:
        """
        Run lineage inference to discover implicit relationships.
        
        Returns:
            Number of inferred edges
        """
        # Train GNN if enough edges
        if len(self.lineage_engine.graph.edges) >= 10:
            self.lineage_engine.train_model(num_epochs=50)
        
        # Infer implicit lineage
        inferred = self.lineage_engine.infer_implicit_lineage()
        
        return len(inferred)
    
    def get_dataset_lineage(self, dataset_id: str) -> Dict:
        """Get full lineage for a dataset."""
        return self.lineage_engine.get_full_lineage(dataset_id)
    
    def get_governance_report(self) -> Dict:
        """
        Generate comprehensive governance report.
        
        Returns:
            Dictionary with governance statistics
        """
        # Lineage statistics
        lineage_stats = {
            'total_nodes': len(self.lineage_engine.graph.nodes),
            'total_edges': len(self.lineage_engine.graph.edges),
            'direct_edges': sum(
                1 for e in self.lineage_engine.graph.edges.values()
                if e.edge_type == 'direct'
            ),
            'inferred_edges': sum(
                1 for e in self.lineage_engine.graph.edges.values()
                if e.edge_type == 'inferred'
            )
        }
        
        # Anomaly statistics
        anomaly_stats = self.anomaly_engine.get_anomaly_statistics()
        
        # Policy statistics
        policy_stats = self.policy_engine.get_compliance_report()
        
        # Event processing statistics
        event_stats = {
            'total_events': len(self.processed_events),
            'events_with_anomalies': sum(
                1 for e in self.processed_events if e.anomaly_detected
            ),
            'events_with_violations': sum(
                1 for e in self.processed_events if e.violations
            ),
            'lineage_updates': sum(
                1 for e in self.processed_events if e.lineage_updated
            )
        }
        
        return {
            'lineage': lineage_stats,
            'anomaly': anomaly_stats,
            'policy': policy_stats,
            'events': event_stats,
            'timestamp': time.time()
        }
    
    def save_checkpoint(self, path: Optional[str] = None) -> str:
        """
        Save framework state to checkpoint.
        
        Args:
            path: Optional checkpoint path
            
        Returns:
            Path to saved checkpoint
        """
        path = path or f"{self.config.checkpoint_dir}/checkpoint_{int(time.time())}.pt"
        
        checkpoint = {
            'config': self.config,
            'anomaly_model_state': self.anomaly_engine.autoencoder.state_dict(),
            'anomaly_threshold': self.anomaly_engine.threshold,
            'lineage_model_state': self.lineage_engine.model.state_dict(),
            'processed_events_count': len(self.processed_events)
        }
        
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")
        
        return path
    
    def load_checkpoint(self, path: str) -> None:
        """
        Load framework state from checkpoint.
        
        Args:
            path: Path to checkpoint file
        """
        checkpoint = torch.load(path, map_location=self.config.device)
        
        self.anomaly_engine.autoencoder.load_state_dict(
            checkpoint['anomaly_model_state']
        )
        self.anomaly_engine.threshold = checkpoint['anomaly_threshold']
        self.anomaly_engine._trained = True
        
        self.lineage_engine.model.load_state_dict(
            checkpoint['lineage_model_state']
        )
        
        self._initialized = True
        
        logger.info(f"Checkpoint loaded from {path}")
    
    def export_report(self, path: Optional[str] = None) -> str:
        """
        Export governance report to JSON file.
        
        Args:
            path: Optional output path
            
        Returns:
            Path to exported report
        """
        path = path or f"{self.config.output_dir}/governance_report_{int(time.time())}.json"
        
        report = self.get_governance_report()
        
        with open(path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Report exported to {path}")
        
        return path
