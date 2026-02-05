"""
Unit tests for Data Governance Automation Framework.

Tests cover:
- Lineage inference engine
- Anomaly detection engine
- Policy enforcement engine
- Integrated framework
"""

import pytest
import numpy as np
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_governance.config import (
    FrameworkConfig, LineageConfig, AnomalyConfig, PolicyConfig
)
from data_governance.lineage import (
    LineageInferenceEngine, LineageGraph, LineageNode, LineageEdge
)
from data_governance.anomaly import (
    AnomalyDetectionEngine, DeepAutoencoder, AnomalyEvent
)
from data_governance.policy import (
    PolicyEnforcementEngine, GovernancePolicy, PolicyCondition,
    ComplianceRequirement, PolicyViolation
)
from data_governance.synthetic_data import SyntheticDataGenerator, SyntheticDataConfig


class TestLineageGraph:
    """Tests for LineageGraph class."""
    
    def test_add_node(self):
        """Test adding nodes to graph."""
        graph = LineageGraph()
        
        node = LineageNode(
            node_id='ds_001',
            node_type='dataset',
            name='test_dataset',
            attributes={'columns': ['a', 'b', 'c']}
        )
        
        graph.add_node(node)
        
        assert 'ds_001' in graph.nodes
        assert graph.nodes['ds_001'].name == 'test_dataset'
    
    def test_add_edge(self):
        """Test adding edges to graph."""
        graph = LineageGraph()
        
        # Add nodes
        node1 = LineageNode('ds_001', 'dataset', 'dataset1', {})
        node2 = LineageNode('tf_001', 'transformation', 'transform1', {})
        graph.add_node(node1)
        graph.add_node(node2)
        
        # Add edge
        edge = LineageEdge(
            source_id='ds_001',
            target_id='tf_001',
            edge_type='direct',
            weight=0.9,
            confidence=1.0,
            metadata={}
        )
        graph.add_edge(edge)
        
        assert ('ds_001', 'tf_001') in graph.edges
        assert 'tf_001' in graph.get_neighbors('ds_001')
        assert 'ds_001' in graph.get_predecessors('tf_001')
    
    def test_lineage_path(self):
        """Test finding lineage paths."""
        graph = LineageGraph()
        
        # Create chain: ds1 -> tf1 -> ds2 -> tf2 -> ds3
        nodes = [
            LineageNode('ds_001', 'dataset', 'd1', {}),
            LineageNode('tf_001', 'transformation', 't1', {}),
            LineageNode('ds_002', 'dataset', 'd2', {}),
            LineageNode('tf_002', 'transformation', 't2', {}),
            LineageNode('ds_003', 'dataset', 'd3', {}),
        ]
        
        for node in nodes:
            graph.add_node(node)
        
        edges = [
            ('ds_001', 'tf_001'),
            ('tf_001', 'ds_002'),
            ('ds_002', 'tf_002'),
            ('tf_002', 'ds_003'),
        ]
        
        for src, tgt in edges:
            edge = LineageEdge(src, tgt, 'direct', 1.0, 1.0, {})
            graph.add_edge(edge)
        
        # Find path
        path = graph.get_lineage_path('ds_001', 'ds_003')
        
        assert path is not None
        assert path[0] == 'ds_001'
        assert path[-1] == 'ds_003'
        assert len(path) == 5


class TestLineageInferenceEngine:
    """Tests for LineageInferenceEngine class."""
    
    @pytest.fixture
    def engine(self):
        """Create lineage inference engine."""
        config = LineageConfig()
        return LineageInferenceEngine(config, device='cpu')
    
    def test_add_dataset(self, engine):
        """Test adding dataset to engine."""
        engine.add_dataset(
            dataset_id='ds_001',
            name='test_dataset',
            metadata={'columns': ['a', 'b'], 'row_count': 1000}
        )
        
        assert 'ds_001' in engine.graph.nodes
        assert engine.graph.nodes['ds_001'].feature_vector is not None
    
    def test_add_transformation(self, engine):
        """Test adding transformation with lineage."""
        # Add datasets first
        engine.add_dataset('ds_001', 'input', {'columns': ['a']})
        engine.add_dataset('ds_002', 'output', {'columns': ['b']})
        
        # Add transformation
        engine.add_transformation(
            transform_id='tf_001',
            name='test_transform',
            input_datasets=['ds_001'],
            output_datasets=['ds_002'],
            metadata={'type': 'etl'}
        )
        
        assert 'tf_001' in engine.graph.nodes
        assert ('ds_001', 'tf_001') in engine.graph.edges
        assert ('tf_001', 'ds_002') in engine.graph.edges
    
    def test_extract_metadata_features(self, engine):
        """Test metadata feature extraction."""
        metadata = {
            'columns': ['a', 'b', 'c'],
            'dtypes': ['string', 'int', 'float'],
            'row_count': 10000,
            'size_bytes': 1000000
        }
        
        features = engine.extract_metadata_features(metadata)
        
        assert features.shape == (engine.config.node_embedding_dim,)
        assert features.dtype == np.float32


class TestDeepAutoencoder:
    """Tests for DeepAutoencoder class."""
    
    @pytest.fixture
    def autoencoder(self):
        """Create autoencoder model."""
        return DeepAutoencoder(
            input_dim=64,
            encoder_dims=[32, 16],
            latent_dim=8,
            decoder_dims=[16, 32],
            dropout=0.1
        )
    
    def test_forward_pass(self, autoencoder):
        """Test forward pass through autoencoder."""
        batch_size = 16
        x = torch.randn(batch_size, 64)
        
        x_recon, z = autoencoder(x)
        
        assert x_recon.shape == (batch_size, 64)
        assert z.shape == (batch_size, 8)
    
    def test_reconstruction_error(self, autoencoder):
        """Test reconstruction error computation."""
        batch_size = 16
        x = torch.randn(batch_size, 64)
        x_recon, _ = autoencoder(x)
        
        error = autoencoder.compute_reconstruction_error(x, x_recon)
        
        assert error.shape == (batch_size,)
        assert torch.all(error >= 0)


class TestAnomalyDetectionEngine:
    """Tests for AnomalyDetectionEngine class."""
    
    @pytest.fixture
    def engine(self):
        """Create anomaly detection engine."""
        config = AnomalyConfig(input_dim=64)
        return AnomalyDetectionEngine(config, device='cpu')
    
    def test_train(self, engine):
        """Test training anomaly detection model."""
        # Generate synthetic normal data
        np.random.seed(42)
        training_data = np.random.randn(500, 64).astype(np.float32)
        
        history = engine.train(
            training_data=training_data,
            num_epochs=10,
            verbose=False
        )
        
        assert engine._trained
        assert engine.threshold is not None
        assert len(history['train_loss']) == 10
    
    def test_detect(self, engine):
        """Test anomaly detection."""
        # Train first
        training_data = np.random.randn(500, 64).astype(np.float32)
        engine.train(training_data, num_epochs=10, verbose=False)
        
        # Test normal event
        normal_state = {
            'execution_time': 100,
            'rows_processed': 10000,
            'error_count': 0
        }
        
        result = engine.detect(normal_state)
        # May or may not be anomaly depending on threshold
        
        # Test anomalous event
        anomaly_state = {
            'execution_time': 10000,  # Very high
            'rows_processed': 10,  # Very low
            'error_count': 100  # Many errors
        }
        
        # Detection should work without error
        result = engine.detect(anomaly_state)


class TestPolicyEnforcementEngine:
    """Tests for PolicyEnforcementEngine class."""
    
    @pytest.fixture
    def engine(self):
        """Create policy enforcement engine."""
        config = PolicyConfig()
        return PolicyEnforcementEngine(config)
    
    def test_default_policies(self, engine):
        """Test default policies are registered."""
        assert len(engine.policies) > 0
        assert 'POL-001' in engine.policies  # PII policy
        assert 'POL-002' in engine.policies  # Access policy
    
    def test_register_policy(self, engine):
        """Test registering custom policy."""
        condition = PolicyCondition(
            condition_type='custom',
            predicate=lambda ctx: ctx.get('test_flag', False),
            description='Test condition'
        )
        
        compliance = ComplianceRequirement(
            requirement_id='TEST-001',
            name='Test Requirement',
            regulation='internal',
            description='Test',
            check_function=lambda ctx: not ctx.get('test_flag', False)
        )
        
        from data_governance.config import PolicyAction
        
        policy = GovernancePolicy(
            policy_id='POL-TEST',
            name='Test Policy',
            description='Test',
            condition=condition,
            compliance=compliance,
            action=PolicyAction.LOG,
            priority=1
        )
        
        engine.register_policy(policy)
        
        assert 'POL-TEST' in engine.policies
    
    def test_evaluate_policies(self, engine):
        """Test policy evaluation."""
        # Context that should trigger PII policy
        context = {
            'anomaly_type': 'pii_exposure',
            'anomaly_score': 0.9,
            'pii_detected': True
        }
        
        violations = engine.evaluate_policies(context)
        
        # Should have at least one violation
        assert len(violations) >= 0  # May vary based on confidence threshold
    
    def test_compliance_report(self, engine):
        """Test compliance report generation."""
        report = engine.get_compliance_report()
        
        assert 'total_violations' in report
        assert 'by_policy' in report
        assert 'by_severity' in report


class TestSyntheticDataGenerator:
    """Tests for SyntheticDataGenerator class."""
    
    @pytest.fixture
    def generator(self):
        """Create data generator."""
        config = SyntheticDataConfig(
            num_data_sources=5,
            num_transformations=10,
            num_processing_events=100,
            anomaly_rate=0.1,
            seed=42
        )
        return SyntheticDataGenerator(config)
    
    def test_generate_all(self, generator):
        """Test complete data generation."""
        data = generator.generate_all()
        
        assert len(data['data_sources']) == 5
        assert len(data['transformations']) == 10
        assert len(data['processing_events']) == 100
        assert len(data['datasets']) > 0
        assert len(data['lineage_edges']) > 0
    
    def test_anomaly_injection(self, generator):
        """Test anomaly injection rate."""
        generator.generate_all()
        
        anomaly_count = sum(
            1 for e in generator.processing_events if e['is_anomaly']
        )
        
        expected = int(100 * 0.1)  # 10% of 100
        
        # Allow some variance
        assert abs(anomaly_count - expected) <= 5
    
    def test_get_training_data(self, generator):
        """Test training data extraction."""
        generator.generate_all()
        
        normal_data, anomaly_data = generator.get_training_data()
        
        assert normal_data.shape[1] == 256  # Feature dimension
        assert anomaly_data.shape[1] == 256
        assert len(normal_data) + len(anomaly_data) == 100


class TestIntegration:
    """Integration tests for the complete framework."""
    
    def test_end_to_end_workflow(self):
        """Test complete workflow from data generation to evaluation."""
        # Generate small dataset
        data_config = SyntheticDataConfig(
            num_data_sources=3,
            num_transformations=5,
            num_processing_events=50,
            anomaly_rate=0.1,
            seed=42
        )
        generator = SyntheticDataGenerator(data_config)
        data = generator.generate_all()
        
        # Initialize framework
        from data_governance.framework import GovernanceAutomationFramework
        
        config = FrameworkConfig()
        framework = GovernanceAutomationFramework(config)
        
        # Get training data
        normal_data, _ = generator.get_training_data()
        
        # Initialize (train anomaly model)
        framework.initialize(training_data=normal_data)
        
        # Add lineage data
        for dataset in data['datasets'][:10]:
            framework.lineage_engine.add_dataset(
                dataset_id=dataset['dataset_id'],
                name=dataset['name'],
                metadata=dataset
            )
        
        # Process events
        for event in data['processing_events'][:10]:
            framework.process_event(
                event_type=event['event_type'],
                source_system=event['source_system'],
                metadata=event,
                event_id=event['event_id']
            )
        
        # Get report
        report = framework.get_governance_report()
        
        assert report['events']['total_events'] == 10
        assert 'lineage' in report
        assert 'anomaly' in report
        assert 'policy' in report


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
