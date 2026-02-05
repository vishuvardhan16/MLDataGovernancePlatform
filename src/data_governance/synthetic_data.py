"""
Synthetic Data Generator for Governance Framework Evaluation.

Generates realistic enterprise-like data for testing lineage tracking,
anomaly detection, and policy enforcement components.

Reference: Section 5.1 - Evaluation Datasets and Setup
"""

import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import hashlib
import random

import numpy as np
from datetime import datetime, timedelta


logger = logging.getLogger(__name__)


@dataclass
class SyntheticDataConfig:
    """Configuration for synthetic data generation."""
    
    # Dataset parameters
    num_data_sources: int = 20
    num_transformations: int = 100
    num_lineage_chains: int = 80
    num_processing_events: int = 10000
    
    # Anomaly injection
    anomaly_rate: float = 0.05  # 5% anomalies
    
    # Time range (6 months)
    start_date: datetime = None
    end_date: datetime = None
    
    # Random seed
    seed: int = 42
    
    def __post_init__(self):
        if self.start_date is None:
            self.end_date = datetime.now()
            self.start_date = self.end_date - timedelta(days=180)


class SyntheticDataGenerator:
    """
    Generator for synthetic enterprise data governance datasets.
    
    Creates realistic data including:
    - Data sources and datasets
    - Transformation workflows
    - Lineage relationships
    - Normal and anomalous processing events
    """
    
    # Data source types
    SOURCE_TYPES = [
        'relational_db', 'data_lake', 'streaming_platform',
        'api_endpoint', 'file_system', 'cloud_storage'
    ]
    
    # Transformation types
    TRANSFORM_TYPES = [
        'etl_job', 'spark_job', 'sql_query', 'python_script',
        'airflow_dag', 'dbt_model', 'kafka_stream'
    ]
    
    # Schema column types
    COLUMN_TYPES = ['string', 'int', 'float', 'datetime', 'bool', 'json']
    
    def __init__(self, config: Optional[SyntheticDataConfig] = None):
        """
        Initialize synthetic data generator.
        
        Args:
            config: Generation configuration
        """
        self.config = config or SyntheticDataConfig()
        
        np.random.seed(self.config.seed)
        random.seed(self.config.seed)
        
        # Generated data storage
        self.data_sources: List[Dict] = []
        self.datasets: List[Dict] = []
        self.transformations: List[Dict] = []
        self.lineage_edges: List[Dict] = []
        self.processing_events: List[Dict] = []
    
    def generate_all(self) -> Dict:
        """
        Generate complete synthetic dataset.
        
        Returns:
            Dictionary containing all generated data
        """
        logger.info("Generating synthetic data...")
        
        # Generate data sources
        self._generate_data_sources()
        
        # Generate datasets
        self._generate_datasets()
        
        # Generate transformations and lineage
        self._generate_transformations()
        
        # Generate processing events
        self._generate_processing_events()
        
        logger.info(
            f"Generated: {len(self.data_sources)} sources, "
            f"{len(self.datasets)} datasets, "
            f"{len(self.transformations)} transformations, "
            f"{len(self.processing_events)} events"
        )
        
        return {
            'data_sources': self.data_sources,
            'datasets': self.datasets,
            'transformations': self.transformations,
            'lineage_edges': self.lineage_edges,
            'processing_events': self.processing_events
        }
    
    def _generate_data_sources(self) -> None:
        """Generate synthetic data sources."""
        for i in range(self.config.num_data_sources):
            source_type = random.choice(self.SOURCE_TYPES)
            
            source = {
                'source_id': f'src_{i:04d}',
                'name': f'{source_type}_{i}',
                'type': source_type,
                'connection_string': f'connection://{source_type}/{i}',
                'created_at': self._random_timestamp(),
                'owner': f'team_{random.randint(1, 10)}',
                'classification': random.choice(['public', 'internal', 'confidential', 'restricted'])
            }
            
            self.data_sources.append(source)
    
    def _generate_datasets(self) -> None:
        """Generate synthetic datasets."""
        num_datasets = self.config.num_data_sources * 5  # ~5 datasets per source
        
        for i in range(num_datasets):
            source = random.choice(self.data_sources)
            num_columns = random.randint(5, 50)
            
            columns = [
                {
                    'name': f'col_{j}',
                    'type': random.choice(self.COLUMN_TYPES),
                    'nullable': random.random() > 0.3
                }
                for j in range(num_columns)
            ]
            
            # Determine if dataset contains PII
            has_pii = random.random() < 0.2
            pii_columns = []
            if has_pii:
                pii_types = ['email', 'phone', 'ssn', 'name', 'address']
                num_pii = random.randint(1, 3)
                pii_columns = random.sample(pii_types, min(num_pii, len(pii_types)))
            
            dataset = {
                'dataset_id': f'ds_{i:05d}',
                'name': f'dataset_{i}',
                'source_id': source['source_id'],
                'columns': columns,
                'dtypes': [c['type'] for c in columns],
                'num_columns': num_columns,
                'row_count': random.randint(1000, 10000000),
                'size_bytes': random.randint(1000000, 10000000000),
                'created_at': self._random_timestamp(),
                'updated_at': self._random_timestamp(),
                'schema_hash': hashlib.md5(str(columns).encode()).hexdigest()[:8],
                'has_pii': has_pii,
                'pii_columns': pii_columns,
                'retention_days': random.choice([30, 90, 365, 730, None]),
                'classification': source['classification']
            }
            
            self.datasets.append(dataset)
    
    def _generate_transformations(self) -> None:
        """Generate transformations and lineage relationships."""
        for i in range(self.config.num_transformations):
            transform_type = random.choice(self.TRANSFORM_TYPES)
            
            # Select input datasets (1-3)
            num_inputs = random.randint(1, 3)
            input_datasets = random.sample(
                self.datasets, 
                min(num_inputs, len(self.datasets))
            )
            
            # Select output datasets (1-2)
            num_outputs = random.randint(1, 2)
            available_outputs = [
                d for d in self.datasets 
                if d not in input_datasets
            ]
            output_datasets = random.sample(
                available_outputs,
                min(num_outputs, len(available_outputs))
            )
            
            transformation = {
                'transform_id': f'tf_{i:05d}',
                'name': f'{transform_type}_{i}',
                'type': transform_type,
                'input_datasets': [d['dataset_id'] for d in input_datasets],
                'output_datasets': [d['dataset_id'] for d in output_datasets],
                'created_at': self._random_timestamp(),
                'schedule': random.choice(['hourly', 'daily', 'weekly', 'on_demand']),
                'owner': f'team_{random.randint(1, 10)}',
                'avg_runtime_seconds': random.randint(10, 3600)
            }
            
            self.transformations.append(transformation)
            
            # Create lineage edges
            for inp in input_datasets:
                self.lineage_edges.append({
                    'source_id': inp['dataset_id'],
                    'target_id': transformation['transform_id'],
                    'edge_type': 'input'
                })
            
            for out in output_datasets:
                self.lineage_edges.append({
                    'source_id': transformation['transform_id'],
                    'target_id': out['dataset_id'],
                    'edge_type': 'output'
                })
    
    def _generate_processing_events(self) -> None:
        """Generate processing events with injected anomalies."""
        num_anomalies = int(self.config.num_processing_events * self.config.anomaly_rate)
        anomaly_indices = set(random.sample(
            range(self.config.num_processing_events),
            num_anomalies
        ))
        
        for i in range(self.config.num_processing_events):
            is_anomaly = i in anomaly_indices
            
            # Select random transformation
            transform = random.choice(self.transformations)
            
            # Generate base event
            event = self._generate_normal_event(transform)
            event['event_id'] = f'evt_{i:07d}'
            
            # Inject anomaly if applicable
            if is_anomaly:
                event = self._inject_anomaly(event)
            
            self.processing_events.append(event)
    
    def _generate_normal_event(self, transform: Dict) -> Dict:
        """Generate a normal processing event."""
        timestamp = self._random_timestamp()
        dt = datetime.fromtimestamp(timestamp)
        
        # Normal execution time (within expected range)
        avg_runtime = transform['avg_runtime_seconds']
        execution_time = max(1, np.random.normal(avg_runtime, avg_runtime * 0.2))
        
        # Normal data volumes
        rows_processed = random.randint(10000, 1000000)
        bytes_processed = rows_processed * random.randint(100, 1000)
        
        return {
            'transform_id': transform['transform_id'],
            'transform_name': transform['name'],
            'event_type': 'transformation',
            'source_system': transform['type'],
            'timestamp': timestamp,
            'execution_time': execution_time,
            'start_hour': dt.hour,
            'day_of_week': dt.weekday(),
            'rows_processed': rows_processed,
            'bytes_processed': bytes_processed,
            'num_columns': random.randint(5, 50),
            'schema_hash': hashlib.md5(str(random.random()).encode()).hexdigest()[:8],
            'num_reads': random.randint(1, 10),
            'num_writes': random.randint(1, 5),
            'unique_users': random.randint(1, 5),
            'error_count': 0,
            'warning_count': random.randint(0, 3),
            'null_ratio': random.uniform(0, 0.1),
            'is_anomaly': False,
            'anomaly_type': None,
            # Compliance flags
            'schema_changed': False,
            'unauthorized_user': False,
            'off_schedule': False,
            'retention_exceeded': False,
            'pii_detected': False,
            'user_authorized': True,
            'schema_change_approved': True
        }
    
    def _inject_anomaly(self, event: Dict) -> Dict:
        """Inject anomaly into event."""
        anomaly_type = random.choice([
            'schema_drift', 'unauthorized_access', 'unexpected_schedule',
            'data_quality', 'retention_violation', 'pii_exposure'
        ])
        
        event['is_anomaly'] = True
        event['anomaly_type'] = anomaly_type
        
        if anomaly_type == 'schema_drift':
            event['schema_changed'] = True
            event['schema_change_approved'] = False
            event['num_columns'] = event['num_columns'] + random.randint(5, 20)
            
        elif anomaly_type == 'unauthorized_access':
            event['unauthorized_user'] = True
            event['user_authorized'] = False
            event['unique_users'] = event['unique_users'] + random.randint(5, 20)
            
        elif anomaly_type == 'unexpected_schedule':
            event['off_schedule'] = True
            # Unusual hour (e.g., 3 AM on weekend)
            event['start_hour'] = random.choice([2, 3, 4, 5])
            event['day_of_week'] = random.choice([5, 6])  # Weekend
            
        elif anomaly_type == 'data_quality':
            event['null_ratio'] = random.uniform(0.5, 0.9)
            event['error_count'] = random.randint(10, 100)
            event['warning_count'] = random.randint(20, 200)
            
        elif anomaly_type == 'retention_violation':
            event['retention_exceeded'] = True
            
        elif anomaly_type == 'pii_exposure':
            event['pii_detected'] = True
        
        # Anomalous execution patterns
        if random.random() > 0.5:
            event['execution_time'] = event['execution_time'] * random.uniform(3, 10)
        
        if random.random() > 0.5:
            event['rows_processed'] = event['rows_processed'] * random.uniform(0.01, 0.1)
        
        return event
    
    def _random_timestamp(self) -> float:
        """Generate random timestamp within configured range."""
        delta = self.config.end_date - self.config.start_date
        random_days = random.uniform(0, delta.days)
        random_date = self.config.start_date + timedelta(days=random_days)
        return random_date.timestamp()
    
    def get_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract training data for anomaly detection.
        
        Returns:
            Tuple of (normal_data, anomaly_data) feature arrays
        """
        normal_events = [e for e in self.processing_events if not e['is_anomaly']]
        anomaly_events = [e for e in self.processing_events if e['is_anomaly']]
        
        def extract_features(events: List[Dict]) -> np.ndarray:
            features = []
            for e in events:
                feat = [
                    np.log1p(e['execution_time']),
                    np.sin(2 * np.pi * e['start_hour'] / 24),
                    np.cos(2 * np.pi * e['start_hour'] / 24),
                    np.sin(2 * np.pi * e['day_of_week'] / 7),
                    np.cos(2 * np.pi * e['day_of_week'] / 7),
                    np.log1p(e['rows_processed']),
                    np.log1p(e['bytes_processed']),
                    e['num_columns'] / 100,
                    np.log1p(e['num_reads']),
                    np.log1p(e['num_writes']),
                    np.log1p(e['unique_users']),
                    np.log1p(e['error_count']),
                    np.log1p(e['warning_count']),
                    e['null_ratio']
                ]
                # Pad to input_dim
                feat.extend([0] * (256 - len(feat)))
                features.append(feat[:256])
            
            return np.array(features, dtype=np.float32)
        
        normal_data = extract_features(normal_events)
        anomaly_data = extract_features(anomaly_events)
        
        return normal_data, anomaly_data
    
    def get_lineage_data(self) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Get data for lineage tracking evaluation.
        
        Returns:
            Tuple of (datasets, transformations, ground_truth_edges)
        """
        return self.datasets, self.transformations, self.lineage_edges
