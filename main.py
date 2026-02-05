#!/usr/bin/env python3
"""
Main Experiment Script for Data Governance Automation Framework.

Implements the experimental evaluation described in Section 5 of the paper:
"Automating Data Governance: A Machine Learning Approach for Lineage Tracking,
Anomaly Detection, and Policy Enforcement in Enterprise Data Platforms"

Usage:
    python main.py [--config CONFIG_PATH] [--output OUTPUT_DIR]
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data_governance.config import FrameworkConfig
from data_governance.framework import GovernanceAutomationFramework
from data_governance.synthetic_data import SyntheticDataGenerator, SyntheticDataConfig
from data_governance.evaluation import FrameworkEvaluator


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('experiment.log')
    ]
)
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_experiment(
    output_dir: str = 'outputs',
    num_events: int = 10000,
    anomaly_rate: float = 0.05
) -> None:
    """
    Run the complete experimental evaluation.
    
    Args:
        output_dir: Directory for output files
        num_events: Number of processing events to generate
        anomaly_rate: Fraction of anomalous events
    """
    logger.info("="*70)
    logger.info("DATA GOVERNANCE AUTOMATION FRAMEWORK - EXPERIMENTAL EVALUATION")
    logger.info("="*70)
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # Step 1: Generate Synthetic Data
    # =========================================================================
    logger.info("\n[Step 1] Generating synthetic enterprise data...")
    
    data_config = SyntheticDataConfig(
        num_data_sources=20,
        num_transformations=100,
        num_lineage_chains=80,
        num_processing_events=num_events,
        anomaly_rate=anomaly_rate,
        seed=42
    )
    
    generator = SyntheticDataGenerator(data_config)
    synthetic_data = generator.generate_all()
    
    logger.info(f"  - Data sources: {len(synthetic_data['data_sources'])}")
    logger.info(f"  - Datasets: {len(synthetic_data['datasets'])}")
    logger.info(f"  - Transformations: {len(synthetic_data['transformations'])}")
    logger.info(f"  - Lineage edges: {len(synthetic_data['lineage_edges'])}")
    logger.info(f"  - Processing events: {len(synthetic_data['processing_events'])}")
    
    # Get training data
    normal_data, anomaly_data = generator.get_training_data()
    logger.info(f"  - Normal samples: {len(normal_data)}")
    logger.info(f"  - Anomaly samples: {len(anomaly_data)}")
    
    # =========================================================================
    # Step 2: Initialize Framework
    # =========================================================================
    logger.info("\n[Step 2] Initializing governance automation framework...")
    
    config = FrameworkConfig()
    framework = GovernanceAutomationFramework(config)
    
    # Split data for training/validation
    split_idx = int(len(normal_data) * 0.8)
    train_data = normal_data[:split_idx]
    val_data = normal_data[split_idx:]
    
    # Initialize framework (train anomaly detection model)
    init_results = framework.initialize(
        training_data=train_data,
        validation_data=val_data
    )
    
    logger.info(f"  - Anomaly threshold: {init_results['anomaly_training']['threshold']:.6f}")
    
    # =========================================================================
    # Step 3: Build Lineage Graph
    # =========================================================================
    logger.info("\n[Step 3] Building lineage graph...")
    
    # Add datasets to lineage graph
    for dataset in synthetic_data['datasets']:
        framework.lineage_engine.add_dataset(
            dataset_id=dataset['dataset_id'],
            name=dataset['name'],
            metadata=dataset
        )
    
    # Add transformations
    for transform in synthetic_data['transformations']:
        framework.lineage_engine.add_transformation(
            transform_id=transform['transform_id'],
            name=transform['name'],
            input_datasets=transform['input_datasets'],
            output_datasets=transform['output_datasets'],
            metadata=transform
        )
    
    logger.info(f"  - Nodes in graph: {len(framework.lineage_engine.graph.nodes)}")
    logger.info(f"  - Edges in graph: {len(framework.lineage_engine.graph.edges)}")
    
    # Infer implicit lineage
    num_inferred = framework.infer_lineage()
    logger.info(f"  - Inferred edges: {num_inferred}")
    
    # =========================================================================
    # Step 4: Process Events
    # =========================================================================
    logger.info("\n[Step 4] Processing governance events...")
    
    start_time = time.time()
    
    for i, event in enumerate(synthetic_data['processing_events']):
        framework.process_event(
            event_type=event['event_type'],
            source_system=event['source_system'],
            metadata=event,
            event_id=event['event_id']
        )
        
        if (i + 1) % 1000 == 0:
            logger.info(f"  - Processed {i + 1}/{len(synthetic_data['processing_events'])} events")
    
    processing_time = time.time() - start_time
    logger.info(f"  - Total processing time: {processing_time:.2f}s")
    logger.info(f"  - Throughput: {len(synthetic_data['processing_events']) / processing_time:.1f} events/sec")
    
    # =========================================================================
    # Step 5: Evaluate Framework
    # =========================================================================
    logger.info("\n[Step 5] Evaluating framework performance...")
    
    evaluator = FrameworkEvaluator(framework, generator)
    results = evaluator.evaluate_all()
    
    # Generate and print report
    report = evaluator.generate_report()
    print(report)
    
    # Save report
    report_path = Path(output_dir) / 'evaluation_report.txt'
    with open(report_path, 'w') as f:
        f.write(report)
    logger.info(f"  - Report saved to: {report_path}")
    
    # Compare with baselines
    comparison = evaluator.compare_with_baselines()
    logger.info("\n[Comparison with Baselines]")
    logger.info(f"  - Lineage F1 improvement: {comparison['improvements']['lineage_f1']:.1f}%")
    logger.info(f"  - Anomaly F1 improvement: {comparison['improvements']['anomaly_f1']:.1f}%")
    
    # =========================================================================
    # Step 6: Generate Governance Report
    # =========================================================================
    logger.info("\n[Step 6] Generating governance report...")
    
    gov_report = framework.get_governance_report()
    
    logger.info(f"  - Total events processed: {gov_report['events']['total_events']}")
    logger.info(f"  - Events with anomalies: {gov_report['events']['events_with_anomalies']}")
    logger.info(f"  - Events with violations: {gov_report['events']['events_with_violations']}")
    logger.info(f"  - Total policy violations: {gov_report['policy']['total_violations']}")
    
    # Export governance report
    report_json_path = framework.export_report(str(Path(output_dir) / 'governance_report.json'))
    logger.info(f"  - Governance report saved to: {report_json_path}")
    
    # Save checkpoint
    checkpoint_path = framework.save_checkpoint(str(Path(output_dir) / 'model_checkpoint.pt'))
    logger.info(f"  - Model checkpoint saved to: {checkpoint_path}")
    
    # =========================================================================
    # Summary
    # =========================================================================
    logger.info("\n" + "="*70)
    logger.info("EXPERIMENT COMPLETE")
    logger.info("="*70)
    logger.info(f"\nKey Results:")
    logger.info(f"  - Lineage Tracking F1: {results.lineage_f1:.4f}")
    logger.info(f"  - Anomaly Detection F1: {results.anomaly_f1:.4f}")
    logger.info(f"  - Policy Enforcement Accuracy: {results.policy_accuracy:.4f}")
    logger.info(f"  - System Throughput: {results.throughput_events_per_sec:.1f} events/sec")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Data Governance Automation Framework Experiment'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='outputs',
        help='Output directory for results'
    )
    parser.add_argument(
        '--num-events', '-n',
        type=int,
        default=10000,
        help='Number of processing events to generate'
    )
    parser.add_argument(
        '--anomaly-rate', '-a',
        type=float,
        default=0.05,
        help='Fraction of anomalous events (0-1)'
    )
    
    args = parser.parse_args()
    
    try:
        run_experiment(
            output_dir=args.output,
            num_events=args.num_events,
            anomaly_rate=args.anomaly_rate
        )
    except Exception as e:
        logger.error(f"Experiment failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
