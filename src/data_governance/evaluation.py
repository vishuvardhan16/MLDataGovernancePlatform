"""
Evaluation Module for Data Governance Framework.

Implements comprehensive evaluation metrics for lineage tracking,
anomaly detection, and policy enforcement components.

Reference: Section 5 - Experimental Evaluation and Results
"""

import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import time

import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    accuracy_score, roc_auc_score, confusion_matrix,
    precision_recall_curve, average_precision_score
)

from .framework import GovernanceAutomationFramework
from .synthetic_data import SyntheticDataGenerator


logger = logging.getLogger(__name__)


@dataclass
class EvaluationResults:
    """Container for evaluation results."""
    
    # Lineage metrics
    lineage_precision: float = 0.0
    lineage_recall: float = 0.0
    lineage_f1: float = 0.0
    
    # Anomaly detection metrics
    anomaly_precision: float = 0.0
    anomaly_recall: float = 0.0
    anomaly_f1: float = 0.0
    anomaly_auc: float = 0.0
    
    # Policy enforcement metrics
    policy_accuracy: float = 0.0
    false_positive_rate: float = 0.0
    response_time_ms: float = 0.0
    
    # System performance
    throughput_events_per_sec: float = 0.0
    latency_ms: float = 0.0


class FrameworkEvaluator:
    """
    Evaluator for the governance automation framework.
    
    Computes metrics for:
    - Lineage tracking performance (Table 1)
    - Anomaly detection performance (Table 2)
    - Policy enforcement outcomes (Figure 3)
    - System performance benchmarks (Table 3)
    """
    
    def __init__(
        self,
        framework: GovernanceAutomationFramework,
        data_generator: SyntheticDataGenerator
    ):
        """
        Initialize evaluator.
        
        Args:
            framework: Governance automation framework
            data_generator: Synthetic data generator
        """
        self.framework = framework
        self.data_generator = data_generator
        self.results = EvaluationResults()
    
    def evaluate_all(self) -> EvaluationResults:
        """
        Run complete evaluation suite.
        
        Returns:
            Evaluation results
        """
        logger.info("Starting comprehensive evaluation...")
        
        # Evaluate lineage tracking
        self._evaluate_lineage()
        
        # Evaluate anomaly detection
        self._evaluate_anomaly_detection()
        
        # Evaluate policy enforcement
        self._evaluate_policy_enforcement()
        
        # Evaluate system performance
        self._evaluate_performance()
        
        return self.results
    
    def _evaluate_lineage(self) -> None:
        """
        Evaluate lineage tracking performance.
        
        Computes precision, recall, and F1 score against ground truth.
        """
        logger.info("Evaluating lineage tracking...")
        
        datasets, transformations, ground_truth = self.data_generator.get_lineage_data()
        
        # Build ground truth edge set
        gt_edges = set()
        for edge in ground_truth:
            gt_edges.add((edge['source_id'], edge['target_id']))
        
        # Get predicted edges from framework
        predicted_edges = set()
        for (src, tgt), edge in self.framework.lineage_engine.graph.edges.items():
            predicted_edges.add((src, tgt))
        
        # Calculate metrics
        if len(gt_edges) == 0 or len(predicted_edges) == 0:
            logger.warning("Insufficient edges for lineage evaluation")
            return
        
        # True positives: edges in both predicted and ground truth
        true_positives = len(predicted_edges & gt_edges)
        
        # False positives: edges in predicted but not in ground truth
        false_positives = len(predicted_edges - gt_edges)
        
        # False negatives: edges in ground truth but not in predicted
        false_negatives = len(gt_edges - predicted_edges)
        
        # Precision, Recall, F1
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        self.results.lineage_precision = precision
        self.results.lineage_recall = recall
        self.results.lineage_f1 = f1
        
        logger.info(
            f"Lineage Tracking - Precision: {precision:.4f}, "
            f"Recall: {recall:.4f}, F1: {f1:.4f}"
        )
    
    def _evaluate_anomaly_detection(self) -> None:
        """
        Evaluate anomaly detection performance.
        
        Computes precision, recall, F1, and AUC-ROC.
        """
        logger.info("Evaluating anomaly detection...")
        
        events = self.data_generator.processing_events
        
        # Get ground truth labels
        y_true = np.array([1 if e['is_anomaly'] else 0 for e in events])
        
        # Get predictions from framework
        y_pred = []
        y_scores = []
        
        for event in events:
            # Check if anomaly was detected
            detected = any(
                pe.event_id == event.get('event_id') and pe.anomaly_detected
                for pe in self.framework.processed_events
            )
            y_pred.append(1 if detected else 0)
            
            # Get anomaly score if available
            anomaly_events = [
                ae for ae in self.framework.anomaly_engine.anomaly_history
                if ae.event_id == event.get('event_id')
            ]
            if anomaly_events:
                y_scores.append(anomaly_events[0].anomaly_score)
            else:
                y_scores.append(0.0)
        
        y_pred = np.array(y_pred)
        y_scores = np.array(y_scores)
        
        # Calculate metrics
        if len(np.unique(y_true)) < 2:
            logger.warning("Insufficient class diversity for anomaly evaluation")
            return
        
        self.results.anomaly_precision = precision_score(y_true, y_pred, zero_division=0)
        self.results.anomaly_recall = recall_score(y_true, y_pred, zero_division=0)
        self.results.anomaly_f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # AUC-ROC (if we have scores)
        if np.any(y_scores > 0):
            self.results.anomaly_auc = roc_auc_score(y_true, y_scores)
        
        logger.info(
            f"Anomaly Detection - Precision: {self.results.anomaly_precision:.4f}, "
            f"Recall: {self.results.anomaly_recall:.4f}, "
            f"F1: {self.results.anomaly_f1:.4f}, "
            f"AUC: {self.results.anomaly_auc:.4f}"
        )
    
    def _evaluate_policy_enforcement(self) -> None:
        """
        Evaluate policy enforcement performance.
        
        Computes accuracy and false positive rate.
        """
        logger.info("Evaluating policy enforcement...")
        
        events = self.data_generator.processing_events
        violations = self.framework.policy_engine.violations
        
        # Ground truth: events that should trigger violations
        should_violate = set()
        for event in events:
            if event['is_anomaly']:
                # Check specific violation conditions
                if event.get('pii_detected'):
                    should_violate.add(event['event_id'])
                if event.get('unauthorized_user'):
                    should_violate.add(event['event_id'])
                if event.get('retention_exceeded'):
                    should_violate.add(event['event_id'])
                if event.get('schema_changed') and not event.get('schema_change_approved'):
                    should_violate.add(event['event_id'])
        
        # Predicted violations
        predicted_violations = set()
        for pe in self.framework.processed_events:
            if pe.violations:
                predicted_violations.add(pe.event_id)
        
        # Calculate metrics
        if len(should_violate) == 0:
            logger.warning("No ground truth violations for policy evaluation")
            return
        
        # True positives
        tp = len(predicted_violations & should_violate)
        
        # False positives
        fp = len(predicted_violations - should_violate)
        
        # False negatives
        fn = len(should_violate - predicted_violations)
        
        # True negatives
        all_events = set(e['event_id'] for e in events)
        tn = len(all_events - predicted_violations - should_violate)
        
        # Accuracy
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        
        # False positive rate
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        self.results.policy_accuracy = accuracy
        self.results.false_positive_rate = fpr
        
        logger.info(
            f"Policy Enforcement - Accuracy: {accuracy:.4f}, "
            f"False Positive Rate: {fpr:.4f}"
        )
    
    def _evaluate_performance(self) -> None:
        """
        Evaluate system performance benchmarks.
        
        Measures throughput and latency.
        """
        logger.info("Evaluating system performance...")
        
        # Measure event processing throughput
        test_events = self.data_generator.processing_events[:1000]
        
        start_time = time.time()
        
        for event in test_events:
            self.framework.process_event(
                event_type=event['event_type'],
                source_system=event['source_system'],
                metadata=event,
                event_id=f"perf_test_{event['event_id']}"
            )
        
        elapsed = time.time() - start_time
        
        self.results.throughput_events_per_sec = len(test_events) / elapsed
        self.results.latency_ms = (elapsed / len(test_events)) * 1000
        
        logger.info(
            f"Performance - Throughput: {self.results.throughput_events_per_sec:.1f} events/sec, "
            f"Latency: {self.results.latency_ms:.2f} ms/event"
        )
    
    def generate_report(self) -> str:
        """
        Generate comprehensive evaluation report.
        
        Returns:
            Formatted report string
        """
        report = f"""
{'='*70}
DATA GOVERNANCE AUTOMATION FRAMEWORK - EVALUATION REPORT
{'='*70}

1. LINEAGE TRACKING PERFORMANCE (Table 1)
{'-'*70}
Metric                  | Proposed (ML-Driven)
{'-'*70}
Precision               | {self.results.lineage_precision:.4f}
Recall                  | {self.results.lineage_recall:.4f}
F1 Score                | {self.results.lineage_f1:.4f}

2. ANOMALY DETECTION PERFORMANCE (Table 2)
{'-'*70}
Model                   | Deep Autoencoder
{'-'*70}
Precision               | {self.results.anomaly_precision:.4f}
Recall                  | {self.results.anomaly_recall:.4f}
F1 Score                | {self.results.anomaly_f1:.4f}
AUC-ROC                 | {self.results.anomaly_auc:.4f}

3. POLICY ENFORCEMENT OUTCOMES
{'-'*70}
Accuracy                | {self.results.policy_accuracy:.4f}
False Positive Rate     | {self.results.false_positive_rate:.4f}

4. SYSTEM PERFORMANCE BENCHMARKS (Table 3)
{'-'*70}
Throughput              | {self.results.throughput_events_per_sec:.1f} events/sec
Latency                 | {self.results.latency_ms:.2f} ms/operation

{'='*70}
"""
        return report
    
    def compare_with_baselines(self) -> Dict:
        """
        Compare results with baseline methods from paper.
        
        Returns:
            Comparison dictionary
        """
        # Baseline values from Table 1 and Table 2 in the paper
        baselines = {
            'rule_based': {
                'lineage_precision': 0.78,
                'lineage_recall': 0.72,
                'lineage_f1': 0.75
            },
            'apache_atlas': {
                'lineage_precision': 0.81,
                'lineage_recall': 0.75,
                'lineage_f1': 0.78
            },
            'databand': {
                'lineage_precision': 0.79,
                'lineage_recall': 0.76,
                'lineage_f1': 0.77
            },
            'isolation_forest': {
                'anomaly_precision': 0.83,
                'anomaly_recall': 0.79,
                'anomaly_f1': 0.81
            },
            'dbscan': {
                'anomaly_precision': 0.78,
                'anomaly_recall': 0.74,
                'anomaly_f1': 0.76
            }
        }
        
        comparison = {
            'proposed': {
                'lineage_precision': self.results.lineage_precision,
                'lineage_recall': self.results.lineage_recall,
                'lineage_f1': self.results.lineage_f1,
                'anomaly_precision': self.results.anomaly_precision,
                'anomaly_recall': self.results.anomaly_recall,
                'anomaly_f1': self.results.anomaly_f1
            },
            'baselines': baselines,
            'improvements': {}
        }
        
        # Calculate improvements over best baseline
        best_lineage_f1 = max(b.get('lineage_f1', 0) for b in baselines.values())
        best_anomaly_f1 = max(b.get('anomaly_f1', 0) for b in baselines.values())
        
        comparison['improvements']['lineage_f1'] = (
            (self.results.lineage_f1 - best_lineage_f1) / best_lineage_f1 * 100
            if best_lineage_f1 > 0 else 0
        )
        comparison['improvements']['anomaly_f1'] = (
            (self.results.anomaly_f1 - best_anomaly_f1) / best_anomaly_f1 * 100
            if best_anomaly_f1 > 0 else 0
        )
        
        return comparison
