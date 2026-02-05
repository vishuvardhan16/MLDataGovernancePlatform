"""
Visualization Module for Data Governance Framework.

Provides visual representations of lineage graphs, anomaly distributions,
and policy compliance dashboards.

Reference: Section 5 - Experimental Evaluation (Figures 2-4)
"""

import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Optional imports for visualization
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import LinearSegmentedColormap
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logger.warning("matplotlib not installed. Visualization features disabled.")

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False


class GovernanceVisualizer:
    """
    Visualization tools for governance framework outputs.
    
    Generates figures matching the paper's experimental results:
    - Figure 2: Lineage graph visualization
    - Figure 3: Anomaly detection performance curves
    - Figure 4: Policy enforcement outcomes
    """
    
    def __init__(self, output_dir: str = "outputs/figures"):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory for saving figures
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Color scheme
        self.colors = {
            'dataset': '#4CAF50',      # Green
            'transformation': '#2196F3', # Blue
            'anomaly': '#F44336',       # Red
            'normal': '#9E9E9E',        # Gray
            'compliant': '#4CAF50',     # Green
            'violation': '#F44336',     # Red
        }
    
    def plot_lineage_graph(
        self,
        nodes: Dict,
        edges: Dict,
        title: str = "Data Lineage Graph",
        save_path: Optional[str] = None
    ) -> Optional[str]:
        """
        Visualize lineage graph structure (Figure 2).
        
        Args:
            nodes: Dictionary of LineageNode objects
            edges: Dictionary of LineageEdge objects
            title: Plot title
            save_path: Optional path to save figure
            
        Returns:
            Path to saved figure or None
        """
        if not HAS_MATPLOTLIB or not HAS_NETWORKX:
            logger.warning("Visualization requires matplotlib and networkx")
            return None
        
        # Build NetworkX graph
        G = nx.DiGraph()
        
        node_colors = []
        node_sizes = []
        
        for node_id, node in nodes.items():
            G.add_node(node_id, label=node.name[:15])
            
            if node.node_type == 'dataset':
                node_colors.append(self.colors['dataset'])
                node_sizes.append(800)
            else:
                node_colors.append(self.colors['transformation'])
                node_sizes.append(500)
        
        edge_colors = []
        edge_styles = []
        
        for (src, tgt), edge in edges.items():
            G.add_edge(src, tgt, weight=edge.weight)
            
            if edge.edge_type == 'direct':
                edge_colors.append('#333333')
                edge_styles.append('solid')
            else:
                edge_colors.append('#999999')
                edge_styles.append('dashed')
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
        
        # Layout
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        
        # Draw edges
        nx.draw_networkx_edges(
            G, pos, ax=ax,
            edge_color=edge_colors,
            alpha=0.6,
            arrows=True,
            arrowsize=15,
            connectionstyle="arc3,rad=0.1"
        )
        
        # Draw nodes
        nx.draw_networkx_nodes(
            G, pos, ax=ax,
            node_color=node_colors,
            node_size=node_sizes,
            alpha=0.9
        )
        
        # Draw labels
        labels = nx.get_node_attributes(G, 'label')
        nx.draw_networkx_labels(
            G, pos, labels, ax=ax,
            font_size=8,
            font_weight='bold'
        )
        
        # Legend
        legend_elements = [
            mpatches.Patch(color=self.colors['dataset'], label='Dataset'),
            mpatches.Patch(color=self.colors['transformation'], label='Transformation'),
        ]
        ax.legend(handles=legend_elements, loc='upper left')
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        
        # Save
        save_path = save_path or str(self.output_dir / "lineage_graph.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Lineage graph saved to {save_path}")
        return save_path
    
    def plot_anomaly_distribution(
        self,
        anomaly_scores: List[float],
        labels: List[int],
        threshold: float,
        title: str = "Anomaly Score Distribution",
        save_path: Optional[str] = None
    ) -> Optional[str]:
        """
        Plot anomaly score distribution (Figure 3a).
        
        Args:
            anomaly_scores: List of anomaly scores
            labels: Ground truth labels (0=normal, 1=anomaly)
            threshold: Detection threshold
            title: Plot title
            save_path: Optional path to save figure
            
        Returns:
            Path to saved figure or None
        """
        if not HAS_MATPLOTLIB:
            return None
        
        scores = np.array(anomaly_scores)
        labels = np.array(labels)
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Separate normal and anomaly scores
        normal_scores = scores[labels == 0]
        anomaly_scores_arr = scores[labels == 1]
        
        # Plot histograms
        bins = np.linspace(0, 1, 50)
        
        ax.hist(
            normal_scores, bins=bins, alpha=0.7,
            color=self.colors['normal'], label='Normal',
            density=True
        )
        ax.hist(
            anomaly_scores_arr, bins=bins, alpha=0.7,
            color=self.colors['anomaly'], label='Anomaly',
            density=True
        )
        
        # Threshold line
        ax.axvline(
            x=threshold, color='black', linestyle='--',
            linewidth=2, label=f'Threshold (τ={threshold:.2f})'
        )
        
        ax.set_xlabel('Anomaly Score', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        save_path = save_path or str(self.output_dir / "anomaly_distribution.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Anomaly distribution saved to {save_path}")
        return save_path
    
    def plot_roc_curve(
        self,
        fpr: np.ndarray,
        tpr: np.ndarray,
        auc_score: float,
        title: str = "ROC Curve - Anomaly Detection",
        save_path: Optional[str] = None
    ) -> Optional[str]:
        """
        Plot ROC curve for anomaly detection (Figure 3b).
        
        Args:
            fpr: False positive rates
            tpr: True positive rates
            auc_score: Area under curve
            title: Plot title
            save_path: Optional path to save figure
            
        Returns:
            Path to saved figure or None
        """
        if not HAS_MATPLOTLIB:
            return None
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        
        ax.plot(
            fpr, tpr,
            color='#2196F3', linewidth=2,
            label=f'ROC Curve (AUC = {auc_score:.3f})'
        )
        ax.plot(
            [0, 1], [0, 1],
            color='gray', linestyle='--', linewidth=1,
            label='Random Classifier'
        )
        
        ax.fill_between(fpr, tpr, alpha=0.2, color='#2196F3')
        
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        
        save_path = save_path or str(self.output_dir / "roc_curve.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"ROC curve saved to {save_path}")
        return save_path
    
    def plot_policy_outcomes(
        self,
        violation_counts: Dict[str, int],
        title: str = "Policy Violation Distribution",
        save_path: Optional[str] = None
    ) -> Optional[str]:
        """
        Plot policy enforcement outcomes (Figure 4).
        
        Args:
            violation_counts: Dictionary of policy_id -> violation count
            title: Plot title
            save_path: Optional path to save figure
            
        Returns:
            Path to saved figure or None
        """
        if not HAS_MATPLOTLIB:
            return None
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        policies = list(violation_counts.keys())
        counts = list(violation_counts.values())
        
        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(policies)))
        
        bars = ax.barh(policies, counts, color=colors, edgecolor='black', linewidth=0.5)
        
        # Add value labels
        for bar, count in zip(bars, counts):
            ax.text(
                bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                str(count), va='center', fontsize=10
            )
        
        ax.set_xlabel('Number of Violations', fontsize=12)
        ax.set_ylabel('Policy', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        save_path = save_path or str(self.output_dir / "policy_outcomes.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Policy outcomes saved to {save_path}")
        return save_path
    
    def plot_training_curves(
        self,
        train_loss: List[float],
        val_loss: Optional[List[float]] = None,
        title: str = "Autoencoder Training Loss",
        save_path: Optional[str] = None
    ) -> Optional[str]:
        """
        Plot training curves for anomaly detection model.
        
        Args:
            train_loss: Training loss per epoch
            val_loss: Optional validation loss per epoch
            title: Plot title
            save_path: Optional path to save figure
            
        Returns:
            Path to saved figure or None
        """
        if not HAS_MATPLOTLIB:
            return None
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        epochs = range(1, len(train_loss) + 1)
        
        ax.plot(
            epochs, train_loss,
            color='#2196F3', linewidth=2, label='Training Loss'
        )
        
        if val_loss:
            ax.plot(
                epochs, val_loss,
                color='#FF9800', linewidth=2, label='Validation Loss'
            )
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        save_path = save_path or str(self.output_dir / "training_curves.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Training curves saved to {save_path}")
        return save_path
    
    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        labels: List[str] = None,
        title: str = "Anomaly Detection Confusion Matrix",
        save_path: Optional[str] = None
    ) -> Optional[str]:
        """
        Plot confusion matrix for anomaly detection.
        
        Args:
            cm: Confusion matrix array
            labels: Class labels
            title: Plot title
            save_path: Optional path to save figure
            
        Returns:
            Path to saved figure or None
        """
        if not HAS_MATPLOTLIB:
            return None
        
        labels = labels or ['Normal', 'Anomaly']
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        ax.figure.colorbar(im, ax=ax)
        
        ax.set(
            xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            xticklabels=labels,
            yticklabels=labels,
            ylabel='True Label',
            xlabel='Predicted Label'
        )
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=14
                )
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        save_path = save_path or str(self.output_dir / "confusion_matrix.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Confusion matrix saved to {save_path}")
        return save_path
