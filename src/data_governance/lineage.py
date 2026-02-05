"""
Lineage Inference Engine for Data Governance.

Implements graph-based data lineage tracking using Graph Neural Networks (GNN)
to infer implicit relationships from metadata.

Reference: Section 3.2 of the paper - Lineage Inference Engine
"""

import logging
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.data import Data
from sklearn.metrics.pairwise import cosine_similarity

from .config import LineageConfig


logger = logging.getLogger(__name__)


@dataclass
class LineageNode:
    """Represents a node in the lineage graph (dataset, transformation, or metadata)."""
    
    node_id: str
    node_type: str  # 'dataset', 'transformation', 'metadata'
    name: str
    attributes: Dict
    feature_vector: Optional[np.ndarray] = None
    
    def __hash__(self):
        return hash(self.node_id)
    
    def __eq__(self, other):
        return self.node_id == other.node_id


@dataclass
class LineageEdge:
    """Represents an edge (lineage relationship) in the graph."""
    
    source_id: str
    target_id: str
    edge_type: str  # 'direct', 'inferred'
    weight: float
    confidence: float
    metadata: Dict


class LineageGraph:
    """
    Directed multigraph representation of data lineage.
    
    G = (V, E) where V = {D (datasets), T (transformations), M (metadata)}
    and E ⊆ V × V represents lineage relationships.
    """
    
    def __init__(self):
        self.nodes: Dict[str, LineageNode] = {}
        self.edges: Dict[Tuple[str, str], LineageEdge] = {}
        self.adjacency: Dict[str, Set[str]] = defaultdict(set)
        self.reverse_adjacency: Dict[str, Set[str]] = defaultdict(set)
    
    def add_node(self, node: LineageNode) -> None:
        """Add a node to the lineage graph."""
        self.nodes[node.node_id] = node
    
    def add_edge(self, edge: LineageEdge) -> None:
        """Add an edge to the lineage graph."""
        key = (edge.source_id, edge.target_id)
        self.edges[key] = edge
        self.adjacency[edge.source_id].add(edge.target_id)
        self.reverse_adjacency[edge.target_id].add(edge.source_id)
    
    def get_neighbors(self, node_id: str) -> Set[str]:
        """Get downstream neighbors of a node."""
        return self.adjacency.get(node_id, set())
    
    def get_predecessors(self, node_id: str) -> Set[str]:
        """Get upstream predecessors of a node."""
        return self.reverse_adjacency.get(node_id, set())
    
    def get_lineage_path(
        self, 
        source_id: str, 
        target_id: str,
        max_depth: int = 10
    ) -> Optional[List[str]]:
        """
        Find lineage path from source to target using BFS.
        
        Args:
            source_id: Starting node
            target_id: Target node
            max_depth: Maximum path length
            
        Returns:
            List of node IDs representing the path, or None if not found
        """
        if source_id not in self.nodes or target_id not in self.nodes:
            return None
        
        visited = {source_id}
        queue = [(source_id, [source_id])]
        
        while queue:
            current, path = queue.pop(0)
            
            if len(path) > max_depth:
                continue
            
            if current == target_id:
                return path
            
            for neighbor in self.adjacency[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        return None
    
    def compute_lineage_completeness(self, node_id: str) -> float:
        """
        Compute lineage completeness score for a node.
        
        Returns ratio of documented vs expected lineage relationships.
        """
        if node_id not in self.nodes:
            return 0.0
        
        upstream = len(self.get_predecessors(node_id))
        downstream = len(self.get_neighbors(node_id))
        
        # Heuristic: datasets should have at least one upstream source
        node = self.nodes[node_id]
        if node.node_type == 'dataset':
            expected_upstream = 1
            expected_downstream = 0
        elif node.node_type == 'transformation':
            expected_upstream = 1
            expected_downstream = 1
        else:
            expected_upstream = 0
            expected_downstream = 0
        
        if expected_upstream + expected_downstream == 0:
            return 1.0
        
        actual = upstream + downstream
        expected = expected_upstream + expected_downstream
        
        return min(1.0, actual / expected)
    
    def to_pyg_data(self) -> Data:
        """Convert to PyTorch Geometric Data object for GNN processing."""
        node_ids = list(self.nodes.keys())
        node_id_to_idx = {nid: idx for idx, nid in enumerate(node_ids)}
        
        # Build feature matrix
        features = []
        for nid in node_ids:
            node = self.nodes[nid]
            if node.feature_vector is not None:
                features.append(node.feature_vector)
            else:
                features.append(np.zeros(128))  # Default feature dim
        
        x = torch.tensor(np.array(features), dtype=torch.float32)
        
        # Build edge index
        edge_sources = []
        edge_targets = []
        edge_weights = []
        
        for (src, tgt), edge in self.edges.items():
            if src in node_id_to_idx and tgt in node_id_to_idx:
                edge_sources.append(node_id_to_idx[src])
                edge_targets.append(node_id_to_idx[tgt])
                edge_weights.append(edge.weight)
        
        edge_index = torch.tensor(
            [edge_sources, edge_targets], 
            dtype=torch.long
        )
        edge_attr = torch.tensor(edge_weights, dtype=torch.float32)
        
        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            node_ids=node_ids
        )



class GraphAttentionLineage(nn.Module):
    """
    Graph Attention Network for lineage inference.
    
    Implements the GNN architecture from Section 3.2:
    h_v^(k) = σ(W^(k) · AGG({h_u^(k-1) : u ∈ N(v)}))
    
    Uses attention mechanism to weight neighbor contributions.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.1
    ):
        """
        Initialize Graph Attention Network.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output embedding dimension
            num_layers: Number of GAT layers
            heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # GAT layers
        self.gat_layers = nn.ModuleList()
        
        # First layer
        self.gat_layers.append(
            GATConv(hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout)
        )
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.gat_layers.append(
                GATConv(hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout)
            )
        
        # Output layer
        self.gat_layers.append(
            GATConv(hidden_dim, output_dim, heads=1, concat=False, dropout=dropout)
        )
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers - 1)
        ])
        self.output_norm = nn.LayerNorm(output_dim)
    
    def forward(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through GAT layers.
        
        Args:
            x: Node features (num_nodes, input_dim)
            edge_index: Edge connectivity (2, num_edges)
            
        Returns:
            Node embeddings (num_nodes, output_dim)
        """
        # Input projection
        x = self.input_proj(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # GAT layers with residual connections
        for i, gat_layer in enumerate(self.gat_layers[:-1]):
            residual = x
            x = gat_layer(x, edge_index)
            x = F.elu(x)
            x = self.layer_norms[i](x + residual)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Output layer
        x = self.gat_layers[-1](x, edge_index)
        x = self.output_norm(x)
        
        return x


class LineageInferenceEngine:
    """
    Engine for inferring data lineage relationships.
    
    Combines metadata extraction, semantic similarity, and GNN-based
    inference to discover both explicit and implicit lineage paths.
    """
    
    def __init__(self, config: LineageConfig, device: str = 'cpu'):
        """
        Initialize lineage inference engine.
        
        Args:
            config: Lineage configuration
            device: Computation device
        """
        self.config = config
        self.device = device
        
        self.graph = LineageGraph()
        
        # Initialize GNN model
        self.model = GraphAttentionLineage(
            input_dim=config.node_embedding_dim,
            hidden_dim=config.hidden_dim,
            output_dim=config.output_dim,
            num_layers=config.num_gnn_layers,
            heads=config.attention_heads,
            dropout=config.dropout
        ).to(device)
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=0.001,
            weight_decay=1e-5
        )
        
        self._trained = False
    
    def extract_metadata_features(
        self, 
        metadata: Dict
    ) -> np.ndarray:
        """
        Extract feature vector from metadata attributes.
        
        Args:
            metadata: Dictionary of metadata attributes
            
        Returns:
            Feature vector of shape (embedding_dim,)
        """
        features = np.zeros(self.config.node_embedding_dim)
        
        # Encode metadata attributes
        idx = 0
        
        # Schema features
        if 'columns' in metadata:
            num_cols = len(metadata['columns'])
            features[idx] = min(num_cols / 100, 1.0)
            idx += 1
        
        # Data type distribution
        if 'dtypes' in metadata:
            dtype_counts = defaultdict(int)
            for dtype in metadata['dtypes']:
                dtype_counts[dtype] += 1
            
            for i, dtype in enumerate(['string', 'int', 'float', 'datetime', 'bool']):
                if i + idx < len(features):
                    features[idx + i] = dtype_counts.get(dtype, 0) / max(len(metadata['dtypes']), 1)
            idx += 5
        
        # Size features
        if 'row_count' in metadata:
            features[idx] = np.log1p(metadata['row_count']) / 20
            idx += 1
        
        if 'size_bytes' in metadata:
            features[idx] = np.log1p(metadata['size_bytes']) / 30
            idx += 1
        
        # Temporal features
        if 'created_at' in metadata:
            features[idx] = 1.0
            idx += 1
        
        if 'updated_at' in metadata:
            features[idx] = 1.0
            idx += 1
        
        # Fill remaining with random noise for uniqueness
        remaining = self.config.node_embedding_dim - idx
        if remaining > 0:
            features[idx:] = np.random.randn(remaining) * 0.01
        
        return features.astype(np.float32)
    
    def compute_edge_weight(
        self,
        source_node: LineageNode,
        target_node: LineageNode
    ) -> float:
        """
        Compute edge weight based on metadata similarity.
        
        w(u, v) = α * semantic_sim(u, v) + β * structural_sim(u, v)
        
        Args:
            source_node: Source node
            target_node: Target node
            
        Returns:
            Edge weight in [0, 1]
        """
        if source_node.feature_vector is None or target_node.feature_vector is None:
            return 0.0
        
        # Semantic similarity (cosine similarity of feature vectors)
        semantic_sim = cosine_similarity(
            source_node.feature_vector.reshape(1, -1),
            target_node.feature_vector.reshape(1, -1)
        )[0, 0]
        semantic_sim = (semantic_sim + 1) / 2  # Normalize to [0, 1]
        
        # Structural similarity (based on attribute overlap)
        src_attrs = set(source_node.attributes.keys())
        tgt_attrs = set(target_node.attributes.keys())
        
        if len(src_attrs | tgt_attrs) > 0:
            structural_sim = len(src_attrs & tgt_attrs) / len(src_attrs | tgt_attrs)
        else:
            structural_sim = 0.0
        
        # Weighted combination
        weight = (
            self.config.semantic_similarity_weight * semantic_sim +
            self.config.structural_similarity_weight * structural_sim
        )
        
        return float(weight)
    
    def add_dataset(
        self,
        dataset_id: str,
        name: str,
        metadata: Dict
    ) -> None:
        """Add a dataset node to the lineage graph."""
        features = self.extract_metadata_features(metadata)
        
        node = LineageNode(
            node_id=dataset_id,
            node_type='dataset',
            name=name,
            attributes=metadata,
            feature_vector=features
        )
        
        self.graph.add_node(node)
    
    def add_transformation(
        self,
        transform_id: str,
        name: str,
        input_datasets: List[str],
        output_datasets: List[str],
        metadata: Dict
    ) -> None:
        """Add a transformation node and its edges to the graph."""
        features = self.extract_metadata_features(metadata)
        
        node = LineageNode(
            node_id=transform_id,
            node_type='transformation',
            name=name,
            attributes=metadata,
            feature_vector=features
        )
        
        self.graph.add_node(node)
        
        # Add edges from inputs to transformation
        for input_id in input_datasets:
            if input_id in self.graph.nodes:
                source_node = self.graph.nodes[input_id]
                weight = self.compute_edge_weight(source_node, node)
                
                edge = LineageEdge(
                    source_id=input_id,
                    target_id=transform_id,
                    edge_type='direct',
                    weight=weight,
                    confidence=1.0,
                    metadata={'relationship': 'input'}
                )
                self.graph.add_edge(edge)
        
        # Add edges from transformation to outputs
        for output_id in output_datasets:
            if output_id in self.graph.nodes:
                target_node = self.graph.nodes[output_id]
                weight = self.compute_edge_weight(node, target_node)
                
                edge = LineageEdge(
                    source_id=transform_id,
                    target_id=output_id,
                    edge_type='direct',
                    weight=weight,
                    confidence=1.0,
                    metadata={'relationship': 'output'}
                )
                self.graph.add_edge(edge)
    
    def infer_implicit_lineage(
        self,
        threshold: Optional[float] = None
    ) -> List[LineageEdge]:
        """
        Infer implicit lineage relationships using GNN embeddings.
        
        Args:
            threshold: Edge weight threshold (uses config default if None)
            
        Returns:
            List of inferred lineage edges
        """
        threshold = threshold or self.config.edge_weight_threshold
        
        if len(self.graph.nodes) < 2:
            return []
        
        # Convert graph to PyG format
        data = self.graph.to_pyg_data()
        data = data.to(self.device)
        
        # Get node embeddings
        self.model.eval()
        with torch.no_grad():
            embeddings = self.model(data.x, data.edge_index)
            embeddings = embeddings.cpu().numpy()
        
        # Compute pairwise similarities
        similarities = cosine_similarity(embeddings)
        
        # Find potential implicit edges
        inferred_edges = []
        node_ids = data.node_ids
        
        for i in range(len(node_ids)):
            for j in range(len(node_ids)):
                if i == j:
                    continue
                
                src_id, tgt_id = node_ids[i], node_ids[j]
                
                # Skip if edge already exists
                if (src_id, tgt_id) in self.graph.edges:
                    continue
                
                # Check similarity threshold
                sim = (similarities[i, j] + 1) / 2  # Normalize to [0, 1]
                
                if sim >= threshold:
                    edge = LineageEdge(
                        source_id=src_id,
                        target_id=tgt_id,
                        edge_type='inferred',
                        weight=float(sim),
                        confidence=float(sim),
                        metadata={'inference_method': 'gnn_similarity'}
                    )
                    inferred_edges.append(edge)
        
        # Add inferred edges to graph
        for edge in inferred_edges:
            self.graph.add_edge(edge)
        
        logger.info(f"Inferred {len(inferred_edges)} implicit lineage relationships")
        
        return inferred_edges
    
    def train_model(
        self,
        num_epochs: int = 100,
        learning_rate: float = 0.001
    ) -> Dict[str, List[float]]:
        """
        Train the GNN model using link prediction objective.
        
        Args:
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            
        Returns:
            Training history
        """
        if len(self.graph.edges) < 10:
            logger.warning("Insufficient edges for training. Need at least 10 edges.")
            return {'loss': []}
        
        data = self.graph.to_pyg_data()
        data = data.to(self.device)
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=1e-5
        )
        
        history = {'loss': []}
        
        self.model.train()
        for epoch in range(num_epochs):
            self.optimizer.zero_grad()
            
            # Get embeddings
            embeddings = self.model(data.x, data.edge_index)
            
            # Link prediction loss
            pos_edges = data.edge_index
            
            # Sample negative edges
            num_neg = pos_edges.size(1)
            neg_src = torch.randint(0, data.x.size(0), (num_neg,), device=self.device)
            neg_tgt = torch.randint(0, data.x.size(0), (num_neg,), device=self.device)
            
            # Positive scores
            pos_scores = (embeddings[pos_edges[0]] * embeddings[pos_edges[1]]).sum(dim=1)
            
            # Negative scores
            neg_scores = (embeddings[neg_src] * embeddings[neg_tgt]).sum(dim=1)
            
            # Binary cross-entropy loss
            pos_loss = F.binary_cross_entropy_with_logits(
                pos_scores, 
                torch.ones_like(pos_scores)
            )
            neg_loss = F.binary_cross_entropy_with_logits(
                neg_scores,
                torch.zeros_like(neg_scores)
            )
            
            loss = pos_loss + neg_loss
            
            loss.backward()
            self.optimizer.step()
            
            history['loss'].append(loss.item())
            
            if (epoch + 1) % 20 == 0:
                logger.info(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}")
        
        self._trained = True
        return history
    
    def get_full_lineage(self, dataset_id: str) -> Dict:
        """
        Get complete lineage information for a dataset.
        
        Args:
            dataset_id: Dataset identifier
            
        Returns:
            Dictionary with upstream and downstream lineage
        """
        if dataset_id not in self.graph.nodes:
            return {'error': f'Dataset {dataset_id} not found'}
        
        # BFS for upstream lineage
        upstream = []
        visited = {dataset_id}
        queue = list(self.graph.get_predecessors(dataset_id))
        
        while queue:
            node_id = queue.pop(0)
            if node_id not in visited:
                visited.add(node_id)
                upstream.append(node_id)
                queue.extend(self.graph.get_predecessors(node_id))
        
        # BFS for downstream lineage
        downstream = []
        visited = {dataset_id}
        queue = list(self.graph.get_neighbors(dataset_id))
        
        while queue:
            node_id = queue.pop(0)
            if node_id not in visited:
                visited.add(node_id)
                downstream.append(node_id)
                queue.extend(self.graph.get_neighbors(node_id))
        
        return {
            'dataset_id': dataset_id,
            'upstream': upstream,
            'downstream': downstream,
            'completeness': self.graph.compute_lineage_completeness(dataset_id)
        }
