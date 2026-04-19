# rrf_reranker.py - Reciprocal Rank Fusion for Dense + Sparse Hybrid Search
"""
Fuses dense (Jina) and sparse (BM25) search results using RRF.
Operates on LlamaIndex NodeWithScore objects for direct pipeline integration.
"""

from typing import List, Optional
from collections import defaultdict

from llama_index.core.schema import NodeWithScore

import config


def rrf_fuse(
    dense_nodes: List[NodeWithScore],
    sparse_nodes: List[NodeWithScore],
    k: int = None,
    top_n: Optional[int] = None,
    dense_weight: float = 1.0,
    sparse_weight: float = 1.0,
) -> List[NodeWithScore]:
    """
    Fuse dense and sparse results using Reciprocal Rank Fusion.
    
    RRF score for node d = w_dense/(k + rank_dense(d)) + w_sparse/(k + rank_sparse(d))
    
    Nodes appearing in both lists get boosted. Nodes in only one list
    still appear but with a single-source score.
    
    Args:
        dense_nodes:   Dense search results (ordered best-first by cosine similarity).
        sparse_nodes:  Sparse search results (ordered best-first by BM25 score).
        k:             RRF constant (default: config.RRF_K). Higher = more uniform weighting.
        top_n:         Max results to return (default: all).
        dense_weight:  Weight for dense ranking (default: 1.0).
        sparse_weight: Weight for sparse ranking (default: 1.0).
    
    Returns:
        List of NodeWithScore sorted by RRF score (highest first).
    """
    if k is None:
        k = config.RRF_K
    
    if not dense_nodes and not sparse_nodes:
        return []
    
    # Normalize weights
    total_w = dense_weight + sparse_weight
    w_d = dense_weight / total_w
    w_s = sparse_weight / total_w
    
    # Map: node_id → accumulated RRF score
    rrf_scores: dict[str, float] = defaultdict(float)
    # Map: node_id → NodeWithScore (first occurrence wins)
    node_map: dict[str, NodeWithScore] = {}
    
    # Score dense results
    for rank, nws in enumerate(dense_nodes, start=1):
        node_id = nws.node.node_id
        rrf_scores[node_id] += w_d / (k + rank)
        if node_id not in node_map:
            node_map[node_id] = nws
    
    # Score sparse results
    for rank, nws in enumerate(sparse_nodes, start=1):
        node_id = nws.node.node_id
        rrf_scores[node_id] += w_s / (k + rank)
        if node_id not in node_map:
            node_map[node_id] = nws
    
    # Build fused list sorted by RRF score
    fused = []
    for node_id, score in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True):
        original = node_map[node_id]
        fused.append(NodeWithScore(node=original.node, score=score))
    
    if top_n:
        fused = fused[:top_n]
    
    return fused
