# rrf_reranker.py - Reciprocal Rank Fusion Reranking Utility
"""
Implements Reciprocal Rank Fusion (RRF) for combining rankings from multiple retrievers.
Can be used standalone or integrated with LlamaIndex retrievers.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class RankedItem:
    """Represents an item with its ID, content, and score."""
    id: str
    content: Any
    score: float
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


def reciprocal_rank_fusion(
    rankings: List[List[RankedItem]],
    k: int = 60,
    weights: Optional[List[float]] = None
) -> List[RankedItem]:
    """
    Combine multiple rankings using Reciprocal Rank Fusion (RRF).
    
    RRF score for document d = Σ (weight_i / (k + rank_i(d)))
    where rank_i(d) is the rank of document d in ranking i (1-indexed).
    
    Args:
        rankings: List of ranked lists, each containing RankedItem objects.
                 Items should be ordered by their original ranking (best first).
        k: Constant to prevent high-ranked items from dominating (default: 60).
           Higher k values give more weight to lower-ranked items.
        weights: Optional weights for each ranking list. If None, all rankings
                are weighted equally.
    
    Returns:
        List of RankedItem objects sorted by RRF score (highest first).
        The score attribute contains the RRF score.
    
    Example:
        >>> dense_results = [RankedItem("doc1", "...", 0.9), RankedItem("doc2", "...", 0.8)]
        >>> sparse_results = [RankedItem("doc2", "...", 0.95), RankedItem("doc3", "...", 0.7)]
        >>> fused = reciprocal_rank_fusion([dense_results, sparse_results])
    """
    if not rankings:
        return []
    
    # Normalize weights
    if weights is None:
        weights = [1.0] * len(rankings)
    else:
        if len(weights) != len(rankings):
            raise ValueError("Number of weights must match number of rankings")
    
    # Normalize weights to sum to 1
    weight_sum = sum(weights)
    weights = [w / weight_sum for w in weights]
    
    # Calculate RRF scores
    rrf_scores: Dict[str, float] = defaultdict(float)
    item_map: Dict[str, RankedItem] = {}
    
    for ranking_idx, ranking in enumerate(rankings):
        weight = weights[ranking_idx]
        
        for rank, item in enumerate(ranking, start=1):
            # RRF formula: weight / (k + rank)
            rrf_score = weight / (k + rank)
            rrf_scores[item.id] += rrf_score
            
            # Keep track of the item (prefer first occurrence for content)
            if item.id not in item_map:
                item_map[item.id] = item
    
    # Create fused results
    fused_results = []
    for item_id, rrf_score in rrf_scores.items():
        original_item = item_map[item_id]
        fused_results.append(RankedItem(
            id=item_id,
            content=original_item.content,
            score=rrf_score,
            metadata={
                **original_item.metadata,
                "rrf_score": rrf_score,
                "original_score": original_item.score
            }
        ))
    
    # Sort by RRF score (descending)
    fused_results.sort(key=lambda x: x.score, reverse=True)
    
    return fused_results


def rrf_from_scores(
    score_lists: List[List[Tuple[str, float, Any]]],
    k: int = 60,
    weights: Optional[List[float]] = None
) -> List[Tuple[str, float, Any]]:
    """
    Convenience function for RRF with raw score tuples.
    
    Args:
        score_lists: List of lists containing (id, score, content) tuples.
                    Each list should be sorted by score (descending).
        k: RRF constant (default: 60)
        weights: Optional weights for each score list
    
    Returns:
        List of (id, rrf_score, content) tuples sorted by RRF score.
    """
    # Convert to RankedItem format
    rankings = []
    for score_list in score_lists:
        ranking = [
            RankedItem(id=item[0], content=item[2], score=item[1])
            for item in score_list
        ]
        rankings.append(ranking)
    
    # Apply RRF
    fused = reciprocal_rank_fusion(rankings, k=k, weights=weights)
    
    # Convert back to tuple format
    return [(item.id, item.score, item.content) for item in fused]


class RRFReranker:
    """
    Reranker class that combines multiple result sets using RRF.
    
    This class is designed to work with LlamaIndex NodeWithScore objects
    or any objects with an 'id_' or 'node_id' attribute.
    """
    
    def __init__(self, k: int = 60, weights: Optional[List[float]] = None):
        """
        Initialize the RRF Reranker.
        
        Args:
            k: RRF constant (default: 60)
            weights: Optional default weights for fusion
        """
        self.k = k
        self.default_weights = weights
    
    def fuse_node_results(
        self,
        result_sets: List[List[Any]],
        weights: Optional[List[float]] = None,
        top_n: Optional[int] = None
    ) -> List[Any]:
        """
        Fuse multiple sets of LlamaIndex NodeWithScore results using RRF.
        
        Args:
            result_sets: List of result sets, each containing NodeWithScore objects
            weights: Optional weights for each result set
            top_n: Optional limit on number of results to return
        
        Returns:
            List of NodeWithScore objects with updated scores (RRF scores)
        """
        if not result_sets:
            return []
        
        weights = weights or self.default_weights
        
        # Extract node IDs and create rankings
        rankings = []
        node_map = {}
        
        for result_set in result_sets:
            ranking = []
            for node_with_score in result_set:
                # Try different ways to get the node ID
                node = getattr(node_with_score, 'node', node_with_score)
                node_id = getattr(node, 'node_id', None) or getattr(node, 'id_', str(id(node)))
                
                ranking.append(RankedItem(
                    id=node_id,
                    content=node_with_score,
                    score=getattr(node_with_score, 'score', 0.0)
                ))
                
                if node_id not in node_map:
                    node_map[node_id] = node_with_score
            
            rankings.append(ranking)
        
        # Apply RRF
        fused = reciprocal_rank_fusion(rankings, k=self.k, weights=weights)
        
        # Update scores on original nodes
        results = []
        for item in fused:
            original_node = node_map[item.id]
            
            # Store the original reranker score in metadata before replacing with RRF score
            # This allows downstream code to display the meaningful score
            if hasattr(original_node, 'node') and hasattr(original_node.node, 'metadata'):
                original_node.node.metadata['original_score'] = item.metadata.get('original_score', original_node.score)
            
            # Update score to RRF score
            if hasattr(original_node, 'score'):
                original_node.score = item.score
            
            results.append(original_node)
        
        if top_n:
            results = results[:top_n]
        
        return results
    
    def fuse_dict_results(
        self,
        result_sets: List[List[Dict[str, Any]]],
        id_key: str = "id",
        score_key: str = "score",
        weights: Optional[List[float]] = None,
        top_n: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Fuse multiple sets of dictionary results using RRF.
        
        Args:
            result_sets: List of result sets, each containing dict objects
            id_key: Key to use for document ID
            score_key: Key to use for score
            weights: Optional weights for each result set
            top_n: Optional limit on number of results
        
        Returns:
            List of dicts with 'rrf_score' added
        """
        if not result_sets:
            return []
        
        weights = weights or self.default_weights
        
        # Create rankings from dicts
        rankings = []
        item_map = {}
        
        for result_set in result_sets:
            ranking = []
            for item in result_set:
                item_id = str(item.get(id_key, id(item)))
                ranking.append(RankedItem(
                    id=item_id,
                    content=item,
                    score=item.get(score_key, 0.0)
                ))
                if item_id not in item_map:
                    item_map[item_id] = item
            rankings.append(ranking)
        
        # Apply RRF
        fused = reciprocal_rank_fusion(rankings, k=self.k, weights=weights)
        
        # Create result dicts
        results = []
        for item in fused:
            result = {**item_map[item.id], "rrf_score": item.score}
            results.append(result)
        
        if top_n:
            results = results[:top_n]
        
        return results


# ============================================================================
#                          INTEGRATION EXAMPLE
# ============================================================================

def example_usage():
    """Example demonstrating RRF usage with mock data."""
    
    # Example 1: Basic RRF with RankedItem
    print("=" * 50)
    print("Example 1: Basic RRF")
    print("=" * 50)
    
    # Dense retrieval results (semantic similarity)
    dense_results = [
        RankedItem(id="doc1", content="Machine learning is...", score=0.95),
        RankedItem(id="doc2", content="Deep learning neural...", score=0.89),
        RankedItem(id="doc4", content="AI fundamentals...", score=0.75),
    ]
    
    # Sparse retrieval results (keyword matching - BM25/SPLADE)
    sparse_results = [
        RankedItem(id="doc2", content="Deep learning neural...", score=0.92),
        RankedItem(id="doc3", content="Neural network basics...", score=0.88),
        RankedItem(id="doc1", content="Machine learning is...", score=0.70),
    ]
    
    # Fuse with RRF
    fused = reciprocal_rank_fusion([dense_results, sparse_results], k=60)
    
    print("\nFused Results (RRF):")
    for i, item in enumerate(fused, 1):
        print(f"  {i}. {item.id}: RRF={item.score:.4f} (original={item.metadata['original_score']:.2f})")
    
    # Example 2: Weighted RRF
    print("\n" + "=" * 50)
    print("Example 2: Weighted RRF (dense=0.7, sparse=0.3)")
    print("=" * 50)
    
    fused_weighted = reciprocal_rank_fusion(
        [dense_results, sparse_results],
        k=60,
        weights=[0.7, 0.3]
    )
    
    print("\nWeighted Fused Results:")
    for i, item in enumerate(fused_weighted, 1):
        print(f"  {i}. {item.id}: RRF={item.score:.4f}")
    
    # Example 3: Using RRFReranker class
    print("\n" + "=" * 50)
    print("Example 3: RRFReranker with dict results")
    print("=" * 50)
    
    reranker = RRFReranker(k=60)
    
    dict_results_1 = [
        {"id": "a", "text": "Document A", "score": 0.9},
        {"id": "b", "text": "Document B", "score": 0.8},
    ]
    dict_results_2 = [
        {"id": "b", "text": "Document B", "score": 0.95},
        {"id": "c", "text": "Document C", "score": 0.85},
    ]
    
    fused_dicts = reranker.fuse_dict_results([dict_results_1, dict_results_2])
    
    print("\nFused Dict Results:")
    for item in fused_dicts:
        print(f"  {item['id']}: {item['text']}, RRF={item['rrf_score']:.4f}")


if __name__ == "__main__":
    example_usage()
