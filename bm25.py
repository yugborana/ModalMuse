# keyword.py - Local BM25 Sparse Encoder for Hybrid Search
"""
Generates sparse vectors (indices + values) from text using BM25-like weighting.
Uses HashingVectorizer — stateless, no fitting required, deployment-friendly.

Produces Qdrant-compatible sparse vectors for hybrid search alongside Jina dense embeddings.
"""

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from typing import List, Dict


class BM25SparseEncoder:
    """
    Local BM25-inspired sparse encoder for Qdrant hybrid search.
    
    Uses HashingVectorizer (no fitting needed) so it works identically
    at index time and query time without persisting any state.
    """

    def __init__(self, n_features: int = 131072, k1: float = 1.5, b: float = 0.75):
        """
        Args:
            n_features: Hash space size (2^17). Larger = fewer collisions.
            k1: BM25 term frequency saturation. Higher = raw TF matters more.
            b:  BM25 length normalization. 0 = no normalization, 1 = full.
        """
        self.k1 = k1
        self.b = b
        self.n_features = n_features

        self.vectorizer = HashingVectorizer(
            n_features=n_features,
            alternate_sign=False,   # Only positive values (TF counts)
            lowercase=True,
            stop_words='english',
            norm=None,              # Raw term frequencies, no L2 normalization
            token_pattern=r'(?u)\b\w+\b'
        )

    def encode_documents(self, documents: List[str]) -> List[Dict]:
        """
        Encode a batch of documents into sparse vectors with BM25 weighting.
        
        Args:
            documents: List of text strings to encode.
            
        Returns:
            List of dicts with 'indices' and 'values' keys for each document.
        """
        if not documents:
            return []

        # Get raw TF matrix (sparse CSR matrix)
        tf_matrix = self.vectorizer.transform(documents)

        # Document lengths for BM25 normalization
        doc_lengths = np.array([len(doc.split()) for doc in documents], dtype=np.float64)
        avg_dl = doc_lengths.mean() if len(doc_lengths) > 0 else 1.0

        # Approximate IDF from this batch's document frequencies
        n_docs = len(documents)
        # df[j] = number of documents containing term j
        df = np.asarray((tf_matrix > 0).sum(axis=0)).flatten()
        # Smoothed IDF: log((N - df + 0.5) / (df + 0.5) + 1)
        idf = np.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)

        results = []
        for i in range(n_docs):
            row = tf_matrix.getrow(i)
            cols = row.indices             # Non-zero column indices
            tfs = row.data                 # Corresponding TF values

            dl = doc_lengths[i]

            # BM25 score per term: idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl/avg_dl))
            numerator = tfs * (self.k1 + 1)
            denominator = tfs + self.k1 * (1 - self.b + self.b * dl / avg_dl)
            bm25_scores = idf[cols] * (numerator / denominator)

            # Filter out zero/negative scores
            mask = bm25_scores > 0
            indices = cols[mask].tolist()
            values = bm25_scores[mask].tolist()

            results.append({"indices": indices, "values": values})

        return results

    def encode_query(self, query: str) -> Dict:
        """
        Encode a query into a sparse vector.
        
        Uses simple log(1 + tf) weighting for query terms.
        The IDF component comes from the document-side BM25 scores.
        
        Args:
            query: Query string.
            
        Returns:
            Dict with 'indices' and 'values' keys.
        """
        tf_vector = self.vectorizer.transform([query])
        row = tf_vector.getrow(0)

        cols = row.indices
        tfs = row.data

        # Query weighting: log(1 + tf) — simple and effective
        scores = np.log1p(tfs)

        mask = scores > 0
        indices = cols[mask].tolist()
        values = scores[mask].tolist()

        return {"indices": indices, "values": values}
