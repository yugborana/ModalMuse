from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from qdrant_client import QdrantClient, AsyncQdrantClient, models
from qdrant_client.http.models import SparseVector
from llama_index.vector_stores.qdrant import QdrantVectorStore

# Placeholder for LlamaIndex Node ID (UUID)
PointID = str 

@dataclass
class HybridPoint:
    """A data structure to hold all information for a single Qdrant point."""
    id: PointID
    text_chunk: str
    dense_vector: List[float]
    sparse_vector_indices: List[int]
    sparse_vector_values: List[float]
    # Payload for the text point
    payload: Dict[str, Any] 

@dataclass
class ImagePoint:
    """A data structure for a single image point."""
    id: PointID
    image_path: str
    image_vector: List[float]
    # Payload for the image point
    payload: Dict[str, Any]

# qdrant_manager.py (Continuation)

class QdrantManager:
    """Manages Qdrant client, collections, and hybrid/multi-modal upserts."""
    
    def __init__(self, url: Optional[str] = "http://localhost:6333", api_key: Optional[str] = None, timeout: int = 60):
        # Sync client for indexing and sync operations
        self.client = QdrantClient(url=url, api_key=api_key, timeout=timeout)
        # Async client for async retrieval operations
        self.async_client = AsyncQdrantClient(url=url, api_key=api_key, timeout=timeout)
        self.text_collection_name = "multimodal_text_index"
        self.image_collection_name = "multimodal_image_index"

    # --- Collection Management ---

    def create_text_collection(self, dense_dim: int, sparse_dim: int):
        """Creates the text collection configured for BGE (dense) and SPLADE (sparse)."""
        print(f"[SEARCH] Checking text collection '{self.text_collection_name}'...")
        try:
            # Check if collection already exists
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]
            print(f"   Existing collections: {collection_names}")
            
            if self.text_collection_name not in collection_names:
                print(f"   Creating collection '{self.text_collection_name}' with dense_dim={dense_dim}...")
                self.client.create_collection(
                    collection_name=self.text_collection_name,
                    vectors_config={
                        "text-dense": models.VectorParams(
                            size=dense_dim, 
                            distance=models.Distance.COSINE
                        )
                    },
                    sparse_vectors_config={
                        "text-sparse": models.SparseVectorParams()
                    }
                )
                print(f"[OK] Created text collection: {self.text_collection_name}")
            else:
                print(f"[OK] Text collection already exists: {self.text_collection_name}")
        except Exception as e:
            print(f"[ERROR] Failed to create text collection: {e}")
            raise

    def create_image_collection(self, image_dim: int):
        """Creates the image collection configured for CLIP (dense).
        
        Uses get_or_create pattern to avoid destroying existing indexed data.
        """
        # Check if collection already exists
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if self.image_collection_name not in collection_names:
            self.client.create_collection(
                collection_name=self.image_collection_name,
                vectors_config=models.VectorParams(
                    # CLIP image vector config
                    size=image_dim, 
                    distance=models.Distance.COSINE
                )
            )
            print(f"[OK] Created image collection: {self.image_collection_name}")
        else:
            print(f"[OK] Image collection already exists: {self.image_collection_name}")

    # --- Upsert Logic ---
    
    def upsert_text_points(self, points: List[HybridPoint], batch_size: int = 128):
        """Inserts a list of HybridPoints into the text collection."""
        
        # Convert custom HybridPoint objects into Qdrant PointStructs
        qdrant_points = [
            models.PointStruct(
                id=p.id,
                # Key step: Define a dictionary of named vectors: 'vector' for dense, 'sparse' for sparse.
                vector={
                    # Dense vector (BGE) - Named implicitly as the primary vector
                    "text-dense": p.dense_vector, 
                    # Sparse vector (SPLADE) - Named explicitly for the sparse field
                    "text-sparse": SparseVector(
                        indices=p.sparse_vector_indices, 
                        values=p.sparse_vector_values
                    )
                },
                # Payload includes the original text chunk and other metadata
                payload={**p.payload, "text_chunk": p.text_chunk}
            )
            for p in points
        ]
        
        # Use the built-in upsert method with a batch iterator for efficiency
        self.client.upload_points(
            collection_name=self.text_collection_name,
            points=qdrant_points,
            wait=True,
            batch_size=batch_size
        )

    def get_existing_text_ids(self, point_ids: List[str]) -> set:
        """Check which point IDs already exist in the text collection."""
        if not point_ids:
            return set()
        
        try:
            # Retrieve points by IDs - only returns existing ones
            result = self.client.retrieve(
                collection_name=self.text_collection_name,
                ids=point_ids,
                with_payload=False,
                with_vectors=False
            )
            return {str(point.id) for point in result}
        except Exception as e:
            print(f"[WARN] Could not check existing text IDs: {e}")
            return set()  # On error, assume none exist (will upsert all)

    def get_existing_image_ids(self, point_ids: List[str]) -> set:
        """Check which point IDs already exist in the image collection."""
        if not point_ids:
            return set()
        
        try:
            result = self.client.retrieve(
                collection_name=self.image_collection_name,
                ids=point_ids,
                with_payload=False,
                with_vectors=False
            )
            return {str(point.id) for point in result}
        except Exception as e:
            print(f"[WARN] Could not check existing image IDs: {e}")
            return set()

    def upsert_image_points(self, points: List[ImagePoint], batch_size: int = 128):
        """Inserts a list of ImagePoints into the image collection."""
        
        qdrant_points = [
            models.PointStruct(
                id=p.id,
                # Image vector (CLIP) - Standard dense vector
                vector=p.image_vector,
                # Payload stores the image path, essential for the VLM later
                payload={**p.payload, "image_path": p.image_path}
            )
            for p in points
        ]
        
        self.client.upload_points(
            collection_name=self.image_collection_name,
            points=qdrant_points,
            wait=True,
            batch_size=batch_size
        )

    # --- LlamaIndex Integration ---

    def get_text_vector_store(self) -> QdrantVectorStore:
        """Returns the LlamaIndex QdrantVectorStore for the text collection."""
        return QdrantVectorStore(
            client=self.client, 
            collection_name=self.text_collection_name,
            # Explicitly name the dense vector field for LlamaIndex to use it
            # The sparse vector is handled in retrieval.py directly via a custom retriever.
            vector_name="text-dense",
            sparse_vector_name="text-sparse",
            text_key="text_chunk"
        )

    def get_image_vector_store(self) -> QdrantVectorStore:
        """Returns the LlamaIndex QdrantVectorStore for the image collection."""
        return QdrantVectorStore(
            client=self.client, 
            collection_name=self.image_collection_name,
            text_key="image_path"
        )