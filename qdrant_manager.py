from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from qdrant_client import QdrantClient, AsyncQdrantClient, models
from qdrant_client.http.models import SparseVector
from llama_index.vector_stores.qdrant import QdrantVectorStore

# Placeholder for LlamaIndex Node ID (UUID)
PointID = str 

@dataclass
class TextPoint:
    """A data structure to hold all information for a single Qdrant text point."""
    id: PointID
    text_chunk: str
    dense_vector: List[float]
    sparse_indices: List[int] = field(default_factory=list)
    sparse_values: List[float] = field(default_factory=list)
    payload: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ImagePoint:
    """A data structure for a single image point."""
    id: PointID
    image_path: str
    image_vector: List[float]
    caption_vector: List[float] = field(default_factory=list)  # Text embedding of caption for text→image search
    payload: Dict[str, Any] = field(default_factory=dict)

# qdrant_manager.py (Continuation)

class QdrantManager:
    """Manages Qdrant client, collections, and multi-modal upserts."""
    
    def __init__(self, url: Optional[str] = "http://localhost:6333", api_key: Optional[str] = None, timeout: int = 60):
        # Sync client for indexing and sync operations
        self.client = QdrantClient(url=url, api_key=api_key, timeout=timeout)
        # Async client for async retrieval operations
        self.async_client = AsyncQdrantClient(url=url, api_key=api_key, timeout=timeout)
        self.text_collection_name = "multimodal_text_index"
        self.image_collection_name = "multimodal_image_index"

    # --- Collection Management ---

    def create_text_collection(self, dense_dim: int):
        """Creates the text collection with a dense vector index."""
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
                        "text-sparse": models.SparseVectorParams(
                            modifier=models.Modifier.IDF  # Qdrant applies IDF weighting at search time
                        )
                    }
                )
                print(f"[OK] Created text collection: {self.text_collection_name}")
            else:
                print(f"[OK] Text collection already exists: {self.text_collection_name}")
        except Exception as e:
            print(f"[ERROR] Failed to create text collection: {e}")
            raise

    def create_image_collection(self, image_dim: int):
        """Creates the image collection with a dense vector index.
        
        Uses get_or_create pattern to avoid destroying existing indexed data.
        """
        # Check if collection already exists
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if self.image_collection_name not in collection_names:
            self.client.create_collection(
                collection_name=self.image_collection_name,
                vectors_config={
                    "image-visual": models.VectorParams(
                        size=image_dim,
                        distance=models.Distance.COSINE
                    ),
                    "caption-text": models.VectorParams(
                        size=image_dim,  # Same Jina model embeds both text & images
                        distance=models.Distance.COSINE
                    )
                }
            )
            print(f"[OK] Created image collection: {self.image_collection_name} (visual + caption vectors)")
        else:
            print(f"[OK] Image collection already exists: {self.image_collection_name}")

    # --- Upsert Logic ---
    
    def upsert_text_points(self, points: List[TextPoint], batch_size: int = 128):
        """Inserts a list of TextPoints into the text collection."""
        
        qdrant_points = []
        for p in points:
            vectors = {"text-dense": p.dense_vector}
            # Add sparse vector if available
            if p.sparse_indices and p.sparse_values:
                vectors["text-sparse"] = SparseVector(
                    indices=p.sparse_indices,
                    values=p.sparse_values
                )
            qdrant_points.append(
                models.PointStruct(
                    id=p.id,
                    vector=vectors,
                    payload={**p.payload, "text_chunk": p.text_chunk}
                )
            )
        
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
        """Inserts a list of ImagePoints into the image collection.
        
        Each point has two named vectors:
          - image-visual: Jina embedding of the image pixels
          - caption-text: Jina embedding of the Groq caption (for text→image search)
        """
        qdrant_points = []
        for p in points:
            vectors = {"image-visual": p.image_vector}
            if p.caption_vector:
                vectors["caption-text"] = p.caption_vector
            qdrant_points.append(
                models.PointStruct(
                    id=p.id,
                    vector=vectors,
                    payload={**p.payload, "image_path": p.image_path}
                )
            )
        
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
            vector_name="text-dense",
            text_key="text_chunk"
        )

    def get_image_vector_store(self) -> QdrantVectorStore:
        """Returns the LlamaIndex QdrantVectorStore for the image collection."""
        return QdrantVectorStore(
            client=self.client, 
            collection_name=self.image_collection_name,
            text_key="image_path"
        )

    # --- Semantic Response Cache ---

    def create_response_cache_collection(self, dense_dim: int):
        """Create or verify the response cache collection."""
        import config
        cache_name = config.RESPONSE_CACHE_COLLECTION
        
        try:
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if cache_name not in collection_names:
                self.client.create_collection(
                    collection_name=cache_name,
                    vectors_config=models.VectorParams(
                        size=dense_dim,
                        distance=models.Distance.COSINE
                    )
                )
                print(f"[OK] Created response cache collection: {cache_name}")
            else:
                print(f"[OK] Response cache collection exists: {cache_name}")
        except Exception as e:
            print(f"[WARN] Could not create response cache collection: {e}")

    async def search_response_cache(
        self, query_embedding: List[float]
    ) -> Optional[Dict[str, Any]]:
        """Search for a semantically similar cached response.
        
        Returns the cached payload if similarity >= threshold, else None.
        """
        import config
        from datetime import datetime, timedelta
        
        try:
            results = await self.async_client.query_points(
                collection_name=config.RESPONSE_CACHE_COLLECTION,
                query=query_embedding,
                limit=1,
                with_payload=True,
                score_threshold=config.SEMANTIC_CACHE_THRESHOLD
            )
            
            if not results.points:
                return None
            
            point = results.points[0]
            payload = point.payload
            
            # Check TTL
            created_at = payload.get("created_at", "")
            if created_at:
                created_time = datetime.fromisoformat(created_at)
                if datetime.now() - created_time > timedelta(hours=config.SEMANTIC_CACHE_TTL_HOURS):
                    # Expired — delete and return None
                    try:
                        await self.async_client.delete(
                            collection_name=config.RESPONSE_CACHE_COLLECTION,
                            points_selector=models.PointIdsList(points=[point.id])
                        )
                    except Exception:
                        pass
                    return None
            
            print(f"   [CACHE HIT] Semantic match (score={point.score:.4f}): "
                  f"'{payload.get('query_text', '')[:40]}...'")
            return payload
            
        except Exception as e:
            # Collection might not exist yet — that's fine
            if "not found" not in str(e).lower():
                print(f"   [WARN] Cache search failed: {e}")
            return None

    async def store_response_cache(
        self,
        query_embedding: List[float],
        query_text: str,
        response_text: str,
        sources: List[Dict[str, Any]]
    ) -> None:
        """Store a query-response pair in the semantic cache."""
        import config
        import uuid
        from datetime import datetime
        
        try:
            # Store the new entry
            point_id = str(uuid.uuid4())
            await self.async_client.upsert(
                collection_name=config.RESPONSE_CACHE_COLLECTION,
                points=[
                    models.PointStruct(
                        id=point_id,
                        vector=query_embedding,
                        payload={
                            "query_text": query_text,
                            "response": response_text,
                            "sources": sources,
                            "created_at": datetime.now().isoformat()
                        }
                    )
                ]
            )
            
            # Enforce max entries — delete oldest if over limit
            count_result = await self.async_client.count(
                collection_name=config.RESPONSE_CACHE_COLLECTION
            )
            
            if count_result.count > config.SEMANTIC_CACHE_MAX_ENTRIES:
                # Scroll oldest entries and delete overflow
                overflow = count_result.count - config.SEMANTIC_CACHE_MAX_ENTRIES
                oldest = await self.async_client.scroll(
                    collection_name=config.RESPONSE_CACHE_COLLECTION,
                    limit=overflow,
                    order_by=models.OrderBy(
                        key="created_at",
                        direction=models.Direction.ASC
                    ),
                    with_payload=False,
                    with_vectors=False
                )
                if oldest[0]:
                    ids_to_delete = [p.id for p in oldest[0]]
                    await self.async_client.delete(
                        collection_name=config.RESPONSE_CACHE_COLLECTION,
                        points_selector=models.PointIdsList(points=ids_to_delete)
                    )
                    print(f"   [CACHE] Pruned {len(ids_to_delete)} old cache entries")
            
            print(f"   [CACHE] Stored response for: '{query_text[:40]}...'")
            
        except Exception as e:
            print(f"   [WARN] Cache store failed: {e}")

    def clear_response_cache(self) -> None:
        """Clear all cached responses (call after indexing new documents)."""
        import config
        try:
            # Delete and recreate the collection
            self.client.delete_collection(config.RESPONSE_CACHE_COLLECTION)
            self.create_response_cache_collection(1024)  # Jina v4 dim
            print("[CACHE] Response cache cleared (new documents indexed)")
        except Exception as e:
            print(f"[WARN] Could not clear response cache: {e}")