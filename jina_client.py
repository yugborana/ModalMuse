# jina_client.py - Jina AI Embeddings and Reranking Client
"""
Client for Jina AI API - handles text/image embeddings and reranking.
Supports both sync (requests) and async (httpx) operations.
Uses shared connection pool for performance.
"""

import requests
import httpx
import base64
from typing import List, Optional, Union
from pathlib import Path

import config

# ═══════════════════════════════════════════════════════════════════
# SHARED CONNECTION POOL (Performance Optimization)
# Reuses TCP connections across requests - saves ~50-100ms per request
# ═══════════════════════════════════════════════════════════════════

_async_client: Optional[httpx.AsyncClient] = None

def get_async_client() -> httpx.AsyncClient:
    """Get or create shared async HTTP client with connection pooling."""
    global _async_client
    if _async_client is None or _async_client.is_closed:
        _async_client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, connect=10.0),
            limits=httpx.Limits(
                max_connections=20,
                max_keepalive_connections=10,
                keepalive_expiry=30.0
            ),
            http2=True  # Enable HTTP/2 for multiplexing
        )
    return _async_client

async def close_async_client():
    """Close the shared async client (call on shutdown)."""
    global _async_client
    if _async_client is not None and not _async_client.is_closed:
        await _async_client.aclose()
        _async_client = None


class JinaEmbeddings:
    """Jina AI Embeddings client for text and images (sync and async)."""
    
    API_URL = "https://api.jina.ai/v1/embeddings"
    
    def __init__(self, api_key: Optional[str] = None, dimensions: int = 1024):
        """
        Initialize Jina embeddings client.
        
        Args:
            api_key: Jina API key (defaults to config.JINA_API_KEY)
            dimensions: Output embedding dimension (256-2048, default 1024)
        """
        self.api_key = api_key or config.JINA_API_KEY
        if not self.api_key:
            raise ValueError("JINA_API_KEY is not set. Get one at jina.ai/embeddings")
        
        self.dimensions = dimensions
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
    
    @property
    def embed_dim(self) -> int:
        """Return embedding dimension for Qdrant collection setup."""
        return self.dimensions
    
    # ═══════════════════════════════════════════════════════════════════
    # SYNC METHODS (using requests)
    # ═══════════════════════════════════════════════════════════════════
    
    def embed_texts(
        self, 
        texts: List[str], 
        task: str = "retrieval.passage",
        late_chunking: bool = False
    ) -> List[List[float]]:
        """Embed multiple texts (sync)."""
        data = {
            "input": texts,
            "model": "jina-embeddings-v4",
            "dimensions": self.dimensions,
            "task": task,
            "late_chunking": late_chunking,
        }
        
        response = requests.post(self.API_URL, headers=self.headers, json=data)
        response.raise_for_status()
        
        return [d["embedding"] for d in response.json()["data"]]
    
    def embed_query(self, query: str) -> List[float]:
        """Embed a single query (sync)."""
        return self.embed_texts([query], task="retrieval.query")[0]
    
    def embed_images(
        self, 
        images: List[Union[str, Path]],
        is_base64: bool = False
    ) -> List[List[float]]:
        """Embed images (sync)."""
        input_data = self._prepare_image_input(images, is_base64)
        if not input_data:
            return []
        
        data = {
            "input": input_data,
            "model": "jina-embeddings-v4",
            "dimensions": self.dimensions,
        }
        
        response = requests.post(self.API_URL, headers=self.headers, json=data)
        
        if response.status_code != 200:
            # Log the actual error details
            try:
                error_detail = response.json()
                print(f"[ERROR] Jina API error: {error_detail}")
            except:
                print(f"[ERROR] Jina API error: {response.text[:500]}")
        
        response.raise_for_status()
        
        return [d["embedding"] for d in response.json()["data"]]
    
    # ═══════════════════════════════════════════════════════════════════
    # ASYNC METHODS (using httpx)
    # ═══════════════════════════════════════════════════════════════════
    
    async def aembed_texts(
        self, 
        texts: List[str], 
        task: str = "retrieval.passage",
        late_chunking: bool = False
    ) -> List[List[float]]:
        """Embed multiple texts (async with connection pooling)."""
        data = {
            "input": texts,
            "model": "jina-embeddings-v4",
            "dimensions": self.dimensions,
            "task": task,
            "late_chunking": late_chunking,
        }
        
        client = get_async_client()
        response = await client.post(self.API_URL, headers=self.headers, json=data)
        response.raise_for_status()
        
        return [d["embedding"] for d in response.json()["data"]]
    
    async def aembed_query(self, query: str) -> List[float]:
        """Embed a single query (async with caching).
        
        Uses two cache layers:
        1. In-memory LRU cache (fast, volatile)
        2. Supabase query_cache (slower, persistent)
        """
        from cache import get_query_embedding_cache
        
        cache = get_query_embedding_cache()
        
        # Check in-memory cache first (fast)
        cached = cache.get(query)
        if cached is not None:
            print(f"   [CACHE HIT] In-memory cache: '{query[:40]}...'")
            return cached
        
        # Check Supabase cache (persistent)
        try:
            from supabase_client import get_cached_embedding
            supabase_cached = await get_cached_embedding(query)
            if supabase_cached is not None:
                print(f"   [CACHE HIT] Supabase cache: '{query[:40]}...'")
                # Store in memory cache for faster future access
                cache.set(query, supabase_cached)
                return supabase_cached
        except Exception as e:
            print(f"   [WARN] Supabase cache lookup failed: {e}")
        
        # Compute embedding
        embeddings = await self.aembed_texts([query], task="retrieval.query")
        result = embeddings[0]
        
        # Store in both caches
        cache.set(query, result)
        
        # Store in Supabase cache (async, fire-and-forget)
        try:
            from supabase_client import cache_embedding
            await cache_embedding(query, result)
            print(f"   [CACHE] Stored in Supabase: '{query[:40]}...'")
        except Exception as e:
            print(f"   [WARN] Supabase cache store failed: {e}")
        
        return result
    
    async def aembed_images(
        self, 
        images: List[Union[str, Path]],
        is_base64: bool = False
    ) -> List[List[float]]:
        """Embed images (async with connection pooling)."""
        input_data = self._prepare_image_input(images, is_base64)
        if not input_data:
            return []
        
        data = {
            "input": input_data,
            "model": "jina-embeddings-v4",
            "dimensions": self.dimensions,
        }
        
        client = get_async_client()
        response = await client.post(self.API_URL, headers=self.headers, json=data)
        response.raise_for_status()
        
        return [d["embedding"] for d in response.json()["data"]]
    
    # ═══════════════════════════════════════════════════════════════════
    # HELPERS
    # ═══════════════════════════════════════════════════════════════════
    
    def _prepare_image_input(
        self, 
        images: List[Union[str, Path]], 
        is_base64: bool = False
    ) -> List[dict]:
        """Prepare image input for API call."""
        input_data = []
        
        for img in images:
            if is_base64:
                input_data.append({"image": img})
            elif str(img).startswith(("http://", "https://")):
                input_data.append({"image": str(img)})
            else:
                img_path = Path(img)
                if img_path.exists():
                    with open(img_path, "rb") as f:
                        b64 = base64.b64encode(f.read()).decode("utf-8")
                    suffix = img_path.suffix.lower()
                    mime = {".jpg": "jpeg", ".jpeg": "jpeg", ".png": "png", ".webp": "webp"}.get(suffix, "jpeg")
                    input_data.append({"image": f"data:image/{mime};base64,{b64}"})
                else:
                    print(f"[WARN] Image not found: {img}")
                    continue
        
        return input_data


class JinaReranker:
    """Jina AI Reranker client (sync and async)."""
    
    API_URL = "https://api.jina.ai/v1/rerank"
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or config.JINA_API_KEY
        if not self.api_key:
            raise ValueError("JINA_API_KEY is not set.")
        
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
    
    # ═══════════════════════════════════════════════════════════════════
    # SYNC METHOD
    # ═══════════════════════════════════════════════════════════════════
    
    def rerank(
        self, 
        query: str, 
        documents: List[str], 
        top_n: int = 5
    ) -> List[dict]:
        """Rerank documents (sync)."""
        if not documents:
            return []
        
        data = {
            "model": "jina-reranker-v2-base-multilingual",
            "query": query,
            "documents": documents,
            "top_n": min(top_n, len(documents)),
        }
        
        response = requests.post(self.API_URL, headers=self.headers, json=data)
        response.raise_for_status()
        
        return response.json()["results"]
    
    # ═══════════════════════════════════════════════════════════════════
    # ASYNC METHOD
    # ═══════════════════════════════════════════════════════════════════
    
    async def arerank(
        self, 
        query: str, 
        documents: List[str], 
        top_n: int = 5
    ) -> List[dict]:
        """Rerank documents (async with connection pooling)."""
        if not documents:
            return []
        
        data = {
            "model": "jina-reranker-v2-base-multilingual",
            "query": query,
            "documents": documents,
            "top_n": min(top_n, len(documents)),
        }
        
        client = get_async_client()
        response = await client.post(self.API_URL, headers=self.headers, json=data)
        response.raise_for_status()
        
        return response.json()["results"]

