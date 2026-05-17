# local_client.py - Local Infinity Server Embeddings and Reranking Client
"""
Client for Local Infinity Server API - handles text/image embeddings and reranking.
Drop-in replacement for jina_client.py — same interface, runs locally via Docker.

Usage:
    docker compose up -d   # starts Infinity server on port 7997
    # Then use LocalEmbeddings / LocalReranker everywhere instead of Jina classes
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
            timeout=httpx.Timeout(60.0, connect=10.0),
            limits=httpx.Limits(
                max_connections=20,
                max_keepalive_connections=10,
                keepalive_expiry=30.0
            ),
        )
    return _async_client

async def close_async_client():
    """Close the shared async client (call on shutdown)."""
    global _async_client
    if _async_client is not None and not _async_client.is_closed:
        await _async_client.aclose()
        _async_client = None


class LocalEmbeddings:
    """Local Embeddings client for text and images (sync and async) using Infinity.
    
    Create separate instances for text vs image embedding:
        text_embedder = LocalEmbeddings(model=config.LOCAL_TEXT_MODEL, dimensions=config.LOCAL_TEXT_DIMENSIONS)
        image_embedder = LocalEmbeddings(model=config.LOCAL_IMAGE_MODEL, dimensions=config.LOCAL_IMAGE_DIMENSIONS)
    """
    
    def __init__(self, model: str = None, dimensions: int = None):
        """
        Initialize Local embeddings client.
        
        Args:
            model: Model ID served by Infinity (defaults to config.LOCAL_TEXT_MODEL)
            dimensions: Output embedding dimension (defaults to config.LOCAL_TEXT_DIMENSIONS)
        """
        self.base_url = config.LOCAL_EMBED_URL
        self.api_url = f"{self.base_url}/embeddings"
        self.model = model or config.LOCAL_TEXT_MODEL
        self.dimensions = dimensions or config.LOCAL_TEXT_DIMENSIONS
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
    
    @property
    def embed_dim(self) -> int:
        """Return embedding dimension for Qdrant collection setup."""
        return self.dimensions
    
    # ═══════════════════════════════════════════════════════════════════
    # SYNC METHODS (using requests — used by indexer thread pool)
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
            "model": self.model
        }
        
        response = requests.post(self.api_url, headers=self.headers, json=data)
        response.raise_for_status()
        
        return [d["embedding"] for d in response.json()["data"]]
    
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
            "model": self.model
        }
        
        response = requests.post(self.api_url, headers=self.headers, json=data)
        
        if response.status_code != 200:
            print(f"[ERROR] Local Embed API error: {response.text[:500]}")
        response.raise_for_status()
        
        return [d["embedding"] for d in response.json()["data"]]
    
    # ═══════════════════════════════════════════════════════════════════
    # ASYNC METHODS (using httpx — used by retriever pipeline)
    # ═══════════════════════════════════════════════════════════════════
    
    async def aembed_texts(
        self, 
        texts: List[str], 
        task: str = "retrieval.passage",
        late_chunking: bool = False
    ) -> List[List[float]]:
        """Embed multiple texts (async)."""
        data = {
            "input": texts,
            "model": self.model
        }
        
        client = get_async_client()
        response = await client.post(self.api_url, headers=self.headers, json=data)
        response.raise_for_status()
        
        return [d["embedding"] for d in response.json()["data"]]
    
    async def aembed_query(self, query: str) -> List[float]:
        """Embed a single query (async)."""
        embeddings = await self.aembed_texts([query], task="retrieval.query")
        return embeddings[0]

    async def aembed_images(
        self, 
        images: List[Union[str, Path]],
        is_base64: bool = False
    ) -> List[List[float]]:
        """Embed images (async)."""
        input_data = self._prepare_image_input(images, is_base64)
        if not input_data:
            return []
        
        data = {
            "input": input_data,
            "model": self.model
        }
        
        client = get_async_client()
        response = await client.post(self.api_url, headers=self.headers, json=data)
        response.raise_for_status()
        
        return [d["embedding"] for d in response.json()["data"]]
    
    # ═══════════════════════════════════════════════════════════════════
    # HELPERS
    # ═══════════════════════════════════════════════════════════════════
    
    def _prepare_image_input(
        self, 
        images: List[Union[str, Path]], 
        is_base64: bool = False
    ) -> List[str]:
        """Prepare image input for Infinity API call."""
        input_data = []
        for img in images:
            if is_base64:
                input_data.append(img if str(img).startswith("data:") else f"data:image/jpeg;base64,{img}")
            elif str(img).startswith(("http://", "https://")):
                input_data.append(str(img))
            else:
                img_path = Path(img)
                if img_path.exists():
                    with open(img_path, "rb") as f:
                        b64 = base64.b64encode(f.read()).decode("utf-8")
                    suffix = img_path.suffix.lower()
                    mime = {".jpg": "jpeg", ".jpeg": "jpeg", ".png": "png", ".webp": "webp"}.get(suffix, "jpeg")
                    input_data.append(f"data:image/{mime};base64,{b64}")
                else:
                    print(f"[WARN] Image not found: {img}")
                    continue
        return input_data


class LocalReranker:
    """Local Reranker client using Infinity (sync and async)."""
    
    def __init__(self):
        self.api_url = f"{config.LOCAL_EMBED_URL}/rerank"
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        self.model = config.LOCAL_RERANK_MODEL
    
    async def arerank(
        self, 
        query: str, 
        documents: List[str], 
        top_n: int = 5
    ) -> List[dict]:
        """Rerank documents (async)."""
        if not documents:
            return []
        
        data = {
            "model": self.model,
            "query": query,
            "documents": documents,
            "top_n": min(top_n, len(documents)),
        }
        
        client = get_async_client()
        response = await client.post(self.api_url, headers=self.headers, json=data)
        response.raise_for_status()
        
        return response.json()["results"]
