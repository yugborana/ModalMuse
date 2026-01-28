# cache.py - Performance Caching Layer for ModalMuse
"""
LRU + TTL caching for query embeddings and responses.
Reduces latency for repeated/similar queries.
"""

import hashlib
import time
from typing import Any, Optional, Dict, List
from functools import wraps
from collections import OrderedDict
import threading


class TTLCache:
    """Thread-safe LRU cache with TTL expiration."""
    
    def __init__(self, maxsize: int = 100, ttl: float = 3600.0):
        """
        Initialize cache.
        
        Args:
            maxsize: Maximum number of items to store
            ttl: Time-to-live in seconds (default 1 hour)
        """
        self.maxsize = maxsize
        self.ttl = ttl
        self._cache: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache, returns None if expired or missing."""
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None
            
            value, timestamp = self._cache[key]
            
            # Check TTL
            if time.time() - timestamp > self.ttl:
                del self._cache[key]
                self._misses += 1
                return None
            
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self._hits += 1
            return value
    
    def set(self, key: str, value: Any) -> None:
        """Store value in cache."""
        with self._lock:
            # Remove oldest if at capacity
            if len(self._cache) >= self.maxsize:
                self._cache.popitem(last=False)
            
            self._cache[key] = (value, time.time())
    
    def clear(self) -> None:
        """Clear all cached items."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = (self._hits / total * 100) if total > 0 else 0
            return {
                "size": len(self._cache),
                "maxsize": self.maxsize,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": f"{hit_rate:.1f}%"
            }


# ═══════════════════════════════════════════════════════════════════
# GLOBAL CACHE INSTANCES
# ═══════════════════════════════════════════════════════════════════

# Query embedding cache - key: query string, value: embedding vector
_query_embedding_cache = TTLCache(maxsize=200, ttl=3600.0)

# Full response cache - key: query hash, value: (answer, sources)
_response_cache = TTLCache(maxsize=50, ttl=1800.0)  # 30 min TTL

# Reranking cache - key: hash(query + docs), value: rerank results
_rerank_cache = TTLCache(maxsize=100, ttl=1800.0)


def get_query_embedding_cache() -> TTLCache:
    """Get the query embedding cache."""
    return _query_embedding_cache


def get_response_cache() -> TTLCache:
    """Get the full response cache."""
    return _response_cache


def get_rerank_cache() -> TTLCache:
    """Get the reranking cache."""
    return _rerank_cache


def hash_key(*args) -> str:
    """Create a hash key from arguments."""
    content = "|".join(str(a) for a in args)
    return hashlib.md5(content.encode()).hexdigest()


def clear_all_caches():
    """Clear all caches (call after indexing new documents)."""
    _query_embedding_cache.clear()
    _response_cache.clear()
    _rerank_cache.clear()
    print("[CACHE] All caches cleared")


def get_all_cache_stats() -> Dict[str, Any]:
    """Get statistics for all caches."""
    return {
        "embedding_cache": _query_embedding_cache.stats(),
        "response_cache": _response_cache.stats(),
        "rerank_cache": _rerank_cache.stats(),
    }


# ═══════════════════════════════════════════════════════════════════
# DECORATORS FOR EASY CACHING
# ═══════════════════════════════════════════════════════════════════

def cached_embedding(func):
    """Decorator to cache embedding results."""
    @wraps(func)
    async def wrapper(self, query: str, *args, **kwargs):
        cache = get_query_embedding_cache()
        
        # Check cache
        cached = cache.get(query)
        if cached is not None:
            print(f"   [CACHE HIT] Embedding for: '{query[:30]}...'")
            return cached
        
        # Compute and cache
        result = await func(self, query, *args, **kwargs)
        cache.set(query, result)
        return result
    
    return wrapper


def cached_rerank(func):
    """Decorator to cache reranking results."""
    @wraps(func)
    async def wrapper(self, query: str, documents: List[str], *args, **kwargs):
        cache = get_rerank_cache()
        
        # Create cache key from query + document hashes
        doc_hash = hash_key(*documents[:5])  # Use first 5 docs for key
        cache_key = hash_key(query, doc_hash)
        
        # Check cache
        cached = cache.get(cache_key)
        if cached is not None:
            print(f"   [CACHE HIT] Reranking for: '{query[:30]}...'")
            return cached
        
        # Compute and cache
        result = await func(self, query, documents, *args, **kwargs)
        cache.set(cache_key, result)
        return result
    
    return wrapper
