# api/shared_resources.py
"""
Thread-safe shared resources for the ModalMuse API.

Provides:
- Singleton Indexer (avoids reloading models per request)
- Thread-safe TaskRegistry (prevents race conditions)
- Dedicated ThreadPoolExecutor (doesn't block API workers)
"""

import threading
import concurrent.futures
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, Optional
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from qdrant_manager import QdrantManager
from indexer import Indexer


# =============================================================================
# Task State Management (Thread-Safe)
# =============================================================================

@dataclass
class TaskState:
    """Immutable task state snapshot."""
    status: str = "pending"
    progress: int = 0
    message: str = ""
    file_name: str = ""
    text_vectors: int = 0
    image_vectors: int = 0
    error: str = ""
    created_at: str = ""
    updated_at: str = ""
    from_cache: bool = False
    
    def to_dict(self) -> dict:
        return asdict(self)


class TaskRegistry:
    """
    Thread-safe task state management with Supabase persistence.
    
    All read/write operations are protected by a lock to prevent
    race conditions when multiple indexing threads update status.
    Task state is synced to Supabase for persistence.
    """
    
    def __init__(self):
        self._tasks: Dict[str, TaskState] = {}
        self._lock = threading.RLock()  # Reentrant lock for nested calls
        
        # Load tasks from Supabase on startup
        self._load_from_supabase()
    
    def _load_from_supabase(self) -> None:
        """Load tasks from Supabase on startup."""
        try:
            from supabase_client import get_supabase
            client = get_supabase()
            
            if client:
                result = client.table("indexing_tasks").select("*").execute()
                if result.data:
                    for task_data in result.data:
                        # Map database fields to TaskState
                        state = TaskState(
                            status=task_data.get("status", "pending"),
                            progress=task_data.get("progress", 0),
                            message=task_data.get("message", ""),
                            file_name=task_data.get("file_name", ""),
                            text_vectors=task_data.get("text_vectors", 0),
                            image_vectors=task_data.get("image_vectors", 0),
                            error=task_data.get("error", ""),
                            created_at=task_data.get("created_at", ""),
                            updated_at=task_data.get("updated_at", ""),
                            from_cache=task_data.get("from_cache", False)
                        )
                        self._tasks[task_data["id"]] = state
                    print(f"[DB] Loaded {len(result.data)} tasks from Supabase")
        except Exception as e:
            print(f"[WARN] Failed to load tasks from Supabase: {e}")
    
    def _sync_to_supabase(self, task_id: str, state: TaskState) -> None:
        """Sync task state to Supabase if available."""
        try:
            # Import here to avoid circular imports
            from supabase_client import get_supabase
            client = get_supabase()
            
            if client:
                data = {
                    "id": task_id,
                    "status": state.status,
                    "file_name": state.file_name,
                    "message": state.message,
                    "progress": state.progress,
                    "text_vectors": state.text_vectors,
                    "image_vectors": state.image_vectors,
                    "error": state.error,
                    "updated_at": state.updated_at
                }
                # Use upsert to handle both create and update
                client.table("indexing_tasks").upsert(data).execute()
        except Exception as e:
            # Fail silently to not disrupt the main flow (usually just config missing)
            pass

    def create(self, task_id: str, file_name: str) -> TaskState:
        """Create a new task entry."""
        with self._lock:
            now = datetime.now().isoformat()
            state = TaskState(
                status="started",
                progress=0,
                file_name=file_name,
                message="Upload complete. Starting indexing...",
                created_at=now,
                updated_at=now
            )
            self._tasks[task_id] = state
            
            # Sync to Supabase
            self._sync_to_supabase(task_id, state)
            
            return state
    
    def update(self, task_id: str, **kwargs) -> Optional[TaskState]:
        """
        Update task state atomically.
        
        Args:
            task_id: The task to update
            **kwargs: Fields to update (status, progress, message, etc.)
        
        Returns:
            Updated TaskState or None if task not found
        """
        with self._lock:
            if task_id not in self._tasks:
                return None
            
            current = self._tasks[task_id]
            # Create new state with updates
            updated_dict = current.to_dict()
            updated_dict.update(kwargs)
            updated_dict["updated_at"] = datetime.now().isoformat()
            
            self._tasks[task_id] = TaskState(**updated_dict)
            
            # Sync to Supabase
            self._sync_to_supabase(task_id, self._tasks[task_id])
            
            return self._tasks[task_id]
    
    def get(self, task_id: str) -> Optional[TaskState]:
        """Get task state (thread-safe read)."""
        with self._lock:
            state = self._tasks.get(task_id)
            return state  # Returns copy of dataclass
    
    def exists(self, task_id: str) -> bool:
        """Check if task exists."""
        with self._lock:
            return task_id in self._tasks
    
    def list_all(self) -> Dict[str, TaskState]:
        """Get all tasks (for debugging/admin)."""
        with self._lock:
            return dict(self._tasks)
    
    def cleanup_old(self, max_age_hours: int = 24) -> int:
        """Remove tasks older than max_age_hours."""
        from datetime import timedelta
        
        with self._lock:
            cutoff = datetime.now() - timedelta(hours=max_age_hours)
            to_remove = []
            
            for task_id, state in self._tasks.items():
                try:
                    created = datetime.fromisoformat(state.created_at)
                    if created < cutoff:
                        to_remove.append(task_id)
                except:
                    pass
            
            for task_id in to_remove:
                del self._tasks[task_id]
                # Also delete from Supabase
                try:
                    from supabase_client import get_supabase
                    client = get_supabase()
                    if client:
                        client.table("indexing_tasks").delete().eq("id", task_id).execute()
                except:
                    pass
            
            return len(to_remove)


# =============================================================================
# Singleton Indexer (Avoids Repeated Model Loading)
# =============================================================================

class SharedResources:
    """
    Singleton pattern for expensive shared resources.
    
    Ensures that:
    - Indexer is only initialized once (models loaded once)
    - QdrantManager connection is reused
    - Thread-safe initialization via double-checked locking
    - Failed initialization can be retried
    """
    
    _lock = threading.Lock()
    _indexer: Optional[Indexer] = None
    _qdrant_manager: Optional[QdrantManager] = None
    _initialized = False
    _init_error: Optional[Exception] = None
    
    @classmethod
    def get_indexer(cls) -> Indexer:
        """
        Get or create the shared Indexer instance.
        
        Uses double-checked locking for thread-safe lazy initialization.
        Retries initialization if previous attempt failed.
        """
        if cls._initialized:
            return cls._indexer
        
        with cls._lock:
            # Check again inside lock
            if cls._initialized:
                return cls._indexer
            
            # Reset any previous error state to allow retry
            cls._init_error = None
            
            try:
                print("[SYNC] Initializing shared Indexer (loading models)...")
                cls._qdrant_manager = QdrantManager(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)
                cls._indexer = Indexer(
                    qdrant_manager=cls._qdrant_manager,
                    llama_parse_api_key=config.LLAMA_PARSE_API_KEY
                )
                cls._initialized = True
                print("[OK] Shared Indexer ready")
                return cls._indexer
            except Exception as e:
                print(f"[ERROR] Failed to initialize Indexer: {e}")
                cls._init_error = e
                # Clean up partial state
                cls._indexer = None
                cls._qdrant_manager = None
                cls._initialized = False
                raise  # Re-raise so caller knows it failed
    
    @classmethod
    def get_qdrant_manager(cls) -> QdrantManager:
        """Get the shared QdrantManager instance."""
        if cls._qdrant_manager is None:
            # Force initialization via get_indexer
            cls.get_indexer()
        return cls._qdrant_manager
    
    @classmethod
    def is_initialized(cls) -> bool:
        """Check if resources are initialized."""
        return cls._initialized
    
    @classmethod
    def get_last_error(cls) -> Optional[Exception]:
        """Get the last initialization error, if any."""
        return cls._init_error
    
    @classmethod
    def reset(cls) -> None:
        """Reset for testing or to retry after failure."""
        with cls._lock:
            cls._indexer = None
            cls._qdrant_manager = None
            cls._initialized = False
            cls._init_error = None


# =============================================================================
# Dedicated Thread Pool for Indexing
# =============================================================================

# Create a dedicated thread pool that won't compete with uvicorn workers
# max_workers=2 limits concurrent indexing to prevent memory exhaustion
INDEXING_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=2,
    thread_name_prefix="indexer_"
)


# =============================================================================
# Global Instances
# =============================================================================

# Task registry with Supabase persistence (no local file)
TASK_REGISTRY = TaskRegistry()


# =============================================================================
# Query Engine (Lazy-loaded Singleton)
# =============================================================================

_query_engine = None
_query_engine_lock = threading.Lock()


async def get_query_engine():
    """
    Lazy load the query engine with thread-safe initialization.
    
    Uses double-checked locking to prevent race conditions when
    multiple requests try to initialize the engine simultaneously.
    """
    global _query_engine
    import asyncio
    
    if _query_engine is None:
        with _query_engine_lock:
            if _query_engine is None:
                # Import here to avoid circular imports and delays at startup
                from retriever import create_query_engine
                
                # Run initialization in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                _query_engine = await loop.run_in_executor(None, create_query_engine)
    
    return _query_engine


def shutdown():
    """Cleanup function to call on app shutdown."""
    print("[CLEAN] Shutting down indexing executor...")
    INDEXING_EXECUTOR.shutdown(wait=True, cancel_futures=False)
    print("[BYE] Executor shutdown complete")
