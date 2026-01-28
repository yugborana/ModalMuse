# api/routes/indexing.py
"""
Document Indexing Endpoints

Provides async, non-blocking document indexing with:
- Thread-safe task state management
- Dedicated thread pool (doesn't block API workers)
- Singleton indexer (models loaded once)
- File-based task persistence
"""

import os
import asyncio
import sys
from pathlib import Path
from typing import Optional
from functools import partial
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
import aiofiles
import uuid

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import config
from api.shared_resources import (
    SharedResources,
    TASK_REGISTRY,
    INDEXING_EXECUTOR,
    TaskState
)


def sanitize_error(error: Exception) -> str:
    """Sanitize error message to remove non-ASCII characters that cause encoding issues."""
    try:
        msg = str(error)
        return msg.encode('ascii', 'replace').decode('ascii')
    except:
        return "An error occurred (message contained invalid characters)"

router = APIRouter(prefix="/index", tags=["Indexing"])


# =============================================================================
# Response Models
# =============================================================================

class IndexingStatus(BaseModel):
    """Response model for indexing status."""
    status: str
    message: str
    file_name: Optional[str] = None
    text_vectors: Optional[int] = None
    image_vectors: Optional[int] = None
    error: Optional[str] = None
    progress: Optional[int] = None
    from_cache: Optional[bool] = None


class CollectionStats(BaseModel):
    """Response model for collection statistics."""
    text_collection: dict
    image_collection: dict


# =============================================================================
# Background Indexing Function (Runs in ThreadPoolExecutor)
# =============================================================================

def index_document_sync(file_path: str, task_id: str) -> None:
    """
    Synchronous indexing function that runs in a dedicated thread pool.
    
    This function:
    - Uses the singleton Indexer (no repeated model loading)
    - Updates task state atomically via TaskRegistry
    - Handles errors gracefully
    
    Args:
        file_path: Path to the PDF file
        task_id: Unique task identifier
    """
    try:
        # Update status: initializing
        TASK_REGISTRY.update(
            task_id,
            status="processing",
            progress=10,
            message="Loading indexer..."
        )
        
        # Get singleton indexer (models already loaded)
        indexer = SharedResources.get_indexer()
        
        # Update status: parsing
        TASK_REGISTRY.update(
            task_id,
            progress=20,
            message="Parsing document with LlamaParse..."
        )
        
        # Perform indexing (this is the long-running operation)
        result = indexer.index_document(file_path)
        
        # Update status: completed
        TASK_REGISTRY.update(
            task_id,
            status="completed",
            progress=100,
            message=f"Successfully indexed: {Path(file_path).name}",
            text_vectors=result.get("text_count", 0),
            image_vectors=result.get("image_count", 0),
            from_cache=result.get("from_cache", False)
        )
        
        print(f"[OK] Task {task_id[:8]}... completed: {result}")
        
    except Exception as e:
        import traceback
        error_msg = sanitize_error(e)
        print(f"[ERROR] Task {task_id[:8]}... failed: {error_msg}")
        traceback.print_exc()
        
        TASK_REGISTRY.update(
            task_id,
            status="failed",
            progress=0,
            error=error_msg,
            message="Indexing failed"
        )


# =============================================================================
# API Endpoints
# =============================================================================

@router.post("/upload", response_model=IndexingStatus)
async def upload_and_index(
    file: UploadFile = File(...),
):
    """
    Upload a PDF file and start indexing in the background.
    
    The indexing runs in a dedicated thread pool, so:
    - Response returns immediately
    - API remains responsive during indexing
    - Multiple files can be indexed concurrently (up to 2)
    
    Use GET /api/index/status/{task_id} to check progress.
    """
    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported"
        )
    
    # Validate API key
    if not config.LLAMA_PARSE_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="LlamaParse API key not configured"
        )
    
    # Create data directory if it doesn't exist
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save the uploaded file
    file_path = config.DATA_DIR / file.filename
    
    async with aiofiles.open(file_path, 'wb') as out_file:
        content = await file.read()
        await out_file.write(content)
    
    # Generate task ID
    task_id = str(uuid.uuid4())
    
    # Create task in registry (thread-safe, persisted)
    TASK_REGISTRY.create(task_id, file.filename)
    
    # Submit to dedicated thread pool (non-blocking!)
    # This is the key fix - we don't use BackgroundTasks
    loop = asyncio.get_event_loop()
    loop.run_in_executor(
        INDEXING_EXECUTOR,
        partial(index_document_sync, str(file_path), task_id)
    )
    
    return IndexingStatus(
        status="started",
        message=f"Indexing started for {file.filename}. Task ID: {task_id}",
        file_name=file.filename,
        progress=0
    )


@router.get("/status/{task_id}", response_model=IndexingStatus)
async def get_indexing_status(task_id: str):
    """
    Get the status of an indexing task.
    
    Returns current progress, status, and results when complete.
    """
    state = TASK_REGISTRY.get(task_id)
    
    if state is None:
        raise HTTPException(
            status_code=404,
            detail="Task not found"
        )
    
    return IndexingStatus(
        status=state.status,
        message=state.message,
        file_name=state.file_name,
        text_vectors=state.text_vectors if state.text_vectors else None,
        image_vectors=state.image_vectors if state.image_vectors else None,
        error=state.error if state.error else None,
        progress=state.progress,
        from_cache=state.from_cache if state.from_cache else None
    )


@router.get("/tasks")
async def list_tasks():
    """
    List all indexing tasks (for debugging/admin).
    """
    tasks = TASK_REGISTRY.list_all()
    return {
        "tasks": {tid: state.to_dict() for tid, state in tasks.items()},
        "count": len(tasks)
    }


@router.post("/tasks/cleanup")
async def cleanup_old_tasks(max_age_hours: int = 24):
    """
    Remove tasks older than max_age_hours.
    """
    removed = TASK_REGISTRY.cleanup_old(max_age_hours)
    return {"removed": removed, "message": f"Removed {removed} old tasks"}


@router.get("/collections", response_model=CollectionStats)
async def get_collection_stats():
    """
    Get statistics for all indexed collections.
    """
    import httpx
    
    async with httpx.AsyncClient() as client:
        try:
            # Get text collection stats
            text_response = await client.get(
                f"{config.QDRANT_URL}/collections/{config.TEXT_COLLECTION_NAME}"
            )
            text_data = text_response.json() if text_response.status_code == 200 else {"error": "Not found"}
            
            # Get image collection stats
            image_response = await client.get(
                f"{config.QDRANT_URL}/collections/{config.IMAGE_COLLECTION_NAME}"
            )
            image_data = image_response.json() if image_response.status_code == 200 else {"error": "Not found"}
            
            return CollectionStats(
                text_collection={
                    "name": config.TEXT_COLLECTION_NAME,
                    "points_count": text_data.get("result", {}).get("points_count", 0),
                    "status": text_data.get("result", {}).get("status", "unknown"),
                } if "error" not in text_data else {"error": text_data["error"]},
                image_collection={
                    "name": config.IMAGE_COLLECTION_NAME,
                    "points_count": image_data.get("result", {}).get("points_count", 0),
                    "status": image_data.get("result", {}).get("status", "unknown"),
                } if "error" not in image_data else {"error": image_data["error"]},
            )
            
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to connect to Qdrant: {sanitize_error(e)}"
            )


@router.delete("/collections/{collection_name}")
async def delete_collection(collection_name: str):
    """
    Delete a vector collection.
    """
    import httpx
    
    # Validate collection name
    valid_names = [config.TEXT_COLLECTION_NAME, config.IMAGE_COLLECTION_NAME]
    if collection_name not in valid_names:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid collection name. Must be one of: {valid_names}"
        )
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.delete(
                f"{config.QDRANT_URL}/collections/{collection_name}"
            )
            
            if response.status_code == 200:
                return {"status": "deleted", "collection": collection_name}
            else:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Failed to delete collection: {response.text}"
                )
                
        except httpx.RequestError as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to connect to Qdrant: {sanitize_error(e)}"
            )


@router.get("/files")
async def list_indexed_files():
    """
    List all PDF files in the data directory.
    """
    files = []
    
    if config.DATA_DIR.exists():
        for file_path in config.DATA_DIR.glob("*.pdf"):
            stat = file_path.stat()
            files.append({
                "name": file_path.name,
                "size_bytes": stat.st_size,
                "modified": stat.st_mtime,
            })
    
    return {"files": files, "count": len(files)}


@router.post("/initialize")
async def initialize_indexer():
    """
    Pre-initialize the indexer and load all models.
    
    Call this endpoint to warm up before sending indexing requests.
    This can take 10-30 seconds depending on hardware.
    """
    if SharedResources.is_initialized():
        return {
            "status": "ready",
            "message": "Indexer already initialized"
        }
    
    try:
        # Initialize in thread pool to not block
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, SharedResources.get_indexer)
        
        return {
            "status": "ready",
            "message": "Indexer initialized successfully"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to initialize indexer: {sanitize_error(e)}"
        )


@router.get("/health")
async def indexer_health():
    """
    Check indexer health and readiness.
    """
    return {
        "indexer_initialized": SharedResources.is_initialized(),
        "executor_threads": INDEXING_EXECUTOR._max_workers,
        "pending_tasks": len([t for t in TASK_REGISTRY.list_all().values() 
                              if t.status in ("started", "processing")])
    }