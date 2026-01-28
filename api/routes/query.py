# api/routes/query.py
"""Query Endpoints for Multi-Modal RAG"""

import asyncio
import sys
from typing import Optional, List, AsyncGenerator
from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import json

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import config


def sanitize_error(error: Exception) -> str:
    """Sanitize error message to remove non-ASCII characters that cause encoding issues."""
    try:
        msg = str(error)
        # Replace non-ASCII characters with their unicode escape or a placeholder
        return msg.encode('ascii', 'replace').decode('ascii')
    except:
        return "An error occurred (message contained invalid characters)"

router = APIRouter(prefix="/query", tags=["Query"])


# Global query engine (lazy loaded)
_query_engine = None


class QueryRequest(BaseModel):
    """Request model for queries."""
    query: str = Field(..., description="The question to ask")
    stream: bool = Field(default=False, description="Enable streaming response")
    include_sources: bool = Field(default=True, description="Include source references")


class SourceNode(BaseModel):
    """Model for a source reference."""
    content: str
    score: float
    type: str  # "text" or "image"
    metadata: dict


class QueryResponse(BaseModel):
    """Response model for queries."""
    answer: str
    sources: Optional[List[SourceNode]] = None
    query: str


# Thread-safe query engine initialization
import threading
_query_engine_lock = threading.Lock()


async def get_query_engine():
    """
    Lazy load the query engine with thread-safe initialization.
    
    Uses double-checked locking to prevent race conditions when
    multiple requests try to initialize the engine simultaneously.
    """
    global _query_engine
    
    if _query_engine is None:
        # Use lock to prevent race condition during initialization
        with _query_engine_lock:
            if _query_engine is None:
                # Import here to avoid circular imports and delays at startup
                from retriever import create_query_engine
                
                # Run initialization in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                _query_engine = await loop.run_in_executor(None, create_query_engine)
    
    return _query_engine


@router.post("/", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Query the multi-modal RAG system.
    
    - **query**: The question to ask about the indexed documents
    - **stream**: If true, response will be streamed (use /query/stream instead)
    - **include_sources**: Include source references in response
    
    Returns the generated answer along with relevant sources.
    """
    if request.stream:
        raise HTTPException(
            status_code=400,
            detail="For streaming responses, use POST /query/stream"
        )
    
    try:
        engine = await get_query_engine()
        
        # Execute query (async)
        response = await engine.acustom_query(request.query)
        
        # Format sources
        sources = None
        if request.include_sources and response.source_nodes:
            sources = []
            for node in response.source_nodes:
                node_type = "image" if "image_path" in node.metadata else "text"
                sources.append(SourceNode(
                    content=node.get_content()[:500],  # Truncate for response size
                    score=node.score or 0.0,
                    type=node_type,
                    metadata={
                        k: v for k, v in node.metadata.items() 
                        if k not in ["image_path"]  # Exclude paths from response
                    }
                ))
        
        return QueryResponse(
            answer=str(response.response),
            sources=sources,
            query=request.query
        )
        
    except Exception as e:
        import traceback
        tb_str = traceback.format_exc()
        # Sanitize the traceback too
        safe_tb = sanitize_error(tb_str)
        print(tb_str) # Still try to print
        raise HTTPException(
            status_code=500,
            detail=f"Query failed: {safe_tb}"
        )


@router.post("/stream")
async def query_stream(request: QueryRequest):
    """
    Stream a query response using Server-Sent Events.
    
    - **query**: The question to ask about the indexed documents
    
    Returns a streaming response with chunks of the generated answer.
    """
    async def generate() -> AsyncGenerator[str, None]:
        try:
            engine = await get_query_engine()
            
            # Stream the response
            full_response = ""
            sources = []
            
            async for event_type, data, nodes in engine.astream_query(request.query):
                if event_type == "status":
                    # Send status update
                    yield f"data: {json.dumps({'type': 'status', 'message': data})}\n\n"
                
                elif event_type == "chunk":
                    full_response += data
                    
                    # Store sources from first chunk
                    if not sources and nodes:
                        for node in nodes:
                            node_type = "image" if "image_path" in node.metadata else "text"
                            sources.append({
                                "content": node.get_content()[:500],
                                "score": node.score or 0.0,
                                "type": node_type,
                                "metadata": dict(node.metadata) if node.metadata else {}
                            })
                    
                    # Send chunk as SSE event
                    yield f"data: {json.dumps({'type': 'chunk', 'chunk': data})}\n\n"
            
            # Send sources first, then done signal
            if sources and request.include_sources:
                yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"
            
            yield f"data: [DONE]\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'error': sanitize_error(e)})}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@router.get("/health")
async def query_health():
    """
    Check if the query engine is ready.
    """
    global _query_engine
    
    return {
        "status": "ready" if _query_engine is not None else "not_initialized",
        "message": "Query engine will be initialized on first query" if _query_engine is None else "Query engine is ready"
    }


@router.post("/initialize")
async def initialize_engine():
    """
    Pre-initialize the query engine.
    
    Call this endpoint to warm up the models before sending queries.
    This can take several minutes depending on GPU availability.
    """
    try:
        engine = await get_query_engine()
        return {
            "status": "ready",
            "message": "Query engine initialized successfully"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to initialize query engine: {sanitize_error(e)}"
        )
