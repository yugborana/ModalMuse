# api/routes/query.py
"""Query Endpoints for Multi-Modal RAG

REST API fallback for the query pipeline.
The primary query interface is the WebSocket at /ws/query.
"""

import sys
from typing import Optional, List
from pathlib import Path
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import config
from api.shared_resources import get_query_engine, sanitize_error

router = APIRouter(prefix="/query", tags=["Query"])


class QueryRequest(BaseModel):
    """Request model for queries."""
    query: str = Field(..., description="The question to ask")
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


@router.post("/", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Query the multi-modal RAG system (REST fallback).
    
    For real-time streaming, use WebSocket at /ws/query instead.
    
    - **query**: The question to ask about the indexed documents
    - **include_sources**: Include source references in response
    
    Returns the generated answer along with relevant sources.
    """
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
        safe_tb = sanitize_error(tb_str)
        print(tb_str)
        raise HTTPException(
            status_code=500,
            detail=f"Query failed: {safe_tb}"
        )


@router.get("/health")
async def query_health():
    """Check if the query engine is ready."""
    from api.shared_resources import _query_engine
    
    return {
        "status": "ready" if _query_engine is not None else "not_initialized",
        "message": "Query engine will be initialized on first query" if _query_engine is None else "Query engine is ready"
    }


@router.post("/initialize")
async def initialize_engine():
    """
    Pre-initialize the query engine.
    
    Call this endpoint to warm up the models before sending queries.
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
