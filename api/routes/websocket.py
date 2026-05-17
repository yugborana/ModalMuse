# api/routes/websocket.py - WebSocket endpoint for real-time query streaming
"""
WebSocket support for live status updates during RAG pipeline.
Provides detailed progress for embedding, search, reranking, and generation phases.
"""

import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Set

router = APIRouter(tags=["websocket"])


# Active WebSocket connections
active_connections: Set[WebSocket] = set()


@router.websocket("/ws/query")
async def websocket_query(websocket: WebSocket):
    """
    WebSocket endpoint for streaming query responses with detailed live status updates.
    
    Client sends: {"query": "your question", "include_sources": true}
    
    Server sends:
        {"type": "phase", "phase": "embedding", "status": "started/completed", "message": "..."}
        {"type": "chunks_found", "text_count": N, "image_count": M, "chunks": [...]}
        {"type": "generation", "chunk": "text"}
        {"type": "sources", "sources": [...]}
        {"type": "done", "total_duration_ms": 1234}
    """
    await websocket.accept()
    active_connections.add(websocket)
    
    try:
        while True:
            # Receive query from client
            data = await websocket.receive_json()
            query = data.get("query", "")
            include_sources = data.get("include_sources", True)
            conversation_id = data.get("conversation_id")  # Optional
            
            if not query:
                await websocket.send_json({"type": "error", "message": "No query provided"})
                continue
            
            # Get query engine (shared singleton from shared_resources.py)
            from api.shared_resources import get_query_engine
            engine = await get_query_engine()
            
            await stream_detailed(websocket, engine, query, include_sources, conversation_id)
    
    except WebSocketDisconnect:
        active_connections.discard(websocket)
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except:
            pass
        active_connections.discard(websocket)


async def stream_detailed(websocket: WebSocket, engine, query: str, include_sources: bool, conversation_id: str = None):
    """Stream with detailed phase updates."""
    try:
        full_response = ""
        sources = []
        
        async for event in engine.astream_query_detailed(query):
            event_type = event.get("type")
            
            if event_type == "phase":
                await websocket.send_json(event)
            
            elif event_type == "chunks_found":
                await websocket.send_json(event)
            
            elif event_type == "generation":
                full_response += event.get("chunk", "")
                await websocket.send_json(event)
            
            elif event_type == "sources":
                sources = event.get("sources", [])
                if include_sources:
                    await websocket.send_json(event)
            
            elif event_type == "done":
                await websocket.send_json(event)
            
            elif event_type == "error":
                await websocket.send_json(event)
        
        # Save to conversation if ID provided
        if conversation_id and full_response:
            try:
                from supabase_client import add_message
                await add_message(
                    conversation_id=conversation_id,
                    role="assistant",
                    content=full_response,
                    sources=sources if include_sources else None
                )
            except Exception as e:
                print(f"[WARN] Failed to save message: {e}")
                
    except Exception as e:
        await websocket.send_json({"type": "error", "message": str(e)})
