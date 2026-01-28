# api/routes/websocket.py - WebSocket endpoint for real-time query streaming
"""
WebSocket support for live status updates during RAG pipeline.
Provides detailed progress for embedding, search, reranking, and generation phases.
"""

import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import List, Set
import asyncio

router = APIRouter(tags=["websocket"])


# Active WebSocket connections
active_connections: Set[WebSocket] = set()


@router.websocket("/ws/query")
async def websocket_query(websocket: WebSocket):
    """
    WebSocket endpoint for streaming query responses with detailed live status updates.
    
    Client sends: {"query": "your question", "include_sources": true, "detailed": true}
    
    Server sends (detailed mode):
        {"type": "phase", "phase": "embedding", "status": "started/completed", "message": "..."}
        {"type": "chunks_found", "text_count": N, "image_count": M, "chunks": [...]}
        {"type": "generation", "chunk": "text"}
        {"type": "sources", "sources": [...]}
        {"type": "done", "total_duration_ms": 1234}
    
    Server sends (simple mode):
        {"type": "status", "message": "🔍 Searching..."}
        {"type": "chunk", "chunk": "text"}
        {"type": "sources", "sources": [...]}
        {"type": "done"}
    """
    await websocket.accept()
    active_connections.add(websocket)
    
    try:
        while True:
            # Receive query from client
            data = await websocket.receive_json()
            query = data.get("query", "")
            include_sources = data.get("include_sources", True)
            detailed = data.get("detailed", True)  # Use detailed mode by default
            conversation_id = data.get("conversation_id")  # Optional
            
            if not query:
                await websocket.send_json({"type": "error", "message": "No query provided"})
                continue
            
            # Get query engine
            from api.shared_resources import get_query_engine
            engine = await get_query_engine()
            
            if detailed:
                # Use detailed streaming with phase updates
                await stream_detailed(websocket, engine, query, include_sources, conversation_id)
            else:
                # Use simple streaming (legacy)
                await stream_simple(websocket, engine, query, include_sources)
    
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


async def stream_simple(websocket: WebSocket, engine, query: str, include_sources: bool):
    """Simple streaming (legacy mode)."""
    try:
        sources = []
        
        async for event_type, event_data, nodes in engine.astream_query(query):
            if event_type == "status":
                await websocket.send_json({
                    "type": "status",
                    "message": event_data
                })
            
            elif event_type == "chunk":
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
                
                await websocket.send_json({
                    "type": "chunk",
                    "chunk": event_data
                })
        
        # Send sources
        if sources and include_sources:
            await websocket.send_json({
                "type": "sources",
                "sources": sources
            })
        
        # Done
        await websocket.send_json({"type": "done"})
        
    except Exception as e:
        await websocket.send_json({"type": "error", "message": str(e)})


# Broadcast helper (for future use)
async def broadcast(message: dict):
    """Broadcast a message to all connected clients."""
    for connection in active_connections.copy():
        try:
            await connection.send_json(message)
        except:
            active_connections.discard(connection)
