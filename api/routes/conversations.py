# api/routes/conversations.py - REST API for conversation history
"""
Endpoints for managing conversations and messages.
Includes in-memory fallback when Supabase is not configured.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import json
import uuid
from datetime import datetime

router = APIRouter(prefix="/api/conversations", tags=["conversations"])


# ═══════════════════════════════════════════════════════════════════
# IN-MEMORY FALLBACK (when Supabase is not configured)
# ═══════════════════════════════════════════════════════════════════

_memory_conversations: Dict[str, Dict] = {}
_memory_messages: Dict[str, List[Dict]] = {}


def _is_supabase_available() -> bool:
    """Check if Supabase is configured and available."""
    try:
        from supabase_client import get_supabase
        client = get_supabase()
        return client is not None
    except:
        return False


# ═══════════════════════════════════════════════════════════════════
# MODELS
# ═══════════════════════════════════════════════════════════════════

class CreateConversationRequest(BaseModel):
    title: Optional[str] = None


class AddMessageRequest(BaseModel):
    role: str  # 'user' or 'assistant'
    content: str
    sources: Optional[List[Dict[str, Any]]] = None


class ConversationResponse(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str


class MessageResponse(BaseModel):
    id: str
    conversation_id: str
    role: str
    content: str
    sources: Optional[List[Dict[str, Any]]] = None
    created_at: str


# ═══════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════

@router.get("")
async def list_conversations(limit: int = 50) -> List[Dict]:
    """Get all conversations, most recent first."""
    if _is_supabase_available():
        try:
            from supabase_client import get_conversations
            return await get_conversations(limit)
        except Exception as e:
            print(f"[WARN] Supabase error: {e}, using memory fallback")
    
    # In-memory fallback
    convs = list(_memory_conversations.values())
    convs.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
    return convs[:limit]


@router.post("")
async def create_conversation(request: CreateConversationRequest) -> Dict:
    """Create a new conversation."""
    now = datetime.now().isoformat()
    
    if _is_supabase_available():
        try:
            from supabase_client import create_conversation as sb_create
            result = await sb_create(request.title)
            if result:
                return result
        except Exception as e:
            print(f"[WARN] Supabase error: {e}, using memory fallback")
    
    # In-memory fallback
    conv_id = str(uuid.uuid4())
    conv = {
        "id": conv_id,
        "title": request.title or f"Chat {datetime.now().strftime('%b %d, %H:%M')}",
        "created_at": now,
        "updated_at": now
    }
    _memory_conversations[conv_id] = conv
    _memory_messages[conv_id] = []
    return conv


@router.get("/{conversation_id}")
async def get_conversation(conversation_id: str) -> Dict:
    """Get a conversation with its messages."""
    if _is_supabase_available():
        try:
            from supabase_client import get_conversation_messages, get_supabase
            
            client = get_supabase()
            conv_result = client.table("conversations").select("*").eq("id", conversation_id).execute()
            
            if not conv_result.data:
                raise HTTPException(status_code=404, detail="Conversation not found")
            
            messages = await get_conversation_messages(conversation_id)
            
            return {
                **conv_result.data[0],
                "messages": messages
            }
        except HTTPException:
            raise
        except Exception as e:
            print(f"[WARN] Supabase error: {e}, using memory fallback")
    
    # In-memory fallback
    if conversation_id not in _memory_conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    return {
        **_memory_conversations[conversation_id],
        "messages": _memory_messages.get(conversation_id, [])
    }


@router.post("/{conversation_id}/messages")
async def add_message(conversation_id: str, request: AddMessageRequest) -> Dict:
    """Add a message to a conversation."""
    now = datetime.now().isoformat()
    
    if _is_supabase_available():
        try:
            from supabase_client import add_message as sb_add, get_supabase
            result = await sb_add(
                conversation_id=conversation_id,
                role=request.role,
                content=request.content,
                sources=request.sources
            )
            if result:
                # Update conversation timestamp
                client = get_supabase()
                client.table("conversations").update({"updated_at": now}).eq("id", conversation_id).execute()
                return result
        except Exception as e:
            print(f"[WARN] Supabase error: {e}, using memory fallback")
    
    # In-memory fallback
    if conversation_id not in _memory_conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    msg = {
        "id": str(uuid.uuid4()),
        "conversation_id": conversation_id,
        "role": request.role,
        "content": request.content,
        "sources": request.sources,
        "created_at": now
    }
    
    if conversation_id not in _memory_messages:
        _memory_messages[conversation_id] = []
    _memory_messages[conversation_id].append(msg)
    
    # Update conversation timestamp
    _memory_conversations[conversation_id]["updated_at"] = now
    
    return msg


@router.delete("/{conversation_id}")
async def delete_conversation(conversation_id: str) -> Dict:
    """Delete a conversation and all its messages."""
    if _is_supabase_available():
        try:
            from supabase_client import delete_conversation as sb_delete
            await sb_delete(conversation_id)
            return {"success": True}
        except Exception as e:
            print(f"[WARN] Supabase error: {e}, using memory fallback")
    
    # In-memory fallback
    if conversation_id in _memory_conversations:
        del _memory_conversations[conversation_id]
    if conversation_id in _memory_messages:
        del _memory_messages[conversation_id]
    
    return {"success": True}
