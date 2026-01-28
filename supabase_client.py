# supabase_client.py - Supabase Integration for ModalMuse
"""
Handles conversations, task state, and caching via Supabase.
"""

import os
from typing import List, Dict, Optional, Any
from datetime import datetime
import json

# Supabase Python client
try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    print("[WARN] supabase-py not installed. Run: pip install supabase")

import config


# ═══════════════════════════════════════════════════════════════════
# SUPABASE CLIENT
# ═══════════════════════════════════════════════════════════════════

_supabase_client: Optional[Any] = None

def get_supabase() -> Optional[Any]:
    """Get or create Supabase client."""
    global _supabase_client
    
    if not SUPABASE_AVAILABLE:
        return None
    
    if _supabase_client is None:
        url = os.getenv("SUPABASE_URL", "")
        key = os.getenv("SUPABASE_SERVICE_KEY", os.getenv("SUPABASE_ANON_KEY", ""))
        
        if url and key:
            _supabase_client = create_client(url, key)
            print("[OK] Supabase client initialized")
        else:
            print("[WARN] SUPABASE_URL or SUPABASE_SERVICE_KEY not set")
    
    return _supabase_client


# ═══════════════════════════════════════════════════════════════════
# CONVERSATIONS
# ═══════════════════════════════════════════════════════════════════

async def create_conversation(title: Optional[str] = None) -> Optional[Dict]:
    """Create a new conversation."""
    client = get_supabase()
    if not client:
        return None
    
    data = {
        "title": title or f"Chat {datetime.now().strftime('%b %d, %H:%M')}",
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat()
    }
    
    result = client.table("conversations").insert(data).execute()
    return result.data[0] if result.data else None


async def get_conversations(limit: int = 50) -> List[Dict]:
    """Get all conversations ordered by most recent."""
    client = get_supabase()
    if not client:
        return []
    
    result = client.table("conversations").select("*").order(
        "updated_at", desc=True
    ).limit(limit).execute()
    
    return result.data or []


async def get_conversation_messages(conversation_id: str) -> List[Dict]:
    """Get all messages for a conversation."""
    client = get_supabase()
    if not client:
        return []
    
    result = client.table("messages").select("*").eq(
        "conversation_id", conversation_id
    ).order("created_at").execute()
    
    # Parse sources from JSON string back to list
    messages = []
    for msg in (result.data or []):
        if msg.get("sources") and isinstance(msg["sources"], str):
            try:
                parsed = json.loads(msg["sources"])
                msg["sources"] = parsed
                # Debug: log first source score to verify
                if parsed and len(parsed) > 0:
                    print(f"[DEBUG] First source score from DB: {parsed[0].get('score')} (type: {type(parsed[0].get('score')).__name__})")
            except json.JSONDecodeError:
                msg["sources"] = None
        messages.append(msg)
    
    return messages


async def add_message(
    conversation_id: str,
    role: str,
    content: str,
    sources: Optional[List[Dict]] = None
) -> Optional[Dict]:
    """Add a message to a conversation."""
    client = get_supabase()
    if not client:
        return None
    
    data = {
        "conversation_id": conversation_id,
        "role": role,
        "content": content,
        "sources": json.dumps(sources) if sources else None,
        "created_at": datetime.now().isoformat()
    }
    
    result = client.table("messages").insert(data).execute()
    
    # Update conversation's updated_at
    client.table("conversations").update({
        "updated_at": datetime.now().isoformat()
    }).eq("id", conversation_id).execute()
    
    return result.data[0] if result.data else None


async def delete_conversation(conversation_id: str) -> bool:
    """Delete a conversation and its messages."""
    client = get_supabase()
    if not client:
        return False
    
    # Messages are deleted via CASCADE
    client.table("conversations").delete().eq("id", conversation_id).execute()
    return True


# ═══════════════════════════════════════════════════════════════════
# QUERY CACHE
# ═══════════════════════════════════════════════════════════════════

async def get_cached_embedding(query: str) -> Optional[List[float]]:
    """Get cached query embedding."""
    client = get_supabase()
    if not client:
        return None
    
    import hashlib
    query_hash = hashlib.md5(query.encode()).hexdigest()
    
    result = client.table("query_cache").select("embedding").eq(
        "query_hash", query_hash
    ).execute()
    
    if result.data and result.data[0].get("embedding"):
        return result.data[0]["embedding"]
    
    return None


async def cache_embedding(query: str, embedding: List[float], ttl_hours: int = 1) -> None:
    """Cache a query embedding."""
    client = get_supabase()
    if not client:
        return
    
    import hashlib
    from datetime import timedelta
    
    query_hash = hashlib.md5(query.encode()).hexdigest()
    expires_at = (datetime.now() + timedelta(hours=ttl_hours)).isoformat()
    
    client.table("query_cache").upsert({
        "query_hash": query_hash,
        "query_text": query,  # Required by schema
        "embedding": embedding,
        "expires_at": expires_at
    }).execute()


# ═══════════════════════════════════════════════════════════════════
# TASK STATE
# ═══════════════════════════════════════════════════════════════════

async def save_task_state(task_id: str, state: Dict) -> None:
    """Save indexing task state to Supabase."""
    client = get_supabase()
    if not client:
        return
    
    data = {
        "id": task_id,
        **state,
        "updated_at": datetime.now().isoformat()
    }
    
    client.table("indexing_tasks").upsert(data).execute()


async def get_task_state(task_id: str) -> Optional[Dict]:
    """Get indexing task state."""
    client = get_supabase()
    if not client:
        return None
    
    result = client.table("indexing_tasks").select("*").eq("id", task_id).execute()
    return result.data[0] if result.data else None


# ═══════════════════════════════════════════════════════════════════
# PARSE CACHE
# ═══════════════════════════════════════════════════════════════════

def get_parse_cache(file_hash: str) -> Optional[Dict]:
    """
    Get cached LlamaParse result from Supabase.
    
    Returns:
        Dict with keys: file_name, job_id, parsed_json, images_data
        or None if not cached
    """
    client = get_supabase()
    if not client:
        return None
    
    try:
        result = client.table("parse_cache").select("*").eq(
            "file_hash", file_hash
        ).execute()
        
        if result.data and len(result.data) > 0:
            cached = result.data[0]
            
            # Handle null/None images_data
            images_data = cached.get("images_data")
            if images_data is None:
                images_data = []
            elif isinstance(images_data, str):
                # If stored as JSON string, parse it
                import json
                try:
                    images_data = json.loads(images_data)
                except:
                    images_data = []
            
            print(f"[CACHE] Read from Supabase: {cached.get('file_name')}, images_data: {len(images_data)} items")
            
            return {
                "file_name": cached.get("file_name"),
                "job_id": cached.get("job_id"),
                "parsed_json": cached.get("parsed_json"),
                "images_data": images_data
            }
    except Exception as e:
        print(f"[WARN] Failed to get parse cache: {e}")
    
    return None


def save_parse_cache(
    file_hash: str,
    file_name: str,
    parsed_json: Any,
    images_data: List[Dict],
    job_id: Optional[str] = None
) -> bool:
    """
    Save LlamaParse result to Supabase cache.
    
    Args:
        file_hash: SHA256 hash of the file
        file_name: Original file name
        parsed_json: LlamaParse JSON result
        images_data: List of image metadata dicts
        job_id: Optional LlamaParse job ID
    
    Returns:
        True if saved successfully
    """
    client = get_supabase()
    if not client:
        return False
    
    try:
        data = {
            "file_hash": file_hash,
            "file_name": file_name,
            "job_id": job_id,
            "parsed_json": parsed_json,
            "images_data": images_data,
            "updated_at": datetime.now().isoformat()
        }
        
        client.table("parse_cache").upsert(data).execute()
        print(f"[CACHE] Saved parse result to Supabase: {file_name}")
        return True
        
    except Exception as e:
        print(f"[WARN] Failed to save parse cache: {e}")
        return False


def delete_parse_cache(file_hash: str) -> bool:
    """Delete a parse cache entry."""
    client = get_supabase()
    if not client:
        return False
    
    try:
        client.table("parse_cache").delete().eq("file_hash", file_hash).execute()
        return True
    except Exception as e:
        print(f"[WARN] Failed to delete parse cache: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════
# IMAGE STORAGE
# ═══════════════════════════════════════════════════════════════════

IMAGES_BUCKET = "images"


def ensure_bucket_exists() -> bool:
    """Ensure the images bucket exists in Supabase storage."""
    client = get_supabase()
    if not client:
        return False
    
    try:
        # List buckets to check if 'images' exists
        buckets = client.storage.list_buckets()
        bucket_names = [b.name for b in buckets]
        
        if IMAGES_BUCKET not in bucket_names:
            # Create the bucket with public access
            client.storage.create_bucket(
                IMAGES_BUCKET,
                options={"public": True}
            )
            print(f"[OK] Created Supabase storage bucket: {IMAGES_BUCKET}")
        
        return True
    except Exception as e:
        print(f"[WARN] Failed to ensure bucket exists: {e}")
        return False


def upload_image(image_bytes: bytes, image_name: str, content_type: str = "image/png") -> Optional[str]:
    """
    Upload an image to Supabase storage.
    
    Args:
        image_bytes: Raw image bytes
        image_name: Name to use for the file in storage
        content_type: MIME type of the image
    
    Returns:
        Public URL of the uploaded image, or None if failed
    """
    client = get_supabase()
    if not client:
        return None
    
    try:
        # Ensure bucket exists
        ensure_bucket_exists()
        
        # Upload to storage
        response = client.storage.from_(IMAGES_BUCKET).upload(
            path=image_name,
            file=image_bytes,
            file_options={"content-type": content_type}
        )
        
        # Get public URL
        public_url = client.storage.from_(IMAGES_BUCKET).get_public_url(image_name)
        print(f"   [OK] Uploaded to Supabase: {image_name}")
        return public_url
        
    except Exception as e:
        # Check if file already exists (duplicate upload)
        if "Duplicate" in str(e) or "already exists" in str(e).lower():
            # Return existing URL
            public_url = client.storage.from_(IMAGES_BUCKET).get_public_url(image_name)
            return public_url
        
        print(f"   [WARN] Failed to upload {image_name}: {e}")
        return None


def upload_image_from_response(response_content: bytes, image_name: str) -> Optional[str]:
    """
    Upload image bytes (from HTTP response) to Supabase storage.
    
    Returns the public URL or None if failed.
    """
    # Detect content type from file extension
    ext = image_name.split(".")[-1].lower() if "." in image_name else "png"
    content_type_map = {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "gif": "image/gif",
        "webp": "image/webp",
        "svg": "image/svg+xml"
    }
    content_type = content_type_map.get(ext, "image/png")
    
    return upload_image(response_content, image_name, content_type)


def get_image_public_url(image_name: str) -> Optional[str]:
    """Get the public URL for an image in storage."""
    client = get_supabase()
    if not client:
        return None
    
    try:
        return client.storage.from_(IMAGES_BUCKET).get_public_url(image_name)
    except Exception as e:
        print(f"[WARN] Failed to get public URL for {image_name}: {e}")
        return None


def delete_image(image_name: str) -> bool:
    """Delete an image from storage."""
    client = get_supabase()
    if not client:
        return False
    
    try:
        client.storage.from_(IMAGES_BUCKET).remove([image_name])
        return True
    except Exception as e:
        print(f"[WARN] Failed to delete {image_name}: {e}")
        return False
