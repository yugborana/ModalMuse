# api/main.py
"""
ModalMuse API - FastAPI Application
Multi-Modal RAG with LlamaParse + llama.cpp
"""

import sys
import os
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import httpx

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from api.routes import indexing_router, query_router
from api.routes.conversations import router as conversations_router
from api.shared_resources import shutdown as shutdown_shared_resources


def sanitize_error(error: Exception) -> str:
    """Sanitize error message to remove non-ASCII characters that cause encoding issues."""
    try:
        msg = str(error)
        return msg.encode('ascii', 'replace').decode('ascii')
    except:
        return "An error occurred (message contained invalid characters)"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    """
    # Startup
    print("[*] ModalMuse API Starting...")
    
    # Validate configuration
    if not config.validate_config():
        print("[!] Configuration warnings found. Some features may not work.")
    
    yield
    
    # Shutdown
    print("[*] ModalMuse API Shutting down...")
    shutdown_shared_resources()  # Gracefully stop indexing threads


# Create FastAPI app
app = FastAPI(
    title="ModalMuse API",
    description="""
## Multi-Modal RAG API

ModalMuse is a multi-modal Retrieval-Augmented Generation system that can:

- **Index PDF documents** with text and image extraction using LlamaParse
- **Hybrid retrieval** using BGE (dense) + SPLADE (sparse) for text, CLIP for images
- **Generate answers** using local LLaVA model via llama.cpp

### Features
- [DOC] PDF document indexing with image extraction
- [SEARCH] Hybrid text search (dense + sparse vectors)
- 🖼️ Multi-modal image retrieval with CLIP
- 🤖 Local LLM inference with LLaVA
- ⚡ Async endpoints with streaming support
- 💬 Conversation history with Supabase
    """,
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for serving downloaded images (only if directory exists - legacy/local dev)
from fastapi.staticfiles import StaticFiles
images_dir = Path(config.BASE_DIR) / "downloaded_images"
if images_dir.exists():
    app.mount("/downloaded_images", StaticFiles(directory=str(images_dir)), name="images")

# Include routers
app.include_router(indexing_router, prefix="/api")
app.include_router(query_router, prefix="/api")
app.include_router(conversations_router)  # /api/conversations

# WebSocket router
from api.routes.websocket import router as websocket_router
app.include_router(websocket_router)


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "ModalMuse API",
        "version": "1.0.0",
        "description": "Multi-Modal RAG with LlamaParse + llama.cpp",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    
    Checks:
    - API status
    - Qdrant connectivity
    - Model file availability
    """
    health = {
        "status": "healthy",
        "api": True,
        "qdrant": False,
        "services": {
            "groq_llm": bool(config.GROQ_API_KEY),
            "jina_embeddings": bool(config.JINA_API_KEY),
        },
    }
    
    # Check Qdrant (with API key for Qdrant Cloud)
    try:
        headers = {}
        if config.QDRANT_API_KEY:
            headers["api-key"] = config.QDRANT_API_KEY
        
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(
                f"{config.QDRANT_URL}/collections",
                headers=headers
            )
            health["qdrant"] = response.status_code == 200
    except Exception:
        health["qdrant"] = False
    
    # Overall status
    if not health["qdrant"]:
        health["status"] = "degraded"
    if not health["services"]["groq_llm"] or not health["services"]["jina_embeddings"]:
        health["status"] = "degraded"
    
    return health


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors."""
    return JSONResponse(
        status_code=500,
        content={
            "detail": sanitize_error(exc),
            "type": type(exc).__name__,
        }
    )


# Run with uvicorn
if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=True,
        log_level="info",
    )
