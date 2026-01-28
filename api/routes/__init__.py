# api/routes/__init__.py
"""API Route Handlers"""

from .indexing import router as indexing_router
from .query import router as query_router
from .websocket import router as websocket_router

__all__ = ["indexing_router", "query_router", "websocket_router"]
