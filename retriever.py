# retriever.py - Multi-Modal Hybrid Retriever with Jina + BM25
"""
Orchestrates Multi-Modal Hybrid Retrieval:
  - Dense search via Jina AI embeddings
  - Sparse search via local BM25 (HashingVectorizer)
  - RRF fusion of dense + sparse text results
  - Jina reranking of fused text + images
  - Groq LLM response generation
"""

import asyncio
from typing import List, Union, Dict, Any, Optional

from qdrant_manager import QdrantManager
from qdrant_client.http.models import SparseVector

# LlamaIndex Core
from llama_index.core import QueryBundle
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, ImageNode, TextNode

from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.base.response.schema import Response

from jina_client import JinaEmbeddings, JinaReranker
from bm25 import BM25SparseEncoder
from rrf_reranker import rrf_fuse

import config
import base64
from groq import AsyncGroq


class MultiModalRetriever(BaseRetriever):
    """
    Multi-Modal Retriever using Jina AI for embeddings and reranking.
    """
    
    def __init__(self, qdrant_manager: QdrantManager):
        super().__init__()
        self.qdrant_manager = qdrant_manager
        
        print("[LOADING] Initializing Jina Embeddings, BM25 Sparse Encoder, and Reranker...")
        
        # 1. Initialize Jina Embeddings (same model for text and images)
        self.embedder = JinaEmbeddings(dimensions=config.JINA_EMBED_DIMENSIONS)
        self.reranker = JinaReranker()
        
        # 2. Initialize BM25 Sparse Encoder (local, no API)
        self.sparse_encoder = BM25SparseEncoder()
        
        # 3. Get Vector Stores from Qdrant
        self.text_store = qdrant_manager.get_text_vector_store()
        self.image_store = qdrant_manager.get_image_vector_store()
        
        print("[OK] Hybrid Retriever Ready (Dense + Sparse + Reranker)")

    async def _get_cached_embedding(self, query_str: str) -> List[float]:
        """Get query embedding with Supabase cache layer.
        
        Checks Supabase query_cache first to avoid redundant Jina API calls.
        Falls back to direct embedding if Supabase is unavailable.
        """
        try:
            from supabase_client import get_cached_embedding, cache_embedding
            
            # Check cache first
            cached = await get_cached_embedding(query_str)
            if cached:
                print(f"   [CACHE] Embedding cache hit for: '{query_str[:40]}...'")
                return cached
            
            # Cache miss — embed via Jina API
            embedding = await self.embedder.aembed_query(query_str)
            
            # Store in cache for future reuse
            try:
                await cache_embedding(query_str, embedding)
            except Exception as e:
                print(f"   [WARN] Failed to cache embedding: {e}")
            
            return embedding
            
        except ImportError:
            # Supabase not available — direct embed
            return await self.embedder.aembed_query(query_str)

    
    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Async hybrid retrieval: [Dense ∥ Sparse ∥ Image] → RRF → Rerank → Return."""
        import time
        start_time = time.perf_counter()
        query_str = query_bundle.query_str
        
        print(f"[HYBRID SEARCH] Retrieving for: '{query_str}'")
        
        # ── Step 1: Get query embeddings (dense + sparse) ──
        embed_start = time.perf_counter()
        query_embedding = await self._get_cached_embedding(query_str)
        sparse_query = self.sparse_encoder.encode_query(query_str)
        print(f"   [PERF] Embedding: {(time.perf_counter() - embed_start)*1000:.0f}ms")
        
        # ── Step 2: PARALLEL Dense + Sparse + Image(visual) + Image(caption) Search ──
        search_start = time.perf_counter()
        
        dense_results, sparse_results, image_visual_results, image_caption_results = await asyncio.gather(
            self.qdrant_manager.async_client.query_points(
                collection_name=config.TEXT_COLLECTION_NAME,
                query=query_embedding,
                using="text-dense",
                limit=config.TEXT_SIMILARITY_TOP_K,
                with_payload=True
            ),
            self.qdrant_manager.async_client.query_points(
                collection_name=config.TEXT_COLLECTION_NAME,
                query=SparseVector(
                    indices=sparse_query["indices"],
                    values=sparse_query["values"]
                ),
                using="text-sparse",
                limit=config.SPARSE_TOP_K,
                with_payload=True
            ),
            self.qdrant_manager.async_client.query_points(
                collection_name=config.IMAGE_COLLECTION_NAME,
                query=query_embedding,
                using="image-visual",
                limit=config.IMAGE_SIMILARITY_TOP_K,
                with_payload=True
            ),
            self.qdrant_manager.async_client.query_points(
                collection_name=config.IMAGE_COLLECTION_NAME,
                query=query_embedding,
                using="caption-text",
                limit=config.IMAGE_SIMILARITY_TOP_K,
                with_payload=True
            )
        )
        print(f"   [PERF] Parallel search: {(time.perf_counter() - search_start)*1000:.0f}ms")
        
        # Convert results
        dense_nodes = self._points_to_text_nodes(dense_results.points)
        sparse_nodes = self._points_to_text_nodes(sparse_results.points)
        
        # Merge visual + caption image results (deduplicate by node ID)
        visual_nodes = self._points_to_image_nodes(image_visual_results.points)
        caption_nodes = self._points_to_image_nodes(image_caption_results.points)
        seen_image_ids = set()
        image_nodes = []
        for n in visual_nodes + caption_nodes:
            if n.node.node_id not in seen_image_ids:
                seen_image_ids.add(n.node.node_id)
                image_nodes.append(n)
        
        print(f"   [DENSE] {len(dense_nodes)}, [SPARSE] {len(sparse_nodes)}, [IMAGE] {len(image_nodes)} (visual:{len(visual_nodes)} + caption:{len(caption_nodes)})")
        
        # ── Step 3: RRF Fusion (dense + sparse text) ──
        rrf_start = time.perf_counter()
        fused_text = rrf_fuse(dense_nodes, sparse_nodes)
        print(f"   [RRF] Fused → {len(fused_text)} unique text nodes ({(time.perf_counter() - rrf_start)*1000:.0f}ms)")
        
        # ── Step 4: Rerank fused text + images ──
        rerank_start = time.perf_counter()
        combined = fused_text + image_nodes
        final_nodes = await self._arerank_nodes(query_str, combined, config.FINAL_RERANK_TOP_N)
        print(f"   [PERF] Reranking: {(time.perf_counter() - rerank_start)*1000:.0f}ms")
        print(f"   [RANK] Final: {len(final_nodes)} nodes")
        
        total_time = (time.perf_counter() - start_time) * 1000
        print(f"[OK] Final: {len(final_nodes)} nodes in {total_time:.0f}ms")
        
        return final_nodes
    
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Sync fallback — required by BaseRetriever abstract class."""
        import asyncio
        return asyncio.get_event_loop().run_until_complete(self._aretrieve(query_bundle))
    
    # ═══════════════════════════════════════════════════════════════════
    # HELPER METHODS
    # ═══════════════════════════════════════════════════════════════════
    
    def _points_to_text_nodes(self, points) -> List[NodeWithScore]:
        """Convert Qdrant query points to LlamaIndex TextNode list."""
        nodes = []
        for point in points:
            payload = point.payload or {}
            text_content = payload.get("text_chunk", "")
            node = TextNode(
                id_=str(point.id),
                text=text_content,
                metadata={k: v for k, v in payload.items() if k != "text_chunk"}
            )
            nodes.append(NodeWithScore(node=node, score=point.score))
        return nodes
    
    def _points_to_image_nodes(self, points) -> List[NodeWithScore]:
        """Convert Qdrant query points to image NodeWithScore list.
        
        Uses caption as the text content for reranking quality.
        Falls back to image path if no caption is available.
        """
        nodes = []
        for point in points:
            payload = point.payload or {}
            img_path = payload.get("image_url") or payload.get("image_path", "")
            caption = payload.get("caption", "")
            
            # Use caption as text content for better reranking
            text_content = caption if caption else f"Image from {payload.get('file_name', 'unknown')} (page {payload.get('page', '?')})"
            
            node = TextNode(
                id_=str(point.id),
                text=text_content,
                metadata={**payload, "image_path": img_path}
            )
            nodes.append(NodeWithScore(node=node, score=point.score))
        return nodes
    
    def _get_rerank_text(self, node_with_score: NodeWithScore) -> str:
        """Get text content for reranking. For image nodes, use caption if available."""
        metadata = node_with_score.node.metadata or {}
        if "image_path" in metadata:
            # Use Groq-generated caption for reranking (much better than filename)
            caption = metadata.get("caption", "")
            if caption:
                return caption
            # Fallback to metadata-based proxy
            parts = []
            if metadata.get("original_name"):
                parts.append(f"Image: {metadata['original_name']}")
            if metadata.get("source"):
                parts.append(f"Source: {metadata['source']}")
            if metadata.get("page") is not None:
                parts.append(f"Page: {metadata['page']}")
            return " | ".join(parts) if parts else "image"
        return node_with_score.node.get_content()
    
    
    async def _arerank_nodes(self, query: str, nodes: List[NodeWithScore], top_n: int) -> List[NodeWithScore]:
        """Async rerank a list of nodes using Jina Reranker."""
        if not nodes:
            return []
        documents = [self._get_rerank_text(n) for n in nodes]
        rerank_results = await self.reranker.arerank(query, documents, top_n=top_n)
        reranked = []
        for result in rerank_results:
            idx = result["index"]
            score = result["relevance_score"]
            reranked.append(NodeWithScore(node=nodes[idx].node, score=score))
        return reranked


class GroqGenerator(CustomQueryEngine):
    """
    Query engine using Groq's Vision models (Llama 3.2).
    Replaces LlamaCppGenerator with Cloud API inference.
    """
    retriever: MultiModalRetriever
    aclient: Any  # AsyncGroq client
    model_name: str
    
    def __init__(
        self, 
        retriever: MultiModalRetriever,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        """
        Initialize the generator with the retriever and async Groq client.
        """
        api_key = api_key or config.GROQ_API_KEY
        if not api_key:
            raise ValueError("GROQ_API_KEY is not set.")
        
        aclient = AsyncGroq(api_key=api_key)
        resolved_model = model_name or config.GROQ_MODEL_NAME
        
        super().__init__(
            retriever=retriever,
            aclient=aclient,
            model_name=resolved_model,
        )

    def _encode_image(self, image_path: str, max_pixels: int = 30000000) -> str:
        """Encode image file or URL to base64 string, resizing if too large.
        
        Supports:
        - Local file paths
        - HTTP/HTTPS URLs (Supabase storage, LlamaParse, etc.)
        
        Groq has a 33M pixel limit, so we resize to stay under 30M for safety.
        """
        from PIL import Image
        import io
        import httpx
        
        try:
            # Check if it's a URL
            is_url = image_path.startswith('http://') or image_path.startswith('https://')
            
            if is_url:
                # Download image from URL
                print(f"   [IMG] Fetching from URL: {image_path[:60]}...")
                with httpx.Client(timeout=30.0) as client:
                    response = client.get(image_path)
                    response.raise_for_status()
                    image_data = io.BytesIO(response.content)
                img = Image.open(image_data)
            else:
                # Open local file
                img = Image.open(image_path)
            
            with img:
                # Check if resizing is needed
                width, height = img.size
                total_pixels = width * height
                
                if total_pixels > max_pixels:
                    # Calculate scale factor
                    scale = (max_pixels / total_pixels) ** 0.5
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    
                    print(f"   [IMG] Resizing {width}x{height} -> {new_width}x{new_height}")
                    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                # Convert to RGB if needed (for RGBA/P modes)
                if img.mode in ('RGBA', 'P'):
                    img = img.convert('RGB')
                
                # Encode to base64
                buffer = io.BytesIO()
                img.save(buffer, format='JPEG', quality=85)
                buffer.seek(0)
                return base64.b64encode(buffer.read()).decode('utf-8')
                
        except Exception as e:
            print(f"   [WARN] Failed to process image {image_path}: {e}")
            
            # Fallback for local files only
            if not (image_path.startswith('http://') or image_path.startswith('https://')):
                try:
                    with open(image_path, "rb") as f:
                        return base64.b64encode(f.read()).decode('utf-8')
                except:
                    pass
            
            return ""  # Return empty string if all fails

    def _construct_messages(self, query_str: str, text_context: List[str], image_paths: List[str]) -> List[Dict]:
        """Construct the message payload for Groq API."""
        # Number sources for citation
        context_str = "\n\n".join([f"[Source {i+1}]: {ctx}" for i, ctx in enumerate(text_context)])
        
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert assistant. Answer using ONLY the provided context. "
                    "Match your response length to the question complexity — brief for simple questions, "
                    "detailed for complex ones. Use formatting (bullet points, headers) when it improves clarity. "
                    "Cite sources as [Source N] when referencing specific information."
                )
            },
            {
                "role": "user",
                "content": []
            }
        ]
        
        # Add text content
        user_content = messages[1]["content"]
        user_prompt = f"Context:\n{context_str}\n\nQuestion: {query_str}"
        user_content.append({"type": "text", "text": user_prompt})
        
        # Add images
        for img_path in image_paths:
            try:
                base64_image = self._encode_image(img_path)
                
                # Skip empty/failed encodings
                if not base64_image:
                    continue
                    
                # Determine mime type simply
                mime_type = "image/jpeg"
                if img_path.lower().endswith(".png"):
                    mime_type = "image/png"
                elif img_path.lower().endswith(".webp"):
                    mime_type = "image/webp"
                    
                user_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{base64_image}"
                    }
                })
            except Exception as e:
                print(f"[WARN] Failed to load image {img_path}: {e}")
                
        return messages

    async def _rewrite_query(self, query_str: str) -> str:
        """Rewrite short/vague queries for better document retrieval."""
        try:
            completion = await self.aclient.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": (
                        "Rewrite the following user query to be more specific and detailed "
                        "for searching a document. Keep it as a single search query. "
                        "Output ONLY the rewritten query, nothing else."
                    )},
                    {"role": "user", "content": query_str}
                ],
                max_tokens=100,
                temperature=0.3
            )
            rewritten = completion.choices[0].message.content.strip().strip('"')
            return rewritten if rewritten else query_str
        except Exception as e:
            print(f"   [WARN] Query rewrite failed: {e}")
            return query_str
    
    def custom_query(self, query_str: str) -> Response:
        """Sync fallback — required by CustomQueryEngine abstract class."""
        import asyncio
        return asyncio.get_event_loop().run_until_complete(self.acustom_query(query_str))

    async def acustom_query(self, query_bundle: Union[str, QueryBundle]) -> Response:
        """Execute a query with async retrieval, semantic cache, and Groq generation."""
        if isinstance(query_bundle, str):
            query_bundle = QueryBundle(query_bundle)
        
        query_str = query_bundle.query_str
        
        # 1. Embed query (needed for both cache check and retrieval)
        query_embedding = await self.retriever._get_cached_embedding(query_str)
        
        # 2. Check semantic cache
        try:
            cached = await self.retriever.qdrant_manager.search_response_cache(query_embedding)
            if cached:
                # Reconstruct nodes from cached sources
                cached_nodes = []
                for src in (cached.get("sources") or []):
                    node = TextNode(text=src.get("content", ""), metadata=src.get("metadata", {}))
                    cached_nodes.append(NodeWithScore(node=node, score=src.get("score", 0.0)))
                return Response(response=cached["response"], source_nodes=cached_nodes)
        except Exception as e:
            print(f"   [WARN] Cache check failed: {e}")
        
        # 3. Retrieve (Async)
        nodes: List[NodeWithScore] = await self.retriever._aretrieve(query_bundle)
        
        # 4. Separate Text and Images
        text_context = []
        image_paths = []
        
        for node in nodes:
            if isinstance(node.node, ImageNode) or 'image_path' in node.metadata:
                img_path = node.metadata.get('image_path')
                if img_path:
                    image_paths.append(img_path)
            else:
                text_context.append(node.get_content())
        
        # 5. Construct Messages
        messages = self._construct_messages(query_str, text_context, image_paths)
        
        # 6. Generate Answer (Async)
        try:
            completion = await self.aclient.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=config.LLM_TEMPERATURE,
                max_tokens=config.LLM_MAX_NEW_TOKENS,
                stream=False
            )
            response_text = completion.choices[0].message.content
        except Exception as e:
            response_text = f"Error generating response: {e}"
        
        # 7. Store in semantic cache
        try:
            sources_for_cache = []
            for node in nodes:
                metadata = node.node.metadata if hasattr(node, 'node') else {}
                node_type = "image" if "image_path" in metadata else "text"
                sources_for_cache.append({
                    "content": node.get_content()[:500],
                    "score": node.score or 0.0,
                    "type": node_type,
                    "metadata": dict(metadata) if metadata else {}
                })
            await self.retriever.qdrant_manager.store_response_cache(
                query_embedding=query_embedding,
                query_text=query_str,
                response_text=response_text,
                sources=sources_for_cache
            )
        except Exception as e:
            print(f"   [WARN] Cache store failed: {e}")
        
        return Response(response=response_text, source_nodes=nodes)
    
    async def astream_query_detailed(self, query_bundle: Union[str, QueryBundle]):
        """
        Enhanced async streaming with detailed progress for WebSocket.
        
        Yields dicts with event types:
        - {"type": "phase", "phase": "embedding", "status": "started"}
        - {"type": "progress", "phase": "search", "data": {...}}
        - {"type": "chunks_found", "text_count": N, "image_count": M, "chunks": [...]}
        - {"type": "reranking", "status": "started/completed", "results": [...]}
        - {"type": "generation", "chunk": "text"}
        - {"type": "sources", "sources": [...]}
        - {"type": "done"}
        """
        import time
        
        if isinstance(query_bundle, str):
            query_bundle = QueryBundle(query_bundle)
        
        query_str = query_bundle.query_str
        start_time = time.perf_counter()
        
        # ═══════════════════════════════════════════════════════════════
        # PHASE 1: EMBEDDING (with optional query expansion)
        # ═══════════════════════════════════════════════════════════════
        original_query = query_str
        
        # Expand short/vague queries for better retrieval
        if len(query_str.split()) <= 8:
            yield {
                "type": "phase",
                "phase": "embedding",
                "status": "started",
                "message": "✏️ Expanding query for better retrieval..."
            }
            rewritten = await self._rewrite_query(query_str)
            if rewritten and rewritten != query_str:
                query_str = rewritten
                print(f"   [REWRITE] '{original_query}' → '{query_str}'")
        else:
            yield {
                "type": "phase",
                "phase": "embedding",
                "status": "started",
                "message": "🔮 Creating query embedding..."
            }
        
        embed_start = time.perf_counter()
        query_embedding = await self.retriever._get_cached_embedding(query_str)
        embed_time = (time.perf_counter() - embed_start) * 1000
        
        embed_msg = f"✓ Embedding created ({embed_time:.0f}ms)"
        if query_str != original_query:
            embed_msg += f" [Expanded]"
        
        yield {
            "type": "phase",
            "phase": "embedding",
            "status": "completed",
            "message": embed_msg,
            "duration_ms": embed_time
        }
        
        # ═══════════════════════════════════════════════════════════════
        # CACHE CHECK: Search for semantically similar cached response
        # ═══════════════════════════════════════════════════════════════
        try:
            cached = await self.retriever.qdrant_manager.search_response_cache(query_embedding)
            if cached:
                yield {
                    "type": "phase",
                    "phase": "cache",
                    "status": "hit",
                    "message": "⚡ Semantic cache hit — returning cached response"
                }
                
                # Emit cached response as generation chunks
                cached_response = cached.get("response", "")
                for i in range(0, len(cached_response), 50):
                    yield {
                        "type": "generation",
                        "chunk": cached_response[i:i+50]
                    }
                
                yield {
                    "type": "phase",
                    "phase": "generation",
                    "status": "completed",
                    "message": "✓ Response complete (from cache)"
                }
                
                # Emit cached sources
                yield {
                    "type": "sources",
                    "sources": cached.get("sources", []),
                    "total_duration_ms": (time.perf_counter() - start_time) * 1000
                }
                
                yield {"type": "done", "total_duration_ms": (time.perf_counter() - start_time) * 1000, "cached": True}
                return  # Short-circuit — skip entire pipeline
        except Exception as e:
            print(f"   [WARN] Cache check failed: {e}")
        
        # ═══════════════════════════════════════════════════════════════
        # PHASE 2: HYBRID SEARCH (Dense + Sparse + Image)
        # ═══════════════════════════════════════════════════════════════
        yield {
            "type": "phase",
            "phase": "search",
            "status": "started",
            "message": "🔍 Searching dense, sparse & image vectors..."
        }
        
        search_start = time.perf_counter()
        
        # Generate sparse query vector (local, instant)
        sparse_query = self.retriever.sparse_encoder.encode_query(query_str)
        
        # Parallel 4-way search (text dense + text sparse + image visual + image caption)
        dense_results, sparse_results, image_visual_results, image_caption_results = await asyncio.gather(
            self.retriever.qdrant_manager.async_client.query_points(
                collection_name=config.TEXT_COLLECTION_NAME,
                query=query_embedding,
                using="text-dense",
                limit=config.TEXT_SIMILARITY_TOP_K,
                with_payload=True
            ),
            self.retriever.qdrant_manager.async_client.query_points(
                collection_name=config.TEXT_COLLECTION_NAME,
                query=SparseVector(
                    indices=sparse_query["indices"],
                    values=sparse_query["values"]
                ),
                using="text-sparse",
                limit=config.SPARSE_TOP_K,
                with_payload=True
            ),
            self.retriever.qdrant_manager.async_client.query_points(
                collection_name=config.IMAGE_COLLECTION_NAME,
                query=query_embedding,
                using="image-visual",
                limit=config.IMAGE_SIMILARITY_TOP_K,
                with_payload=True
            ),
            self.retriever.qdrant_manager.async_client.query_points(
                collection_name=config.IMAGE_COLLECTION_NAME,
                query=query_embedding,
                using="caption-text",
                limit=config.IMAGE_SIMILARITY_TOP_K,
                with_payload=True
            )
        )
        search_time = (time.perf_counter() - search_start) * 1000
        
        # Convert to nodes
        dense_nodes = self.retriever._points_to_text_nodes(dense_results.points)
        sparse_nodes = self.retriever._points_to_text_nodes(sparse_results.points)
        
        # Merge visual + caption image results (deduplicate by node ID)
        visual_nodes = self.retriever._points_to_image_nodes(image_visual_results.points)
        caption_nodes = self.retriever._points_to_image_nodes(image_caption_results.points)
        seen_image_ids = set()
        image_nodes = []
        for n in visual_nodes + caption_nodes:
            if n.node.node_id not in seen_image_ids:
                seen_image_ids.add(n.node.node_id)
                image_nodes.append(n)
        
        # Emit found chunks
        yield {
            "type": "chunks_found",
            "text_count": len(dense_nodes) + len(sparse_nodes),
            "dense_count": len(dense_nodes),
            "sparse_count": len(sparse_nodes),
            "image_count": len(image_nodes),
            "duration_ms": search_time,
            "message": f"📚 Found {len(dense_nodes)} dense, {len(sparse_nodes)} sparse & {len(image_nodes)} images ({search_time:.0f}ms)",
            "chunks": [
                {
                    "id": n.node.id_,
                    "preview": n.node.get_content()[:150] + "..." if len(n.node.get_content()) > 150 else n.node.get_content(),
                    "score": n.score,
                    "type": "text",
                    "metadata": dict(n.node.metadata) if n.node.metadata else {}
                }
                for n in dense_nodes[:5]
            ] + [
                {
                    "id": n.node.id_,
                    "preview": n.node.metadata.get("image_path", "")[-50:] if n.node.metadata else "",
                    "score": n.score,
                    "type": "image",
                    "metadata": dict(n.node.metadata) if n.node.metadata else {}
                }
                for n in image_nodes[:5]
            ]
        }
        
        # ═══════════════════════════════════════════════════════════════
        # PHASE 3: RRF FUSION (dense + sparse text)
        # ═══════════════════════════════════════════════════════════════
        yield {
            "type": "phase",
            "phase": "fusion",
            "status": "started",
            "message": "🔀 RRF fusing dense + sparse text results..."
        }
        
        fused_text = rrf_fuse(dense_nodes, sparse_nodes)
        
        yield {
            "type": "phase",
            "phase": "fusion",
            "status": "completed",
            "message": f"✓ RRF fused → {len(fused_text)} unique text nodes"
        }
        
        # ═══════════════════════════════════════════════════════════════
        # PHASE 4: RERANKING (fused text + images)
        # ═══════════════════════════════════════════════════════════════
        yield {
            "type": "phase",
            "phase": "reranking",
            "status": "started",
            "message": "⚖️ Reranking fused text + images..."
        }
        
        rerank_start = time.perf_counter()
        combined = fused_text + image_nodes
        all_nodes = await self.retriever._arerank_nodes(query_str, combined, config.FINAL_RERANK_TOP_N)
        rerank_time = (time.perf_counter() - rerank_start) * 1000
        
        yield {
            "type": "phase",
            "phase": "reranking",
            "status": "completed",
            "message": f"✓ Reranked → {len(all_nodes)} results ({rerank_time:.0f}ms)",
            "duration_ms": rerank_time,
            "reranked": [
                {
                    "id": n.node.id_,
                    "score": n.score,
                    "preview": n.node.get_content()[:100] + "..."
                }
                for n in all_nodes[:5]
            ]
        }
        
        # ═══════════════════════════════════════════════════════════════
        # PHASE 4: LLM GENERATION
        # ═══════════════════════════════════════════════════════════════
        yield {
            "type": "phase",
            "phase": "generation",
            "status": "started",
            "message": "✨ Generating response..."
        }
        
        # Prepare context
        text_context = []
        image_paths = []
        for node in all_nodes:
            if isinstance(node.node, ImageNode) or 'image_path' in node.metadata:
                img_path = node.metadata.get('image_path')
                if img_path:
                    image_paths.append(img_path)
            else:
                text_context.append(node.get_content())
        
        messages = self._construct_messages(query_str, text_context, image_paths)
        
        try:
            stream = await self.aclient.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=config.LLM_TEMPERATURE,
                max_tokens=config.LLM_MAX_NEW_TOKENS,
                stream=True
            )
            
            full_response_text = ""
            async for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    full_response_text += content
                    yield {
                        "type": "generation",
                        "chunk": content
                    }
        except Exception as e:
            yield {
                "type": "error",
                "message": f"Generation error: {e}"
            }
        
        yield {
            "type": "phase",
            "phase": "generation",
            "status": "completed",
            "message": "✓ Response complete"
        }
        
        # ═══════════════════════════════════════════════════════════════
        # FINAL: SOURCES + CACHE STORE
        # ═══════════════════════════════════════════════════════════════
        sources = []
        for node in all_nodes:
            # Access metadata from the inner node (NodeWithScore wraps the actual node)
            metadata = node.node.metadata if hasattr(node, 'node') else (node.metadata if hasattr(node, 'metadata') else {})
            node_type = "image" if "image_path" in metadata else "text"
            
            score = node.score or 0.0
            
            sources.append({
                "content": node.get_content()[:500] if hasattr(node, 'get_content') else node.node.get_content()[:500],
                "score": score,
                "type": node_type,
                "metadata": dict(metadata) if metadata else {}
            })
        
        # Store in semantic cache
        try:
            await self.retriever.qdrant_manager.store_response_cache(
                query_embedding=query_embedding,
                query_text=query_str,
                response_text=full_response_text,
                sources=sources
            )
        except Exception as e:
            print(f"   [WARN] Cache store failed: {e}")
        
        total_time = (time.perf_counter() - start_time) * 1000
        
        yield {
            "type": "sources",
            "sources": sources,
            "total_duration_ms": total_time
        }
        
        yield {"type": "done", "total_duration_ms": total_time}
    
    async def astream_query(self, query_bundle: Union[str, QueryBundle]):
        """
        Async streaming version of query with live status updates and semantic cache.
        
        Yields tuples of (event_type, data, nodes):
        - ("status", "Searching...", None) - Status update
        - ("chunk", "text", nodes) - LLM response chunk
        """
        if isinstance(query_bundle, str):
            query_bundle = QueryBundle(query_bundle)
        
        query_str = query_bundle.query_str
        
        # 1. Embed query (needed for cache check)
        query_embedding = await self.retriever._get_cached_embedding(query_str)
        
        # 2. Check semantic cache
        try:
            cached = await self.retriever.qdrant_manager.search_response_cache(query_embedding)
            if cached:
                yield ("status", "⚡ Cache hit — returning cached response", None)
                cached_response = cached.get("response", "")
                # Emit in chunks to match the expected interface
                for i in range(0, len(cached_response), 50):
                    yield ("chunk", cached_response[i:i+50], [])
                return
        except Exception as e:
            print(f"   [WARN] Cache check failed: {e}")
        
        # === Status: Searching ===
        yield ("status", "🔍 Searching documents...", None)
        
        # 3. Retrieve
        nodes: List[NodeWithScore] = await self.retriever._aretrieve(query_bundle)
        
        # === Status: Found nodes ===
        text_count = sum(1 for n in nodes if 'image_path' not in n.metadata)
        img_count = len(nodes) - text_count
        yield ("status", f"📚 Found {text_count} text & {img_count} image sources", None)
        
        # 4. Prepare context
        text_context = []
        image_paths = []
        for node in nodes:
            if isinstance(node.node, ImageNode) or 'image_path' in node.metadata:
                img_path = node.metadata.get('image_path')
                if img_path:
                    image_paths.append(img_path)
            else:
                text_context.append(node.get_content())
        
        # === Status: Generating ===
        yield ("status", "✨ Generating response...", None)
                
        # 5. Construct Messages
        messages = self._construct_messages(query_str, text_context, image_paths)
        
        # 6. Stream response + collect for cache
        full_response_text = ""
        try:
            stream = await self.aclient.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=config.LLM_TEMPERATURE,
                max_tokens=config.LLM_MAX_NEW_TOKENS,
                stream=True
            )
            
            async for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    full_response_text += content
                    yield ("chunk", content, nodes)
        except Exception as e:
            yield ("chunk", f"Error: {e}", nodes)
        
        # 7. Store in semantic cache
        try:
            sources_for_cache = []
            for node in nodes:
                metadata = node.node.metadata if hasattr(node, 'node') else {}
                sources_for_cache.append({
                    "content": node.get_content()[:500],
                    "score": node.score or 0.0,
                    "type": "image" if "image_path" in metadata else "text",
                    "metadata": dict(metadata) if metadata else {}
                })
            await self.retriever.qdrant_manager.store_response_cache(
                query_embedding=query_embedding,
                query_text=query_str,
                response_text=full_response_text,
                sources=sources_for_cache
            )
        except Exception as e:
            print(f"   [WARN] Cache store failed: {e}")


# Convenience function to initialize the full pipeline
def create_query_engine(qdrant_url: Optional[str] = None) -> GroqGenerator:
    """
    Create and return a fully initialized query engine.
    
    Args:
        qdrant_url: Optional Qdrant URL (defaults to config)
        
    Returns:
        GroqGenerator instance ready for queries
    """
    print("[*] Initializing Query Engine Pipeline...")
    
    qdrant_url = qdrant_url or config.QDRANT_URL
    qdrant_api_key = config.QDRANT_API_KEY
    q_manager = QdrantManager(url=qdrant_url, api_key=qdrant_api_key)
    
    # Ensure semantic response cache collection exists
    # Non-fatal: if Qdrant is temporarily unreachable, continue without cache
    try:
        q_manager.create_response_cache_collection(config.JINA_EMBED_DIMENSIONS)
    except Exception as e:
        print(f"[WARN] Response cache collection unavailable (will skip caching): {e}")
    
    retriever = MultiModalRetriever(q_manager)
    engine = GroqGenerator(retriever)
    
    print("[OK] Query Engine Ready!")
    return engine


if __name__ == "__main__":
    # Test the pipeline
    import asyncio
    
    async def main():
        print("[*] Testing Multimodal Pipeline...")
        try:
            engine = create_query_engine()
            user_query = "Summarize the document and explain any diagrams found."
            print(f"\n[?] Asking: {user_query}")
            final_answer = await engine.acustom_query(user_query)
            print("\n" + "=" * 50)
            print("[ANSWER] FINAL ANSWER:")
            print("=" * 50)
            print(final_answer)
            print("=" * 50)
        except Exception as e:
            print(f"[ERROR] Error during generation: {e}")
            import traceback
            traceback.print_exc()
    
    asyncio.run(main())