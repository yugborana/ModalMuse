# retriever.py - Multi-Modal Hybrid Retriever with Local Embeddings + BM25
"""
Orchestrates Multi-Modal Hybrid Retrieval:
  - Dense search via local Infinity embeddings (jina-clip-v1)
  - Sparse search via local BM25 (HashingVectorizer)
  - RRF fusion of dense + sparse text results
  - Local reranking of fused text + images
  - Groq LLM response generation
"""

import asyncio
from typing import List, Union, Dict, Any, Optional, Tuple

from qdrant_manager import QdrantManager
from qdrant_client.http.models import SparseVector

# LlamaIndex Core
from llama_index.core import QueryBundle
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, ImageNode, TextNode

from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.base.response.schema import Response

from local_client import LocalEmbeddings, LocalReranker
from bm25 import BM25SparseEncoder
from rrf_reranker import rrf_fuse

import config
import base64
from groq import AsyncGroq


class MultiModalRetriever(BaseRetriever):
    """
    Multi-Modal Retriever using local Infinity server for embeddings and reranking.
    Uses separate text (bge-small) and image (jina-clip) embedding models.
    """
    
    def __init__(self, qdrant_manager: QdrantManager):
        super().__init__()
        self.qdrant_manager = qdrant_manager
        
        print("[LOADING] Initializing Local Embeddings, BM25 Sparse Encoder, and Reranker...")
        
        # 1. Initialize separate text and image embedders
        self.text_embedder = LocalEmbeddings(
            model=config.LOCAL_TEXT_MODEL,
            dimensions=config.LOCAL_TEXT_DIMENSIONS
        )
        self.image_embedder = LocalEmbeddings(
            model=config.LOCAL_IMAGE_MODEL,
            dimensions=config.LOCAL_IMAGE_DIMENSIONS
        )
        # Backward compat alias (used by aembed_query cache flow)
        self.embedder = self.text_embedder
        self.reranker = LocalReranker()
        
        # 2. Initialize BM25 Sparse Encoder (local, no API)
        self.sparse_encoder = BM25SparseEncoder()
        
        # Note: We query Qdrant directly via async_client.query_points()
        # LlamaIndex QdrantVectorStore is not used in the retrieval pipeline.
        
        print(f"[OK] Text Embedder: {config.LOCAL_TEXT_MODEL} ({config.LOCAL_TEXT_DIMENSIONS}-dim)")
        print(f"[OK] Image Embedder: {config.LOCAL_IMAGE_MODEL} ({config.LOCAL_IMAGE_DIMENSIONS}-dim)")
        print("[OK] Hybrid Retriever Ready (Dense + Sparse + Reranker)")

    # Removed `_get_cached_embedding` since local Infinity embeddings are instantaneous,

    
    async def _retrieve_hybrid(self, query_str: str, text_query_embedding: List[float], image_query_embedding: List[float]) -> Dict:
        """Core hybrid retrieval pipeline returning detailed intermediate results.
        
        This is the single source of truth for the retrieval pipeline.
        Both `_aretrieve` (non-streaming) and `astream_query_detailed` (streaming) call this.
        
        Pipeline:
          Text:  [Dense ∥ Sparse] → RRF Fusion
          Image: [Caption-text ∥ Visual] → Deduplicate
          All:   [Fused text + Images] → Unified Cross-Encoder Rerank → Slot allocation
          Final: Top text_slots text + Top image_slots images
        
        Returns:
            Dict with keys: dense_nodes, sparse_nodes, image_nodes, fused_text,
                            reranked_text, reranked_images, all_nodes, timings
        """
        import time
        
        # Configurable slot allocation
        image_slots = getattr(config, 'IMAGE_RESULT_SLOTS', 2)
        text_slots = config.FINAL_RERANK_TOP_N - image_slots
        
        # ── Step 1: Sparse query (local, instant) ──
        sparse_query = self.sparse_encoder.encode_query(query_str)
        
        # ── Step 2: PARALLEL 4-way Search ──
        search_start = time.perf_counter()
        
        dense_results, sparse_results, image_visual_results, image_caption_results = await asyncio.gather(
            self.qdrant_manager.async_client.query_points(
                collection_name=config.TEXT_COLLECTION_NAME,
                query=text_query_embedding,
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
                query=image_query_embedding,
                using="image-visual",
                limit=config.IMAGE_SIMILARITY_TOP_K,
                with_payload=True
            ),
            self.qdrant_manager.async_client.query_points(
                collection_name=config.IMAGE_COLLECTION_NAME,
                query=text_query_embedding,
                using="caption-text",
                limit=config.IMAGE_SIMILARITY_TOP_K,
                with_payload=True
            )
        )
        search_time = (time.perf_counter() - search_start) * 1000
        
        # Convert results
        dense_nodes = self._points_to_text_nodes(dense_results.points)
        sparse_nodes = self._points_to_text_nodes(sparse_results.points)
        
        # Merge image results: prioritize caption-text over visual
        caption_nodes = self._points_to_image_nodes(image_caption_results.points)
        visual_nodes = self._points_to_image_nodes(image_visual_results.points)
        seen_image_ids = set()
        image_nodes = []
        for n in caption_nodes:
            if n.node.node_id not in seen_image_ids:
                seen_image_ids.add(n.node.node_id)
                image_nodes.append(n)
        for n in visual_nodes:
            if n.node.node_id not in seen_image_ids:
                seen_image_ids.add(n.node.node_id)
                image_nodes.append(n)
        
        print(f"   [DENSE] {len(dense_nodes)}, [SPARSE] {len(sparse_nodes)}, [IMAGE] {len(image_nodes)} (caption:{len(caption_nodes)} + visual:{len(visual_nodes)})")
        
        # ── Step 3: RRF Fusion (dense + sparse text only) ──
        fused_text = rrf_fuse(dense_nodes, sparse_nodes)
        print(f"   [RRF] Fused → {len(fused_text)} unique text nodes")
        
        # ── Step 4: Unified Reranking (text + images together) ──
        # With rich captions (domain terms + visible text + keywords), the
        # cross-encoder can now meaningfully compare image captions against
        # text passages. We rerank all candidates in a single pass, then
        # allocate the top results into text/image slots.
        rerank_start = time.perf_counter()
        
        # Combine all candidates for unified reranking
        all_candidates = fused_text + image_nodes
        total_rerank_slots = config.FINAL_RERANK_TOP_N + 4  # Over-fetch to ensure enough of each type
        
        reranked_all = await self._arerank_nodes(query_str, all_candidates, total_rerank_slots)
        rerank_time = (time.perf_counter() - rerank_start) * 1000
        
        # Split reranked results back into text and image buckets
        reranked_text = []
        reranked_images = []
        for node in reranked_all:
            is_image = "image_path" in (node.node.metadata or {})
            if is_image and len(reranked_images) < image_slots:
                reranked_images.append(node)
            elif not is_image and len(reranked_text) < text_slots:
                reranked_text.append(node)
            # Stop once both buckets are full
            if len(reranked_text) >= text_slots and len(reranked_images) >= image_slots:
                break
        
        all_nodes = reranked_text + reranked_images
        
        # Log what got selected
        for n in reranked_text:
            print(f"   [TEXT] score={n.score:.4f} page={n.node.metadata.get('page','?')} text={n.node.get_content()[:60]}...")
        for n in reranked_images:
            print(f"   [IMG]  score={n.score:.4f} page={n.node.metadata.get('page','?')} caption={n.node.metadata.get('caption','')[:60]}...")
        
        return {
            "dense_nodes": dense_nodes,
            "sparse_nodes": sparse_nodes,
            "image_nodes": image_nodes,
            "fused_text": fused_text,
            "reranked_text": reranked_text,
            "reranked_images": reranked_images,
            "all_nodes": all_nodes,
            "timings": {
                "search_ms": search_time,
                "rerank_ms": rerank_time,
            }
        }

    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Async hybrid retrieval — thin wrapper around _retrieve_hybrid.
        
        Handles embedding, then delegates to the shared pipeline.
        """
        import time
        start_time = time.perf_counter()
        query_str = query_bundle.query_str
        
        print(f"[HYBRID SEARCH] Retrieving for: '{query_str}'")
        
        # Step 1: Get query embeddings (text + image in parallel)
        embed_start = time.perf_counter()
        text_query_embedding, image_query_embedding = await asyncio.gather(
            self.embedder.aembed_query(query_str),
            self.image_embedder.aembed_query(query_str),
        )
        print(f"   [PERF] Embedding: {(time.perf_counter() - embed_start)*1000:.0f}ms")
        
        # Step 2: Run shared retrieval pipeline
        result = await self._retrieve_hybrid(query_str, text_query_embedding, image_query_embedding)
        
        total_time = (time.perf_counter() - start_time) * 1000
        all_nodes = result["all_nodes"]
        print(f"[OK] Final: {len(all_nodes)} nodes ({len(result['reranked_text'])} text + {len(result['reranked_images'])} images) in {total_time:.0f}ms")
        
        return all_nodes
    
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
        """Async rerank a list of nodes using Local Reranker."""
        if not nodes:
            return []
        # Clamp top_n to available nodes
        top_n = min(top_n, len(nodes))
        if top_n <= 0:
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

    async def _encode_image(self, image_path: str, max_pixels: int = 30000000) -> str:
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
                # Download image from URL (async to avoid blocking event loop)
                print(f"   [IMG] Fetching from URL: {image_path[:60]}...")
                from local_client import get_async_client
                client = get_async_client()
                response = await client.get(image_path)
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

    async def _construct_messages(self, query_str: str, text_context: List[str], image_paths: List[str]) -> List[Dict]:
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
                base64_image = await self._encode_image(img_path)
                
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

    async def _check_semantic_cache(self, query_embedding: List[float]) -> Optional[Response]:
        """Check semantic cache and return a fully formed Response if hit."""
        try:
            cached = await self.retriever.qdrant_manager.search_response_cache(query_embedding)
            if cached:
                cached_nodes = []
                for src in (cached.get("sources") or []):
                    node = TextNode(text=src.get("content", ""), metadata=src.get("metadata", {}))
                    cached_nodes.append(NodeWithScore(node=node, score=src.get("score", 0.0)))
                return Response(response=cached["response"], source_nodes=cached_nodes)
        except Exception as e:
            print(f"   [WARN] Cache check failed: {e}")
        return None

    def _separate_nodes(self, nodes: List[NodeWithScore]) -> Tuple[List[str], List[str]]:
        """Separate retrieved nodes into text contexts and image paths."""
        text_context = []
        image_paths = []
        for node in nodes:
            if isinstance(node.node, ImageNode) or 'image_path' in node.metadata:
                img_path = node.metadata.get('image_path')
                if img_path:
                    image_paths.append(img_path)
            else:
                text_context.append(node.get_content())
        return text_context, image_paths

    async def _store_response_cache(self, query_embedding: List[float], query_str: str, response_text: str, nodes: List[NodeWithScore]) -> None:
        """Store the generated response and sources in the semantic cache."""
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

    async def acustom_query(self, query_bundle: Union[str, QueryBundle]) -> Response:
        """Execute a query with async retrieval, semantic cache, and Groq generation."""
        if isinstance(query_bundle, str):
            query_bundle = QueryBundle(query_bundle)
        
        original_query = query_bundle.query_str
        query_str = original_query
        
        # 1. Expand short/vague queries for better retrieval
        if len(query_str.split()) <= 8:
            rewritten = await self._rewrite_query(query_str)
            if rewritten and rewritten != query_str:
                query_str = rewritten
                print(f"   [REWRITE] '{original_query}' → '{query_str}'")
                query_bundle.query_str = query_str
        
        # 2. Embed query
        query_embedding = await self.retriever.embedder.aembed_query(query_str)
        
        # 3. Check semantic cache
        cached_response = await self._check_semantic_cache(query_embedding)
        if cached_response:
            return cached_response
        
        # 4. Retrieve (Async)
        nodes: List[NodeWithScore] = await self.retriever._aretrieve(query_bundle)
        
        # 5. Separate Text and Images
        text_context, image_paths = self._separate_nodes(nodes)
        
        # 6. Construct Messages
        messages = await self._construct_messages(query_str, text_context, image_paths)
        
        # 7. Generate Answer (Async)
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
        
        # 8. Store in semantic cache
        await self._store_response_cache(query_embedding, query_str, response_text, nodes)
        
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
        query_embedding, image_query_embedding = await asyncio.gather(
            self.retriever.embedder.aembed_query(query_str),
            self.retriever.image_embedder.aembed_query(query_str),
        )
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
        # PHASE 2+3+4: HYBRID SEARCH + FUSION + RERANKING (shared pipeline)
        # ═══════════════════════════════════════════════════════════════
        yield {
            "type": "phase",
            "phase": "search",
            "status": "started",
            "message": "🔍 Searching dense, sparse & image vectors..."
        }
        
        # Delegate to the single shared retrieval pipeline
        result = await self.retriever._retrieve_hybrid(query_str, query_embedding, image_query_embedding)
        
        dense_nodes = result["dense_nodes"]
        sparse_nodes = result["sparse_nodes"]
        image_nodes = result["image_nodes"]
        search_time = result["timings"]["search_ms"]
        
        # Emit search results
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
                    "preview": n.node.metadata.get("caption", "")[:100] if n.node.metadata else "",
                    "score": n.score,
                    "type": "image",
                    "metadata": dict(n.node.metadata) if n.node.metadata else {}
                }
                for n in image_nodes[:5]
            ]
        }
        
        # Emit fusion phase
        fused_text = result["fused_text"]
        yield {
            "type": "phase",
            "phase": "fusion",
            "status": "completed",
            "message": f"✓ RRF fused → {len(fused_text)} unique text nodes"
        }
        
        # Emit reranking phase
        reranked_text = result["reranked_text"]
        reranked_images = result["reranked_images"]
        all_nodes = result["all_nodes"]
        rerank_time = result["timings"]["rerank_ms"]
        
        yield {
            "type": "phase",
            "phase": "reranking",
            "status": "completed",
            "message": f"✓ Reranked → {len(reranked_text)} text + {len(reranked_images)} images ({rerank_time:.0f}ms)",
            "duration_ms": rerank_time,
            "reranked": [
                {
                    "id": n.node.id_,
                    "score": n.score,
                    "type": "image" if "image_path" in (n.node.metadata or {}) else "text",
                    "preview": n.node.get_content()[:100] + "..."
                }
                for n in all_nodes[:7]
            ]
        }
        
        # ═══════════════════════════════════════════════════════════════
        # PHASE 5: LLM GENERATION
        # ═══════════════════════════════════════════════════════════════
        yield {
            "type": "phase",
            "phase": "generation",
            "status": "started",
            "message": "✨ Generating response..."
        }
        
        # Prepare context using shared helper
        text_context, image_paths = self._separate_nodes(all_nodes)
        messages = await self._construct_messages(query_str, text_context, image_paths)
        
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
            metadata = node.node.metadata if hasattr(node, 'node') else (node.metadata if hasattr(node, 'metadata') else {})
            node_type = "image" if "image_path" in metadata else "text"
            
            sources.append({
                "content": node.get_content()[:500] if hasattr(node, 'get_content') else node.node.get_content()[:500],
                "score": node.score or 0.0,
                "type": node_type,
                "metadata": dict(metadata) if metadata else {}
            })
        
        # Store in semantic cache
        await self._store_response_cache(query_embedding, query_str, full_response_text, all_nodes)
        
        total_time = (time.perf_counter() - start_time) * 1000
        
        yield {
            "type": "sources",
            "sources": sources,
            "total_duration_ms": total_time
        }
        
        yield {"type": "done", "total_duration_ms": total_time}


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
        q_manager.create_response_cache_collection(config.LOCAL_TEXT_DIMENSIONS)
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