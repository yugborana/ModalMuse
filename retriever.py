# retriever.py - Multi-Modal Hybrid Retriever with Jina Embeddings
"""
Orchestrates Multi-Modal Retrieval using Jina AI embeddings,
with Llama-3.2 Vision-based response generation using Groq.
"""

import asyncio
from typing import List, Union, Dict, Any, Optional

from qdrant_manager import QdrantManager

# LlamaIndex Core
from llama_index.core import VectorStoreIndex, QueryBundle
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, ImageNode, TextNode
from llama_index.core.vector_stores.types import VectorStoreQueryMode, VectorStoreQuery

from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.schema import ImageDocument
from llama_index.core.base.response.schema import Response
from qdrant_client import models

from jina_client import JinaEmbeddings, JinaReranker
from rrf_reranker import RRFReranker
import config
import base64
from groq import Groq


class MultiModalHybridRetriever(BaseRetriever):
    """
    Multi-Modal Retriever using Jina AI for embeddings and reranking.
    """
    
    def __init__(self, qdrant_manager: QdrantManager):
        super().__init__()
        self.qdrant_manager = qdrant_manager
        
        print("[LOADING] Initializing Jina Embeddings and Reranker...")
        
        # 1. Initialize Jina Embeddings (same model for text and images)
        self.embedder = JinaEmbeddings(dimensions=config.JINA_EMBED_DIMENSIONS)
        self.reranker = JinaReranker()
        
        # 2. Get Vector Stores from Qdrant
        self.text_store = qdrant_manager.get_text_vector_store()
        self.image_store = qdrant_manager.get_image_vector_store()
        
        # 3. Custom RRF Fusion for multi-modal results
        self.rrf_reranker = RRFReranker(
            k=config.RRF_K,
            weights=config.RRF_WEIGHTS
        )
        
        print("[OK] Jina Embeddings and Reranker Ready")

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Core Retrieval Logic using Jina Embeddings."""
        query_str = query_bundle.query_str
        
        print(f"[SEARCH] Retrieving for: '{query_str}'")

        # --- Step 1: Dense Text Retrieval via Jina ---
        # Generate query embedding using Jina API
        query_embedding = self.embedder.embed_query(query_str)
        
        # Search Qdrant with the embedding using query_points API (qdrant-client 1.10+)
        from qdrant_client import models
        text_results = self.qdrant_manager.client.query_points(
            collection_name=config.TEXT_COLLECTION_NAME,
            query=query_embedding,
            using="text-dense",
            limit=config.TEXT_SIMILARITY_TOP_K,
            with_payload=True
        )
        
        # Convert to NodeWithScore
        text_nodes = []
        for point in text_results.points:
            payload = point.payload or {}
            text_content = payload.get("text_chunk", "")
            node = TextNode(
                id_=str(point.id),
                text=text_content,
                metadata={k: v for k, v in payload.items() if k != "text_chunk"}
            )
            text_nodes.append(NodeWithScore(node=node, score=point.score))

        print(f"   [DATA] Text retrieval: {len(text_nodes)} nodes")
        
        # --- Step 2: Rerank with Jina ---
        if text_nodes:
            documents = [n.node.get_content() for n in text_nodes]
            rerank_results = self.reranker.rerank(
                query_str, 
                documents, 
                top_n=config.TEXT_RERANK_TOP_N
            )
            
            # Reorder nodes based on reranking
            reranked_nodes = []
            for result in rerank_results:
                idx = result["index"]
                score = result["relevance_score"]
                reranked_nodes.append(NodeWithScore(
                    node=text_nodes[idx].node, 
                    score=score
                ))
            
            print(f"   [RANK] After Jina reranking: {len(reranked_nodes)} nodes")
        else:
            reranked_nodes = text_nodes
        
        # --- Step 3: Image Retrieval via Jina ---
        # Generate query embedding (same Jina model works for images too)
        image_results = self.qdrant_manager.client.query_points(
            collection_name=config.IMAGE_COLLECTION_NAME,
            query=query_embedding,  # Same embedding for text-to-image
            limit=config.IMAGE_SIMILARITY_TOP_K,
            with_payload=True
        )
        
        image_nodes = []
        for point in image_results.points:
            payload = point.payload or {}
            # Prioritize image_url (Supabase/URLs) over image_path (local)
            img_path = payload.get("image_url") or payload.get("image_path", "")
            node = TextNode(
                id_=str(point.id),
                text=img_path,
                metadata={**payload, "image_path": img_path}
            )
            image_nodes.append(NodeWithScore(node=node, score=point.score))
        
        print(f"[OK] Found {len(reranked_nodes)} text chunks and {len(image_nodes)} images.")

        # --- Step 4: Final Multi-Modal RRF Fusion ---
        if image_nodes and config.MULTIMODAL_RRF_ENABLED:
            final_nodes = self.rrf_reranker.fuse_node_results(
                result_sets=[reranked_nodes, image_nodes],
                weights=config.MULTIMODAL_RRF_WEIGHTS,
                top_n=config.FINAL_TOP_K
            )
            print(f"   [MM] Multi-modal RRF fusion: {len(final_nodes)} final nodes")
            return final_nodes
        
        return reranked_nodes + image_nodes
    
    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Truly async retrieval with PARALLEL text + image search."""
        import time
        start_time = time.perf_counter()
        query_str = query_bundle.query_str
        
        print(f"[ASYNC SEARCH] Retrieving for: '{query_str}'")
        
        # --- Step 1: Get query embedding (required first) ---
        embed_start = time.perf_counter()
        query_embedding = await self.embedder.aembed_query(query_str)
        print(f"   [PERF] Embedding: {(time.perf_counter() - embed_start)*1000:.0f}ms")
        
        # --- Step 2: PARALLEL Text + Image Retrieval ---
        search_start = time.perf_counter()
        
        # Create both search tasks
        text_search_task = self.qdrant_manager.async_client.query_points(
            collection_name=config.TEXT_COLLECTION_NAME,
            query=query_embedding,
            using="text-dense",
            limit=config.TEXT_SIMILARITY_TOP_K,
            with_payload=True
        )
        
        image_search_task = self.qdrant_manager.async_client.query_points(
            collection_name=config.IMAGE_COLLECTION_NAME,
            query=query_embedding,
            limit=config.IMAGE_SIMILARITY_TOP_K,
            with_payload=True
        )
        
        # Execute both in parallel!
        text_results, image_results = await asyncio.gather(
            text_search_task, 
            image_search_task
        )
        print(f"   [PERF] Parallel search: {(time.perf_counter() - search_start)*1000:.0f}ms")
        
        # Convert text results to NodeWithScore
        text_nodes = []
        for point in text_results.points:
            payload = point.payload or {}
            text_content = payload.get("text_chunk", "")
            node = TextNode(
                id_=str(point.id),
                text=text_content,
                metadata={k: v for k, v in payload.items() if k != "text_chunk"}
            )
            text_nodes.append(NodeWithScore(node=node, score=point.score))
        
        print(f"   [DATA] Text retrieval: {len(text_nodes)} nodes")
        
        # --- Step 3: Async Rerank with Jina ---
        if text_nodes:
            rerank_start = time.perf_counter()
            documents = [n.node.get_content() for n in text_nodes]
            rerank_results = await self.reranker.arerank(
                query_str, 
                documents, 
                top_n=config.TEXT_RERANK_TOP_N
            )
            
            reranked_nodes = []
            for result in rerank_results:
                idx = result["index"]
                score = result["relevance_score"]
                reranked_nodes.append(NodeWithScore(
                    node=text_nodes[idx].node, 
                    score=score
                ))
            
            print(f"   [PERF] Reranking: {(time.perf_counter() - rerank_start)*1000:.0f}ms")
            print(f"   [RANK] After Jina reranking: {len(reranked_nodes)} nodes")
        else:
            reranked_nodes = text_nodes
        
        # Convert image results
        image_nodes = []
        for point in image_results.points:
            payload = point.payload or {}
            # Prioritize image_url (Supabase/URLs) over image_path (local)
            img_path = payload.get("image_url") or payload.get("image_path", "")
            node = TextNode(
                id_=str(point.id),
                text=img_path,
                metadata={**payload, "image_path": img_path}
            )
            image_nodes.append(NodeWithScore(node=node, score=point.score))
        
        total_time = (time.perf_counter() - start_time) * 1000
        print(f"[OK] Found {len(reranked_nodes)} text + {len(image_nodes)} images in {total_time:.0f}ms")
        
        # --- Step 4: Final Multi-Modal RRF Fusion ---
        if image_nodes and config.MULTIMODAL_RRF_ENABLED:
            final_nodes = self.rrf_reranker.fuse_node_results(
                result_sets=[reranked_nodes, image_nodes],
                weights=config.MULTIMODAL_RRF_WEIGHTS,
                top_n=config.FINAL_TOP_K
            )
            print(f"   [MM] Multi-modal RRF fusion: {len(final_nodes)} final nodes")
            return final_nodes
        
        return reranked_nodes + image_nodes


class GroqGenerator(CustomQueryEngine):
    """
    Query engine using Groq's Vision models (Llama 3.2).
    Replaces LlamaCppGenerator with Cloud API inference.
    """
    retriever: MultiModalHybridRetriever
    client: Any  # Groq client - use Any to avoid Pydantic validation issues
    aclient: Any  # AsyncGroq client
    model_name: str
    
    def __init__(
        self, 
        retriever: MultiModalHybridRetriever,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        """
        Initialize the generator with the retriever and Groq client.
        """
        api_key = api_key or config.GROQ_API_KEY
        if not api_key:
            raise ValueError("GROQ_API_KEY is not set.")
        
        # Create clients
        client = Groq(api_key=api_key)
        from groq import AsyncGroq
        aclient = AsyncGroq(api_key=api_key)
        resolved_model = model_name or config.GROQ_MODEL_NAME
        
        # Pass all fields to Pydantic parent constructor
        super().__init__(
            retriever=retriever,
            client=client,
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
        context_str = "\n\n".join(text_context)
        
        messages = [
            {
                "role": "system",
                "content": "You are an expert assistant. Answer using ONLY the provided context. Be EXTREMELY CONCISE: 1-2 short paragraphs maximum. No bullet points or lists unless asked. Get straight to the point."
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

    def custom_query(self, query_bundle: Union[str, QueryBundle]) -> Response:
        """Execute a query with retrieval and Groq generation."""
        if isinstance(query_bundle, str):
            query_bundle = QueryBundle(query_bundle)
        
        # 1. Retrieve
        nodes: List[NodeWithScore] = self.retriever.retrieve(query_bundle.query_str)
        
        # 2. Separate Text and Images
        text_context = []
        image_paths = []
        
        for node in nodes:
            if isinstance(node.node, ImageNode) or 'image_path' in node.metadata:
                img_path = node.metadata.get('image_path')
                if img_path:
                    image_paths.append(img_path)
            else:
                text_context.append(node.get_content())
        
        # 3. Construct Messages
        messages = self._construct_messages(query_bundle.query_str, text_context, image_paths)
        
        # 4. Generate Answer via Groq
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=config.LLM_TEMPERATURE,
                max_tokens=config.LLM_MAX_NEW_TOKENS,
                stream=False
            )
            response_text = completion.choices[0].message.content
        except Exception as e:
            response_text = f"Error generating response: {e}"
        
        return Response(response=response_text, source_nodes=nodes)
    
    async def acustom_query(self, query_bundle: Union[str, QueryBundle]) -> Response:
        """Async version of custom_query."""
        if isinstance(query_bundle, str):
            query_bundle = QueryBundle(query_bundle)
        
        # 1. Retrieve (Async)
        nodes: List[NodeWithScore] = await self.retriever._aretrieve(query_bundle)
        
        # 2. Separate Text and Images
        text_context = []
        image_paths = []
        
        for node in nodes:
            if isinstance(node.node, ImageNode) or 'image_path' in node.metadata:
                img_path = node.metadata.get('image_path')
                if img_path:
                    image_paths.append(img_path)
            else:
                text_context.append(node.get_content())
        
        # 3. Construct Messages
        messages = self._construct_messages(query_bundle.query_str, text_context, image_paths)
        
        # 4. Generate Answer (Async)
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
        # PHASE 1: EMBEDDING
        # ═══════════════════════════════════════════════════════════════
        yield {
            "type": "phase",
            "phase": "embedding",
            "status": "started",
            "message": "🔮 Creating query embedding..."
        }
        
        embed_start = time.perf_counter()
        query_embedding = await self.retriever.embedder.aembed_query(query_str)
        embed_time = (time.perf_counter() - embed_start) * 1000
        
        yield {
            "type": "phase",
            "phase": "embedding",
            "status": "completed",
            "message": f"✓ Embedding created ({embed_time:.0f}ms)",
            "duration_ms": embed_time
        }
        
        # ═══════════════════════════════════════════════════════════════
        # PHASE 2: VECTOR SEARCH
        # ═══════════════════════════════════════════════════════════════
        yield {
            "type": "phase",
            "phase": "search",
            "status": "started",
            "message": "🔍 Searching vector store..."
        }
        
        search_start = time.perf_counter()
        
        # Parallel search
        text_search_task = self.retriever.qdrant_manager.async_client.query_points(
            collection_name=config.TEXT_COLLECTION_NAME,
            query=query_embedding,
            using="text-dense",
            limit=config.TEXT_SIMILARITY_TOP_K,
            with_payload=True
        )
        
        image_search_task = self.retriever.qdrant_manager.async_client.query_points(
            collection_name=config.IMAGE_COLLECTION_NAME,
            query=query_embedding,
            limit=config.IMAGE_SIMILARITY_TOP_K,
            with_payload=True
        )
        
        text_results, image_results = await asyncio.gather(
            text_search_task, 
            image_search_task
        )
        search_time = (time.perf_counter() - search_start) * 1000
        
        # Convert to nodes
        text_nodes = []
        for point in text_results.points:
            payload = point.payload or {}
            text_content = payload.get("text_chunk", "")
            node = TextNode(
                id_=str(point.id),
                text=text_content,
                metadata={k: v for k, v in payload.items() if k != "text_chunk"}
            )
            text_nodes.append(NodeWithScore(node=node, score=point.score))
        
        image_nodes = []
        for point in image_results.points:
            payload = point.payload or {}
            # Prioritize image_url (Supabase/URLs) over image_path (local)
            img_path = payload.get("image_url") or payload.get("image_path", "")
            node = TextNode(
                id_=str(point.id),
                text=img_path,
                metadata={**payload, "image_path": img_path}
            )
            image_nodes.append(NodeWithScore(node=node, score=point.score))
        
        # Emit found chunks
        yield {
            "type": "chunks_found",
            "text_count": len(text_nodes),
            "image_count": len(image_nodes),
            "duration_ms": search_time,
            "message": f"📚 Found {len(text_nodes)} text chunks & {len(image_nodes)} images ({search_time:.0f}ms)",
            "chunks": [
                {
                    "id": n.node.id_,
                    "preview": n.node.get_content()[:150] + "..." if len(n.node.get_content()) > 150 else n.node.get_content(),
                    "score": n.score,
                    "type": "text",
                    "metadata": dict(n.node.metadata) if n.node.metadata else {}
                }
                for n in text_nodes[:5]  # Preview first 5
            ] + [
                {
                    "id": n.node.id_,
                    "preview": n.node.metadata.get("image_path", "")[-50:] if n.node.metadata else "",
                    "score": n.score,
                    "type": "image",
                    "metadata": dict(n.node.metadata) if n.node.metadata else {}
                }
                for n in image_nodes[:5]  # Preview first 5 images
            ]
        }
        
        # ═══════════════════════════════════════════════════════════════
        # PHASE 3: RERANKING
        # ═══════════════════════════════════════════════════════════════
        yield {
            "type": "phase",
            "phase": "reranking",
            "status": "started",
            "message": "⚖️ Reranking with Jina..."
        }
        
        rerank_start = time.perf_counter()
        
        if text_nodes:
            documents = [n.node.get_content() for n in text_nodes]
            rerank_results = await self.retriever.reranker.arerank(
                query_str, 
                documents, 
                top_n=config.TEXT_RERANK_TOP_N
            )
            
            reranked_nodes = []
            for result in rerank_results:
                idx = result["index"]
                score = result["relevance_score"]
                reranked_nodes.append(NodeWithScore(
                    node=text_nodes[idx].node, 
                    score=score
                ))
        else:
            reranked_nodes = text_nodes
            rerank_results = []
        
        rerank_time = (time.perf_counter() - rerank_start) * 1000
        
        yield {
            "type": "phase",
            "phase": "reranking",
            "status": "completed",
            "message": f"✓ Reranked to top {len(reranked_nodes)} ({rerank_time:.0f}ms)",
            "duration_ms": rerank_time,
            "reranked": [
                {
                    "id": n.node.id_,
                    "score": n.score,
                    "preview": n.node.get_content()[:100] + "..."
                }
                for n in reranked_nodes[:5]
            ]
        }
        
        # ═══════════════════════════════════════════════════════════════
        # PHASE 4: RRF FUSION
        # ═══════════════════════════════════════════════════════════════
        if image_nodes and config.MULTIMODAL_RRF_ENABLED:
            yield {
                "type": "phase",
                "phase": "fusion",
                "status": "started",
                "message": "🔀 Multi-modal RRF fusion..."
            }
            
            final_nodes = self.retriever.rrf_reranker.fuse_node_results(
                result_sets=[reranked_nodes, image_nodes],
                weights=config.MULTIMODAL_RRF_WEIGHTS,
                top_n=config.FINAL_TOP_K
            )
            
            yield {
                "type": "phase",
                "phase": "fusion",
                "status": "completed",
                "message": f"✓ Fused {len(final_nodes)} final results"
            }
        else:
            final_nodes = reranked_nodes + image_nodes
        
        all_nodes = final_nodes
        
        # ═══════════════════════════════════════════════════════════════
        # PHASE 5: LLM GENERATION
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
            
            async for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
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
        # FINAL: SOURCES
        # ═══════════════════════════════════════════════════════════════
        sources = []
        for node in all_nodes:
            # Access metadata from the inner node (NodeWithScore wraps the actual node)
            metadata = node.node.metadata if hasattr(node, 'node') else (node.metadata if hasattr(node, 'metadata') else {})
            node_type = "image" if "image_path" in metadata else "text"
            
            # Use original reranker score if available (RRF fusion stores it in metadata)
            # RRF scores are very small (0.01 range) while reranker scores are more meaningful (0.4-0.9)
            score = metadata.get("original_score") or node.score or 0.0
            
            sources.append({
                "content": node.get_content()[:500] if hasattr(node, 'get_content') else node.node.get_content()[:500],
                "score": score,
                "type": node_type,
                "metadata": dict(metadata) if metadata else {}
            })
        
        total_time = (time.perf_counter() - start_time) * 1000
        
        yield {
            "type": "sources",
            "sources": sources,
            "total_duration_ms": total_time
        }
        
        yield {"type": "done", "total_duration_ms": total_time}
    
    async def astream_query(self, query_bundle: Union[str, QueryBundle]):
        """
        Async streaming version of query with live status updates.
        
        Yields tuples of (event_type, data, nodes):
        - ("status", "Searching...", None) - Status update
        - ("chunk", "text", nodes) - LLM response chunk
        """
        if isinstance(query_bundle, str):
            query_bundle = QueryBundle(query_bundle)
        
        # === Status: Searching ===
        yield ("status", "🔍 Searching documents...", None)
        
        # 1. Retrieve
        nodes: List[NodeWithScore] = await self.retriever._aretrieve(query_bundle)
        
        # === Status: Found nodes ===
        text_count = sum(1 for n in nodes if 'image_path' not in n.metadata)
        img_count = len(nodes) - text_count
        yield ("status", f"📚 Found {text_count} text & {img_count} image sources", None)
        
        # 2. Prepare context
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
                
        # 3. Construct Messages
        messages = self._construct_messages(query_bundle.query_str, text_context, image_paths)
        
        # 4. Stream response
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
                    yield ("chunk", content, nodes)
        except Exception as e:
            yield ("chunk", f"Error: {e}", nodes)


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
    
    retriever = MultiModalHybridRetriever(q_manager)
    engine = GroqGenerator(retriever)
    
    print("[OK] Query Engine Ready!")
    return engine


if __name__ == "__main__":
    # Test the pipeline
    print("[*] Testing Multimodal Pipeline...")
    
    try:
        engine = create_query_engine()
        
        user_query = "Summarize the document and explain any diagrams found."
        
        print(f"\n[?] Asking: {user_query}")
        final_answer = engine.custom_query(user_query)
        
        print("\n" + "=" * 50)
        print("[ANSWER] FINAL ANSWER:")
        print("=" * 50)
        print(final_answer)
        print("=" * 50)
        
    except Exception as e:
        print(f"[ERROR] Error during generation: {e}")
        import traceback
        traceback.print_exc()