# indexer.py

import os
import uuid
import json
import hashlib
import numpy as np
import httpx
import base64
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

# External Libraries
from qdrant_manager import QdrantManager, TextPoint, ImagePoint
from jina_client import JinaEmbeddings
from bm25 import BM25SparseEncoder

# LlamaIndex Components
from llama_parse import LlamaParse
from llama_index.core.schema import Document
from llama_index.core.node_parser import SentenceSplitter

# Import configuration
import config

class Indexer:
    """Handles document parsing, multi-modal embedding generation, and Qdrant indexing."""
    
    def __init__(self, qdrant_manager: QdrantManager, llama_parse_api_key: str):
        self.qdrant_manager = qdrant_manager
        
        # 1. Initialize Jina Embeddings (for both text and images)
        print("[LOADING] Initializing Jina Embeddings...")
        self.embedder = JinaEmbeddings(dimensions=config.JINA_EMBED_DIMENSIONS)
        print("[OK] Jina Embeddings initialized")
        
        # 2. Initialize BM25 Sparse Encoder (local, no API)
        self.sparse_encoder = BM25SparseEncoder()
        print("[OK] BM25 Sparse Encoder initialized")
        
        # 3. Initialize Groq client once (reused for all image captions)
        self._groq_client = None
        if config.GROQ_API_KEY:
            try:
                from groq import Groq
                self._groq_client = Groq(api_key=config.GROQ_API_KEY)
                print("[OK] Groq Vision client initialized (for image captioning)")
            except ImportError:
                print("[WARN] Groq not installed — image captioning disabled")
        
        # Initialize LlamaParse (API required for parsing)
        self.parser = LlamaParse(api_key=llama_parse_api_key, result_type="markdown", verbose=True)
        
        # LlamaIndex Text Splitter
        self.text_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=200)
        
        # --- Pre-calculate Dims for Qdrant Setup ---
        self.dense_dim = self.embedder.embed_dim  # From Jina config (1024)
        self.image_dim = self.embedder.embed_dim  # Same model for images!
        
        # Initialize Qdrant Collections
        self.qdrant_manager.create_text_collection(
            dense_dim=self.dense_dim
        )
        self.qdrant_manager.create_image_collection(
            image_dim=self.image_dim
        )
        
        # --- Supabase Parse Cache ---
        try:
            from supabase_client import get_parse_cache, save_parse_cache
            self._supabase_cache_available = True
            print("[OK] Supabase parse cache enabled")
        except ImportError:
            self._supabase_cache_available = False
            print("[WARN] Supabase not available - parse cache disabled")
    
    # --- Cache Helper Methods ---
    
    def _get_file_hash(self, file_path: str) -> str:
        """Compute SHA256 hash of file contents for cache key."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def _load_from_cache(self, file_path: str) -> Optional[Tuple[List, List, str]]:
        """Load parsed result from Supabase cache.
        
        Returns:
            Tuple of (json_objs, images_data, file_hash) if cached, None otherwise
        """
        file_hash = self._get_file_hash(file_path)
        
        if not self._supabase_cache_available:
            return None
        
        try:
            from supabase_client import get_parse_cache
            cached = get_parse_cache(file_hash)
            if cached and cached.get("parsed_json"):
                print(f"[CACHE] Supabase cache hit for {Path(file_path).name}")
                return cached["parsed_json"], cached.get("images_data", []), file_hash
        except Exception as e:
            print(f"[WARN] Supabase cache read failed: {e}")
        
        return None
    
    def _save_to_cache(self, file_hash: str, json_objs: List, images_data: List, file_name: str = "") -> None:
        """Save parsed result to Supabase cache."""
        if not self._supabase_cache_available:
            return
        
        # Extract job_id if available
        job_id = None
        if json_objs and isinstance(json_objs, list) and len(json_objs) > 0:
            job_id = json_objs[0].get("job_id")
        
        try:
            from supabase_client import save_parse_cache
            save_parse_cache(
                file_hash=file_hash,
                file_name=file_name or f"{file_hash[:8]}.pdf",
                parsed_json=json_objs,
                images_data=images_data,
                job_id=job_id
            )
        except Exception as e:
            print(f"[WARN] Supabase cache save failed: {e}")
    
    def _caption_image(self, image_source: str) -> str:
        """Generate a text caption/summary for an image using Groq Vision.
        
        Args:
            image_source: Local file path or HTTP URL to the image.
            
        Returns:
            Caption string, or empty string if captioning fails.
        """
        import io
        import httpx
        from groq import Groq
        
        try:
            is_url = image_source.startswith("http://") or image_source.startswith("https://")
            
            if is_url:
                # Download image from URL
                with httpx.Client(timeout=30.0) as client:
                    response = client.get(image_source)
                    response.raise_for_status()
                    image_data = response.content
            else:
                # Read local file
                with open(image_source, "rb") as f:
                    image_data = f.read()
            
            # Resize if needed (Groq 33M pixel limit)
            from PIL import Image
            img = Image.open(io.BytesIO(image_data))
            width, height = img.size
            total_pixels = width * height
            max_pixels = 30_000_000
            
            if total_pixels > max_pixels:
                scale = (max_pixels / total_pixels) ** 0.5
                new_w, new_h = int(width * scale), int(height * scale)
                img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            
            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')
            
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=85)
            buffer.seek(0)
            b64_image = base64.b64encode(buffer.read()).decode('utf-8')
            
            # Call Groq Vision for captioning
            groq_client = Groq(api_key=config.GROQ_API_KEY)
            
            response = groq_client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Describe this image in 1-2 sentences. Focus on what the image shows — diagrams, charts, text, formulas, or visual content. Be specific and factual."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{b64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=150,
                temperature=0.3
            )
            
            caption = response.choices[0].message.content.strip()
            return caption
            
        except Exception as e:
            print(f"   [WARN] Caption generation failed for {image_source[:50]}: {e}")
            return ""
    
    def _download_images_from_json(self, json_objs: List, download_dir: Path) -> List[Dict]:
        """
        Download images from LlamaCloud API and upload to Supabase storage.
        
        Constructs URL: https://api.cloud.llamaindex.ai/api/parsing/job/{job_id}/result/image/{image_name}
        Then uploads to Supabase storage and returns public URLs.
        """
        # Check if Supabase storage is enabled
        use_supabase = config.SUPABASE_STORAGE_ENABLED
        
        # Import Supabase storage functions if enabled
        supabase_available = False
        if use_supabase:
            try:
                from supabase_client import upload_image_from_response, get_image_public_url
                supabase_available = True
                print("[OK] Using Supabase storage for images")
            except ImportError:
                print("[ERROR] Supabase storage not available - images will not be stored")
        else:
            print("[INFO] Supabase storage disabled")
        
        images_data = []
        base_url = "https://api.cloud.llamaindex.ai/api/parsing"
        headers = {"Authorization": f"Bearer {config.LLAMA_PARSE_API_KEY}"}
        
        print(f"[SEARCH] Inspecting {len(json_objs)} objects for images...")
        
        # Extract all image URLs from the JSON
        for i, doc in enumerate(json_objs):
            job_id = doc.get("job_id")
            if not job_id:
                print(f"[WARN] Doc {i} has no job_id, cannot download images.")
                continue
                
            pages = doc.get("pages", [])
            print(f"   Doc {i} (Job {job_id}) has {len(pages)} pages.")
            
            for page_idx, page in enumerate(pages):
                page_images = page.get("images", [])
                
                for img_idx, img in enumerate(page_images):
                    image_name = img.get("name")
                    if not image_name:
                        continue
                    
                    # Construct API URL
                    url = f"{base_url}/job/{job_id}/result/image/{image_name}"
                    
                    # Generate filename for storage
                    storage_name = f"{job_id}_{image_name}"
                    
                    # Only proceed if Supabase is available
                    if not supabase_available:
                        continue
                    
                    try:
                        # Download with httpx 
                        with httpx.Client(timeout=30.0) as client:
                            response = client.get(url, headers=headers)
                            if response.status_code == 200:
                                image_bytes = response.content
                                
                                # Upload to Supabase storage
                                public_url = upload_image_from_response(image_bytes, storage_name)
                                if public_url:
                                    images_data.append({
                                        "path": public_url,  # Store Supabase URL
                                        "url": public_url,
                                        "name": storage_name,
                                        "page": page_idx,
                                        "job_id": job_id,
                                        "storage": "supabase"
                                    })
                                    # Print only every 10th image to avoid spam
                                    if len(images_data) % 10 == 1:
                                        print(f"   [OK] Uploaded to Supabase: {storage_name}")
                            else:
                                print(f"   [WARN] Failed to download {image_name}: HTTP {response.status_code}")
                    except Exception as e:
                        print(f"   [WARN] Failed to download {image_name}: {e}")
        
        return images_data

    def _caption_image(self, image_source: str) -> str:
        """Generate a caption for an image using Groq Vision model.
        
        Args:
            image_source: URL or local file path to the image
            
        Returns:
            Caption string, or empty string if captioning fails
        """
        if not self._groq_client:
            return ""
        
        import io
        from PIL import Image
        
        try:
            # Load image
            if image_source.startswith('http://') or image_source.startswith('https://'):
                with httpx.Client(timeout=30.0) as client:
                    resp = client.get(image_source)
                    resp.raise_for_status()
                    img_data = io.BytesIO(resp.content)
            else:
                img_data = open(image_source, 'rb')
            
            img = Image.open(img_data)
            
            # Resize if too large (Groq limit: 33M pixels)
            width, height = img.size
            if width * height > 30_000_000:
                scale = (30_000_000 / (width * height)) ** 0.5
                img = img.resize((int(width * scale), int(height * scale)), Image.Resampling.LANCZOS)
            
            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')
            
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=85)
            buffer.seek(0)
            base64_image = base64.b64encode(buffer.read()).decode('utf-8')
            
            # Free PIL image immediately
            img.close()
            del img
            
            completion = self._groq_client.chat.completions.create(
                model=config.GROQ_MODEL_NAME,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image in 2-3 sentences. Focus on what it depicts, any text/labels visible, and its purpose in a technical/academic context."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }],
                max_tokens=150,
                temperature=0.3
            )
            
            # Free base64 string immediately
            del base64_image
            
            return completion.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"   [WARN] Image captioning failed: {e}")
            return ""

    # --- Core Indexing Function ---

    def index_document(self, file_path: str) -> dict:
        """
        Processes the PDF, generates embeddings, and upserts to Qdrant.
        
        Flow:
        1. Check Supabase parse_cache for existing cached result
        2. If cached: verify images exist (re-upload if missing)
        3. If not cached: parse with LlamaParse, upload images, save to cache
        
        Returns:
            dict: Contains 'text_count' and 'image_count' of indexed vectors
        """
        result = {"text_count": 0, "image_count": 0, "from_cache": False}
        
        print(f"[DOC] Processing document: {file_path}")
        
        # Free any lingering garbage before heavy allocation
        import gc; gc.collect()
        
        # Ensure collections exist (in case they were deleted)
        self.qdrant_manager.create_text_collection(dense_dim=self.dense_dim)
        self.qdrant_manager.create_image_collection(image_dim=self.image_dim)
        
        # --- Check Supabase Cache First ---
        try:
            cached = self._load_from_cache(file_path)
        except MemoryError:
            print("[WARN] OOM loading parse cache — will re-parse instead")
            cached = None
        except Exception as e:
            print(f"[WARN] Cache load failed: {e}")
            cached = None
        
        if cached:
            json_objs, images_data, file_hash = cached
            print(f"[CACHE] Found cached parse result in Supabase!")
            
            # Check if images need to be populated
            needs_image_upload = False
            
            if len(images_data) == 0:
                print(f"[INFO] Cache has no images, will populate...")
                needs_image_upload = True
            else:
                # Check if images are stored in Supabase (not LlamaParse temp URLs or local paths)
                first_img = images_data[0] if images_data else {}
                img_path = first_img.get("path", "") or first_img.get("url", "")
                storage_type = first_img.get("storage", "")
                
                # Check if it's a Supabase URL or has storage="supabase" marker
                is_supabase_url = (
                    "supabase" in img_path.lower() or 
                    storage_type == "supabase" or
                    ".supabase.co" in img_path
                )
                is_llamaparse_url = "api.cloud.llamaindex.ai" in img_path
                is_local_path = not img_path.startswith("http")
                
                if is_supabase_url:
                    print(f"[OK] Images already in Supabase storage ({len(images_data)} images)")
                elif is_llamaparse_url:
                    print(f"[INFO] Cached images are LlamaParse temp URLs, re-uploading to Supabase...")
                    needs_image_upload = True
                elif is_local_path:
                    print(f"[INFO] Cached images are local paths, re-uploading to Supabase...")
                    needs_image_upload = True
                else:
                    print(f"[INFO] Unknown image storage, re-uploading to Supabase...")
            
            # Re-upload images if needed (uses job_id from cached json_objs)
            if needs_image_upload:
                # Get job_id from cached parse result
                job_id = json_objs[0].get("job_id") if json_objs else None
                print(f"[INFO] Using cached job_id: {job_id}")
                
                if config.URL_BASED_IMAGE_INDEXING:
                    print(f"[URL] Extracting image URLs from LlamaParse...")
                    images_data = self._extract_image_urls(json_objs)
                else:
                    print(f"[UPLOAD] Downloading images from LlamaParse (job_id: {job_id}) → Supabase...")
                    images_data = self._download_images_from_json(json_objs, None)
                    print(f"[OK] Uploaded {len(images_data)} images to Supabase.")
                
                # Update cache with new image URLs
                if len(images_data) > 0:
                    self._save_to_cache(file_hash, json_objs, images_data, file_name=Path(file_path).name)
                    print(f"[CACHE] Updated cache with {len(images_data)} image URLs")
            
            result["from_cache"] = True
        else:
            # Not cached - Parse with LlamaParse API
            print(f"[PARSE] No cache found, parsing with LlamaParse API...")
            file_hash = self._get_file_hash(file_path)
            json_objs = self.parser.get_json_result(file_path)
            
            if isinstance(json_objs, list):
                json_objs = json_objs
            else:
                json_objs = [json_objs]
            
            # Handle images based on config
            if config.URL_BASED_IMAGE_INDEXING:
                print(f"[URL] Extracting image URLs for indexing (skipping download)...")
                images_data = self._extract_image_urls(json_objs)
            else:
                # Upload images to Supabase storage
                print(f"[UPLOAD] Uploading images to Supabase storage...")
                images_data = self._download_images_from_json(json_objs, None)
                print(f"[OK] Uploaded {len(images_data)} images to Supabase.")
            
            # Save to Supabase cache
            self._save_to_cache(file_hash, json_objs, images_data, file_name=Path(file_path).name)
            print(f"[CACHE] Saved parse result to Supabase cache")

        pages = []
        if isinstance(json_objs, list) and len(json_objs) > 0:
             # Sometimes it returns a list of docs, check the first one
            if "pages" in json_objs[0]:
                pages = json_objs[0]["pages"]
            else:
                pages = json_objs
        elif isinstance(json_objs, dict) and "pages" in json_objs:
             pages = json_objs["pages"]
        
        # FREE the massive JSON objects — we only need pages from here on
        del json_objs
        import gc; gc.collect()
        
        if not pages:
            print("[ERROR] Parsing failed: No pages found in API response.")
            return result

        print(f"[OK] Parsed {len(pages)} pages.")
        
        # Step 1: Create per-page Documents (preserves page numbers in metadata)
        file_name = Path(file_path).name
        docs = []
        for page_idx, page in enumerate(pages):
            page_text = page.get("text", "")
            if page_text.strip():
                docs.append(Document(
                    text=page_text,
                    metadata={"source": file_path, "page": page_idx + 1, "file_name": file_name}
                ))
        
        if not docs:
            print("[ERROR] Error: Document text is empty!")
            return result
        
        # Split into chunks — each chunk inherits page metadata from its Document
        nodes = self.text_splitter.get_nodes_from_documents(docs)
        del docs  # Free page text — nodes have their own copies
        
        print(f"Split document into {len(nodes)} text nodes (with page metadata).")
        
        # Step 2: Pre-filter — skip text chunks already in Qdrant
        NAMESPACE_UUID = uuid.UUID("00000000-0000-0000-0000-000000000000")
        
        # Compute deterministic IDs for ALL chunks
        chunk_id_map = {}  # point_id -> (node_index, node)
        for i, node in enumerate(nodes):
            content_hash = hashlib.md5(node.get_content().encode()).hexdigest()
            point_id = str(uuid.uuid5(NAMESPACE_UUID, content_hash))
            chunk_id_map[point_id] = (i, node)
        
        # Check Qdrant for already-indexed chunks
        all_candidate_ids = list(chunk_id_map.keys())
        existing_text_ids = set(self.qdrant_manager.get_existing_text_ids(all_candidate_ids))
        
        # Filter to only NEW chunks
        new_chunks = [(pid, idx, node) for pid, (idx, node) in chunk_id_map.items() if pid not in existing_text_ids]
        
        print(f"[TEXT] {len(nodes)} total chunks, {len(existing_text_ids)} already in Qdrant, {len(new_chunks)} need embedding")
        
        if not new_chunks:
            print(f"[OK] All {len(nodes)} text points already exist. Skipping embedding & upsert.")
            result["text_count"] = 0
        else:
            # Step 3: Embed NEW text chunks in batches to limit peak memory
            new_text_contents = [node.get_content() for _, _, node in new_chunks]
            EMBED_BATCH = 50  # Process 50 chunks at a time
            all_dense = []
            for batch_start in range(0, len(new_text_contents), EMBED_BATCH):
                batch = new_text_contents[batch_start:batch_start + EMBED_BATCH]
                batch_num = batch_start // EMBED_BATCH + 1
                total_batches = (len(new_text_contents) + EMBED_BATCH - 1) // EMBED_BATCH
                print(f"[EMBED] Batch {batch_num}/{total_batches}: embedding {len(batch)} chunks via Jina v4...")
                batch_vecs = self.embedder.embed_texts(batch, task="retrieval.passage")
                all_dense.extend(batch_vecs)
                del batch_vecs  # Free batch immediately
            dense_vectors = all_dense
            print(f"[OK] Generated {len(dense_vectors)} dense embeddings")
            
            # Step 3b: Generate BM25 sparse vectors (local, instant)
            print(f"[SPARSE] Generating BM25 sparse vectors for {len(new_text_contents)} chunks...")
            sparse_vectors = self.sparse_encoder.encode_documents(new_text_contents)
            print(f"[OK] Generated {len(sparse_vectors)} sparse vectors")
            
            # Free raw text contents — no longer needed
            del new_text_contents
            
            # Step 4: Package and upsert (dense + sparse)
            text_points: List[TextPoint] = []
            for (point_id, idx, node), dense_vec, sparse_vec in zip(new_chunks, dense_vectors, sparse_vectors):
                text_points.append(
                    TextPoint(
                        id=point_id,
                        text_chunk=node.get_content(),
                        dense_vector=dense_vec,
                        sparse_indices=sparse_vec["indices"],
                        sparse_values=sparse_vec["values"],
                        payload={"source": file_path, **node.metadata}
                    )
                )
            
            # Free vectors before upsert (upsert copies data)
            del dense_vectors, sparse_vectors, new_chunks
            
            self.qdrant_manager.upsert_text_points(text_points)
            print(f"[OK] Upserted {len(text_points)} NEW text points (skipped {len(existing_text_ids)} existing).")
            result["text_count"] = len(text_points)
            del text_points  # Free immediately
        
        # Free nodes and pages to reclaim memory before image processing
        del nodes
        import gc; gc.collect()
        
        # Step 5: Handle Images (Multi-Modal)
        # Note: json_objs was already freed above, so URL-based indexing
        # uses images_data which was extracted earlier
        image_count = self._index_downloaded_images(images_data, source_file=file_path)
        
        result["image_count"] = image_count
        
        # Final memory cleanup
        del pages, images_data
        gc.collect()
        
        return result
    
    def _extract_image_urls(self, json_objs: List) -> List[Dict]:
        """
        Extract image URLs from LlamaParse JSON without downloading.
        
        Returns list of dicts with: url, name, page, job_id
        """
        images_data = []
        base_url = "https://api.cloud.llamaindex.ai/api/parsing"
        
        for doc in json_objs:
            job_id = doc.get("job_id")
            if not job_id:
                continue
                
            pages = doc.get("pages", [])
            for page_idx, page in enumerate(pages):
                page_images = page.get("images", [])
                
                for img in page_images:
                    image_name = img.get("name")
                    if not image_name:
                        continue
                    
                    # Construct S3 URL (public access)
                    url = f"{base_url}/job/{job_id}/result/image/{image_name}"
                    
                    images_data.append({
                        "url": url,
                        "name": f"{job_id}_{image_name}",
                        "page": page_idx,
                        "job_id": job_id
                    })
        
        return images_data
    
    def _index_images_from_urls(self, json_objs: List, source_file: str) -> int:
        """
        Embeds images directly from URLs using Jina API (no local download).
        
        Returns:
            int: Number of images successfully indexed
        """
        images_data = self._extract_image_urls(json_objs)
        
        if not images_data:
            print("[INFO] No images found in document.")
            return 0
        
        print(f"[EMBED] Embedding {len(images_data)} images from URLs via Jina API...")
        
        # Extract URLs for embedding
        image_urls = [img["url"] for img in images_data]
        
        try:
            # Jina supports URL-based image embedding directly
            image_vectors = self.embedder.embed_images(image_urls)
        except Exception as e:
            print(f"[ERROR] Failed to embed images from URLs: {e}")
            return 0
        
        print(f"[OK] Generated {len(image_vectors)} image embeddings from URLs")
        
        # Create image points
        image_points: List[ImagePoint] = []
        NAMESPACE_UUID = uuid.UUID("00000000-0000-0000-0000-000000000000")
        
        for img_info, image_vector in zip(images_data, image_vectors):
            # Deterministic ID based on URL
            url_hash = hashlib.md5(img_info["url"].encode()).hexdigest()
            point_id = str(uuid.uuid5(NAMESPACE_UUID, url_hash))
            
            image_points.append(
                ImagePoint(
                    id=point_id,
                    image_path=img_info["url"],
                    image_vector=image_vector,
                    payload={
                        "source": source_file,
                        "original_name": img_info.get("name", "unknown"),
                        "image_url": img_info["url"],
                        "type": "image",
                        "page": img_info.get("page", 0) + 1,
                        "file_name": Path(source_file).name,
                        "caption": "",
                        "indexed_from": "url"
                    }
                )
            )
        
        # Check which image points already exist and only upsert new ones
        if image_points:
            all_image_ids = [p.id for p in image_points]
            existing_image_ids = self.qdrant_manager.get_existing_image_ids(all_image_ids)
            
            new_image_points = [p for p in image_points if p.id not in existing_image_ids]
            
            if new_image_points:
                # Caption new images using Groq Vision + embed captions for text→image search
                import time as time_mod
                print(f"[CAPTION] Generating captions for {len(new_image_points)} images via Groq...")
                captions_to_embed = []
                for i, point in enumerate(new_image_points):
                    img_url = point.payload.get("image_url", "")
                    caption = self._caption_image(img_url)
                    point.payload["caption"] = caption
                    captions_to_embed.append(caption if caption else "image")
                    if caption:
                        print(f"   [{i+1}/{len(new_image_points)}] \"{caption[:60]}...\"")
                    if i < len(new_image_points) - 1:
                        time_mod.sleep(2)
                
                # Embed captions as text vectors for text→image retrieval
                print(f"[EMBED] Embedding {len(captions_to_embed)} captions for text→image search...")
                try:
                    caption_vectors = self.embedder.embed_texts(captions_to_embed, task="retrieval.passage")
                    for point, cap_vec in zip(new_image_points, caption_vectors):
                        point.caption_vector = cap_vec
                except Exception as e:
                    print(f"   [WARN] Caption embedding failed: {e}")
                
                self.qdrant_manager.upsert_image_points(new_image_points)
                print(f"[OK] Upserted {len(new_image_points)} NEW URL-indexed image points (skipped {len(existing_image_ids)} existing).")
            else:
                print(f"[OK] All {len(image_points)} image points already exist. Skipping upsert.")
            
            return len(new_image_points)
        
        return 0
        
    def _index_downloaded_images(self, images_data: List[Dict], source_file: str) -> int:
        """
        Embeds images using Jina API.
        Supports both Supabase storage URLs and local file paths.
        
        Returns:
            int: Number of images successfully indexed
        """
        if not images_data:
            print("[INFO] No images to index.")
            return 0

        # Separate images by storage type
        url_images = []
        local_images = []
        
        for img_info in images_data:
            storage_type = img_info.get("storage", "")
            image_path = img_info.get("path") or img_info.get("url", "")
            
            # Check if it's a URL (Supabase or any HTTP URL)
            is_url = image_path.startswith("http://") or image_path.startswith("https://")
            
            if is_url:
                # URL images (Supabase, LlamaParse, etc.)
                url_images.append(img_info)
                print(f"   [URL] {img_info.get('name', 'unknown')}")
            elif image_path and Path(image_path).exists():
                # Only add local images that exist
                local_images.append(img_info)
                print(f"   [LOCAL] {img_info.get('name', 'unknown')}")
            else:
                print(f"   [SKIP] Invalid image: {image_path[:50]}...")
        
        valid_images = url_images + local_images
        
        if not valid_images:
            print("[WARN] No valid images found (checked URLs and local paths).")
            return 0
        
        # ── Pre-filter: Skip images that already exist in Qdrant ──
        NAMESPACE_UUID = uuid.UUID("00000000-0000-0000-0000-000000000000")
        
        # Compute deterministic IDs for ALL valid images (same logic used later for upserting)
        image_id_map = {}  # point_id -> img_info
        for img_info in valid_images:
            name_for_hash = img_info.get("name", img_info.get("path") or img_info.get("url", ""))
            path_hash = hashlib.md5(name_for_hash.encode()).hexdigest()
            point_id = str(uuid.uuid5(NAMESPACE_UUID, path_hash))
            image_id_map[point_id] = img_info
        
        # Check Qdrant for already-indexed images
        all_candidate_ids = list(image_id_map.keys())
        existing_image_ids = set(self.qdrant_manager.get_existing_image_ids(all_candidate_ids))
        
        # Filter to only NEW images (not yet in Qdrant)
        new_images = [(pid, img_info) for pid, img_info in image_id_map.items() if pid not in existing_image_ids]
        
        print(f"[EMBED] {len(valid_images)} total images, {len(existing_image_ids)} already in Qdrant, {len(new_images)} need embedding")
        print(f"   - {len(url_images)} from URLs")
        print(f"   - {len(local_images)} from local files")
        
        if not new_images:
            print(f"[OK] All {len(valid_images)} image points already exist. Skipping embedding & upsert.")
            return 0
        
        # Process images in batches: embed → caption → upsert per batch
        # This ensures partial progress is saved — if we crash at batch 3,
        # batches 1-2 are already in Qdrant and will be skipped on re-run.
        BATCH_SIZE = config.JINA_IMAGE_BATCH_SIZE
        BATCH_DELAY = config.JINA_BATCH_DELAY_SECONDS
        total_upserted = 0
        total_batches = (len(new_images) + BATCH_SIZE - 1) // BATCH_SIZE
        
        import time as time_mod
        
        for batch_idx in range(0, len(new_images), BATCH_SIZE):
            batch_items = new_images[batch_idx:batch_idx + BATCH_SIZE]
            batch_num = batch_idx // BATCH_SIZE + 1
            
            batch_sources = [
                img_info.get("path") or img_info.get("url")
                for _, img_info in batch_items
            ]
            
            # Rate-limit delay between batches
            if batch_idx > 0 and BATCH_DELAY > 0:
                print(f"   [WAIT] Sleeping {BATCH_DELAY}s to respect Jina rate limit...")
                time_mod.sleep(BATCH_DELAY)
            
            # ── 1. Embed this batch ──
            print(f"   [BATCH {batch_num}/{total_batches}] Embedding {len(batch_sources)} images...")
            try:
                batch_vectors = self.embedder.embed_images(batch_sources)
            except Exception as e:
                print(f"   [ERROR] Batch {batch_num} embedding failed: {e}")
                print(f"   [SAVE] {total_upserted} images already saved to Qdrant from previous batches.")
                continue  # Skip this batch, move to next
            
            print(f"   [OK] Batch {batch_num} embedded")
            
            # ── 2. Caption + build points for this batch ──
            batch_points: List[ImagePoint] = []
            
            for (point_id, img_info), vec in zip(batch_items, batch_vectors):
                if vec is None:
                    continue
                
                image_path = img_info.get("path") or img_info.get("url", "")
                storage_type = img_info.get("storage", "")
                if image_path.startswith("http"):
                    storage_type = "url"
                
                # Caption this image
                caption = self._caption_image(image_path)
                if caption:
                    print(f"   [CAPTION] \"{caption[:60]}...\"")
                
                # Embed the caption as text vector for text→image search
                caption_vec = []
                if caption:
                    try:
                        caption_vecs = self.embedder.embed_texts([caption], task="retrieval.passage")
                        caption_vec = caption_vecs[0] if caption_vecs else []
                    except Exception as e:
                        print(f"   [WARN] Caption embedding failed: {e}")
                
                # Brief delay between API calls
                time_mod.sleep(1)
                
                batch_points.append(
                    ImagePoint(
                        id=point_id,
                        image_path=image_path,
                        image_vector=vec,
                        caption_vector=caption_vec,
                        payload={
                            "source": source_file,
                            "original_name": img_info.get("name", "unknown"),
                            "type": "image",
                            "page": img_info.get("page", 0) + 1,
                            "file_name": Path(source_file).name,
                            "caption": caption,
                            "storage": storage_type,
                            "image_url": image_path if image_path.startswith("http") else None,
                            "image_path": image_path
                        }
                    )
                )
            
            # ── 3. Upsert this batch immediately ──
            if batch_points:
                self.qdrant_manager.upsert_image_points(batch_points)
                total_upserted += len(batch_points)
                print(f"   [OK] Batch {batch_num}/{total_batches} upserted → {len(batch_points)} points ({total_upserted} total so far)")
        
        skipped = len(existing_image_ids)
        print(f"[OK] Image indexing complete: {total_upserted} NEW upserted, {skipped} skipped (already existed).")
        return total_upserted