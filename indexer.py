# indexer.py

import os
import uuid
import json
import hashlib
import numpy as np
import httpx
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

# External Libraries
from qdrant_manager import QdrantManager, HybridPoint, ImagePoint
from jina_client import JinaEmbeddings

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
        
        # Initialize LlamaParse (API required for parsing)
        self.parser = LlamaParse(api_key=llama_parse_api_key, result_type="markdown", verbose=True)
        
        # LlamaIndex Text Splitter
        self.text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=20)
        
        # --- Pre-calculate Dims for Qdrant Setup ---
        self.dense_dim = self.embedder.embed_dim  # From Jina config (1024)
        self.image_dim = self.embedder.embed_dim  # Same model for images!
        
        # Initialize Qdrant Collections (dense only - no sparse with Jina)
        self.qdrant_manager.create_text_collection(
            dense_dim=self.dense_dim, 
            sparse_dim=0  # No SPLADE with Jina
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
        
        # Ensure collections exist (in case they were deleted)
        self.qdrant_manager.create_text_collection(dense_dim=self.dense_dim, sparse_dim=0)
        self.qdrant_manager.create_image_collection(image_dim=self.image_dim)
        
        # --- Check Supabase Cache First ---
        cached = self._load_from_cache(file_path)
        
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
        
        if not pages:
            print("[ERROR] Parsing failed: No pages found in API response.")
            return result

        print(f"[OK] Parsed {len(pages)} pages.")
        
        full_text = "\n".join([p.get("text", "") for p in pages])
        
        if not full_text.strip():
            print("[ERROR] Error: Document text is empty!")
            return result
        
        # Step 1: Split Text into Nodes (Chunks)
        temp_doc = Document(text=full_text, metadata={"source": file_path})
        nodes = self.text_splitter.get_nodes_from_documents([temp_doc])
        text_chunks = [node.get_content() for node in nodes]
        
        print(f"Split document into {len(nodes)} text nodes.")
        
        # Step 2: Generate Dense Embeddings via Jina API (batch processing)
        print(f"[EMBED] Generating embeddings via Jina API...")
        dense_vectors = self.embedder.embed_texts(text_chunks, task="retrieval.passage")
        print(f"[OK] Generated {len(dense_vectors)} embeddings")
        
        # Step 3: Package Text Data for Upsert (no sparse vectors with Jina)
        text_points: List[HybridPoint] = []
        NAMESPACE_UUID = uuid.UUID("00000000-0000-0000-0000-000000000000")
        
        for i, node in enumerate(nodes):
            # Deterministic ID based on content hash to prevent duplicates
            content_hash = hashlib.md5(node.get_content().encode()).hexdigest()
            point_id = str(uuid.uuid5(NAMESPACE_UUID, content_hash))
            
            dense_vec = dense_vectors[i]
            
            text_points.append(
                HybridPoint(
                    id=point_id,
                    text_chunk=node.get_content(),
                    dense_vector=dense_vec,
                    sparse_vector_indices=[],  # No sparse with Jina
                    sparse_vector_values=[],   # No sparse with Jina
                    payload={"source": file_path, **node.metadata}
                )
            )
        
        # Step 4: Check which text points already exist and only upsert new ones
        all_text_ids = [p.id for p in text_points]
        existing_text_ids = self.qdrant_manager.get_existing_text_ids(all_text_ids)
        
        new_text_points = [p for p in text_points if p.id not in existing_text_ids]
        
        if new_text_points:
            self.qdrant_manager.upsert_text_points(new_text_points)
            print(f"[OK] Upserted {len(new_text_points)} NEW text points (skipped {len(existing_text_ids)} existing).")
        else:
            print(f"[OK] All {len(text_points)} text points already exist. Skipping upsert.")
        
        result["text_count"] = len(new_text_points)
        
        # Step 5: Handle Images (Multi-Modal)
        all_images_meta = []
        for p in pages:
            if "images" in p:
                all_images_meta.extend(p["images"])
        
        # Choose indexing method based on config
        if config.URL_BASED_IMAGE_INDEXING:
            # URL-based: Embed directly from LlamaParse URLs (no download)
            image_count = self._index_images_from_urls(json_objs, source_file=file_path)
        else:
            # Local: Use downloaded images
            image_count = self._index_downloaded_images(images_data, source_file=file_path)
        
        result["image_count"] = image_count
        
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
                    image_path=img_info["url"],  # Store URL instead of local path
                    image_vector=image_vector,
                    payload={
                        "source": source_file,
                        "original_name": img_info.get("name", "unknown"),
                        "image_url": img_info["url"],  # Explicit URL field
                        "type": "image",
                        "indexed_from": "url"  # Mark as URL-indexed
                    }
                )
            )
        
        # Check which image points already exist and only upsert new ones
        if image_points:
            all_image_ids = [p.id for p in image_points]
            existing_image_ids = self.qdrant_manager.get_existing_image_ids(all_image_ids)
            
            new_image_points = [p for p in image_points if p.id not in existing_image_ids]
            
            if new_image_points:
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
        
        print(f"[EMBED] Embedding {len(valid_images)} images via Jina API...")
        print(f"   - {len(url_images)} from URLs")
        print(f"   - {len(local_images)} from local files")
        
        # Get paths/URLs for embedding
        image_sources = [img.get("path") or img.get("url") for img in valid_images]
        
        # Process images in batches to avoid API limits
        BATCH_SIZE = 10
        all_vectors = []
        
        for i in range(0, len(image_sources), BATCH_SIZE):
            batch = image_sources[i:i + BATCH_SIZE]
            batch_num = i // BATCH_SIZE + 1
            total_batches = (len(image_sources) + BATCH_SIZE - 1) // BATCH_SIZE
            
            print(f"   [BATCH {batch_num}/{total_batches}] Embedding {len(batch)} images...")
            
            try:
                batch_vectors = self.embedder.embed_images(batch)
                all_vectors.extend(batch_vectors)
                print(f"   [OK] Batch {batch_num} completed")
            except Exception as e:
                print(f"   [ERROR] Batch {batch_num} failed: {e}")
                # Add None placeholders for failed batch
                all_vectors.extend([None] * len(batch))
        
        # Filter out failed embeddings
        image_vectors = [v for v in all_vectors if v is not None]
        
        if not image_vectors:
            print(f"[ERROR] No images were successfully embedded")
            return 0
        
        print(f"[OK] Generated {len(image_vectors)} image embeddings (out of {len(valid_images)} total)")
        
        # Create image points - pair successfully embedded images with their vectors
        image_points: List[ImagePoint] = []
        NAMESPACE_UUID = uuid.UUID("00000000-0000-0000-0000-000000000000")
        
        # Build list of (image_info, vector) pairs, skipping None vectors
        vector_idx = 0
        for img_info, vec in zip(valid_images, all_vectors):
            if vec is None:
                continue  # Skip failed embeddings
            
            image_path = img_info.get("path") or img_info.get("url", "")
            storage_type = img_info.get("storage", "")
            
            # Mark as URL storage if it's a URL
            if image_path.startswith("http"):
                storage_type = "url"
            
            # Deterministic ID based on image name
            name_for_hash = img_info.get("name", image_path)
            path_hash = hashlib.md5(name_for_hash.encode()).hexdigest()
            point_id = str(uuid.uuid5(NAMESPACE_UUID, path_hash))
            
            image_points.append(
                ImagePoint(
                    id=point_id,
                    image_path=image_path,
                    image_vector=vec,
                    payload={
                        "source": source_file, 
                        "original_name": img_info.get("name", "unknown"),
                        "type": "image",
                        "storage": storage_type,
                        "image_url": image_path if image_path.startswith("http") else None,
                        "image_path": image_path  # For backward compatibility
                    }
                )
            )

        # Check which image points already exist and only upsert new ones
        if image_points:
            all_image_ids = [p.id for p in image_points]
            existing_image_ids = self.qdrant_manager.get_existing_image_ids(all_image_ids)
            
            new_image_points = [p for p in image_points if p.id not in existing_image_ids]
            
            if new_image_points:
                self.qdrant_manager.upsert_image_points(new_image_points)
                print(f"[OK] Upserted {len(new_image_points)} NEW image points (skipped {len(existing_image_ids)} existing).")
            else:
                print(f"[OK] All {len(image_points)} image points already exist. Skipping upsert.")
            
            return len(new_image_points)
        
        return 0