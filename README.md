# ModalMuse 🎵🖼️

> **Multi-Modal Retrieval-Augmented Generation (RAG) — PDF understanding with text + image intelligence**

ModalMuse is a production-grade RAG system that parses complex PDFs (text and images), indexes them into a unified multi-modal vector space, and answers questions using a vision-capable LLM. It combines dense, sparse, and image-visual search with a sophisticated two-stage reranking pipeline and real-time streaming updates via WebSocket.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Tech Stack](#tech-stack)
- [System Architecture](#system-architecture)
- [Repository Structure](#repository-structure)
- [Indexing Pipeline (Deep Dive)](#indexing-pipeline-deep-dive)
- [Retrieval Pipeline (Deep Dive)](#retrieval-pipeline-deep-dive)
- [User Request Flow](#user-request-flow)
- [Data Models & Qdrant Schema](#data-models--qdrant-schema)
- [Configuration Reference](#configuration-reference)
- [Building Locally](#building-locally)
- [Docker Setup](#docker-setup)
- [Environment Variables](#environment-variables)
- [API Reference](#api-reference)
- [Engineering Decisions & Challenges](#engineering-decisions--challenges)
- [Performance Characteristics](#performance-characteristics)
- [Contributing](#contributing)
- [Acknowledgements](#acknowledgements)

---

## Overview

ModalMuse solves a fundamental gap in standard RAG systems: **images are first-class citizens**. Most RAG pipelines discard figures, charts, diagrams and tables buried in PDFs. ModalMuse treats every page element as retrievable, projecting both text chunks and images into the same 1024-dimensional embedding space so a single query can pull back the most relevant content regardless of modality.

The pipeline is:

```
PDF Upload → LlamaParse (text + images) → Dual Embedding (bge-small + jina-clip)
         → Qdrant (two collections) → Hybrid Search (Dense + Sparse + Visual + Caption)
         → RRF Fusion → Cross-Encoder Rerank → Llama 4 Vision LLM → Streaming Response
```

---

## Key Features

| Feature | Description |
|---|---|
| **Multi-Modal Indexing** | Extracts and indexes both text chunks and images from PDFs via LlamaParse |
| **Hybrid Search** | Parallel dense (bge-small), sparse (BM25), image-visual (jina-clip) and caption-text searches |
| **RRF Fusion** | Score-distribution-agnostic Reciprocal Rank Fusion merges dense + sparse text results |
| **Cross-Encoder Reranking** | Unified reranking of text and image candidates via `bge-reranker-base` |
| **AI Image Captioning** | Groq Vision (Llama 4 Scout) generates rich, domain-specific captions for each image during indexing |
| **Semantic Response Cache** | Cosine-similarity-based cache in Qdrant deduplicates paraphrased queries at ≥ 0.80 threshold |
| **SHA-256 Parse Cache** | Supabase-backed parse cache prevents re-parsing identical PDFs |
| **Streaming WebSocket** | Real-time phase updates (Embedding → Search → Fusion → Reranking → Generation) |
| **Idempotent Indexing** | MD5 content-hash deduplication — re-uploading the same PDF is a no-op |
| **Docker-first** | Full `docker-compose.yml` for Qdrant + Infinity embedding server + backend |

---

## Tech Stack

| Layer | Technology | Role |
|---|---|---|
| **Frontend** | Next.js (TypeScript) | Chat UI with streaming WebSocket client |
| **Backend API** | FastAPI (Python, async) | REST + WebSocket endpoints |
| **PDF Parsing** | LlamaParse (LlamaIndex Cloud) | Layout-aware extraction of text, tables, and images |
| **Text Embeddings** | `BAAI/bge-small-en-v1.5` (384-dim) | Dense text chunk embeddings via local Infinity server |
| **Image Embeddings** | `jinaai/jina-clip-v1` (768-dim) | Cross-modal image and text-to-image embeddings via Infinity |
| **Sparse Embeddings** | BM25 (HashingVectorizer, local) | Keyword-weighted sparse vectors for hybrid retrieval |
| **Reranker** | `BAAI/bge-reranker-base` | Cross-encoder reranking of combined text + image candidates |
| **Vector Database** | Qdrant | Stores dense, sparse, and multi-vector collections |
| **LLM** | `meta-llama/llama-4-scout-17b-16e-instruct` (Groq) | Vision-capable answer generation + image captioning |
| **Caching** | Supabase (PostgreSQL + Storage) | Parse result cache + persistent image hosting |
| **Embedding Server** | Infinity (self-hosted) | Local model serving for text and image embeddings |
| **Containerization** | Docker + Docker Compose | One-command local setup |

---

## System Architecture

The system is organized into three major layers: the **frontend**, the **backend API**, and the **data/AI services layer**. The backend itself is split into two sub-systems: the **Ingestion Pipeline** and the **Retrieval Pipeline**.

```
┌──────────────────────────────────────────────────────────┐
│                     Frontend (Next.js)                   │
│   File Upload UI  ──  Chat UI  ──  Streaming WS Client   │
└───────────────────────────┬──────────────────────────────┘
                            │ HTTP + WebSocket
┌───────────────────────────▼──────────────────────────────┐
│                    FastAPI Backend                        │
│   POST /upload  ──  POST /query  ──  WS /ws/query        │
│              ↕ Indexer          ↕ GroqGenerator           │
└───┬───────────────────────────────────────────┬──────────┘
    │ Ingestion Pipeline                        │ Retrieval Pipeline
    ▼                                           ▼
LlamaParse API                          Infinity Embed Server
    │                                     (bge-small + jina-clip
    │ Markdown + Image URLs               + bge-reranker-base)
    ▼                                           │
Supabase Storage (images)                      │
Supabase DB (parse cache)                      ▼
    │                                       Qdrant
    └──────► Qdrant ◄──────────────────────────┘
         (2 collections)          Groq API (Llama 4 Vision)
```

### Component Responsibilities

**`indexer.py` — Ingestion Engine**
Orchestrates the entire ingestion flow: SHA-256 hash check → LlamaParse → image download/upload to Supabase → BM25 + bge-small text embedding → jina-clip image embedding + Groq captioning → Qdrant upsert. All operations are idempotent via content-hash deduplication.

**`retriever.py` — Retrieval + Generation Engine**
Contains two classes: `MultiModalRetriever` (handles 4-way parallel vector search, RRF fusion, and cross-encoder reranking) and `GroqGenerator` (query expansion, semantic cache check, message construction with base64 images, streaming LLM call).

**`qdrant_manager.py` — Vector DB Abstraction**
Wraps all Qdrant operations: collection creation (multi-vector schema), upsert with both dense + sparse vectors, deduplication ID lookups, and semantic response cache queries.

**`bm25.py` — Sparse Encoder**
Local BM25 encoder using `HashingVectorizer` from scikit-learn. Generates sparse indices/values compatible with Qdrant's sparse vector format, requiring no external API.

**`rrf_reranker.py` — Rank Fusion**
Pure Python implementation of Reciprocal Rank Fusion. Takes dense and sparse node lists, assigns `1/(k + rank)` scores with configurable per-source weights, deduplicates by node ID, and returns a unified sorted list.

**`local_client.py` — Infinity Client**
Async HTTP client for the self-hosted Infinity embedding server. Handles text embedding, image embedding (URL or local path), and cross-encoder reranking calls.

**`supabase_client.py` — Cache Layer**
Supabase integration for parse result caching (PostgreSQL) and image storage (Supabase Storage bucket). Provides `get_parse_cache`, `save_parse_cache`, `upload_image_from_response`, and `get_image_public_url`.

**`config.py` — Centralized Config**
Single source of truth for all tuneable parameters: model names, dimensions, collection names, top-K values, chunk size/overlap, RRF constant, cache thresholds, and feature flags.

**`api/`** — FastAPI application with REST and WebSocket routes.

**`frontend/`** — Next.js app with TypeScript, handling file upload, query input, and streaming chat rendering.

---

## Repository Structure

```
ModalMuse/
├── api/                        # FastAPI application
│   └── ...                     # Route definitions, request/response models
├── frontend/                   # Next.js chat UI
│   └── ...                     # Pages, components, WebSocket client
├── indexer.py                  # Ingestion pipeline (755 lines)
├── retriever.py                # Retrieval + generation pipeline (909 lines)
├── qdrant_manager.py           # Qdrant collection management + vector ops
├── bm25.py                     # Local BM25 sparse encoder
├── rrf_reranker.py             # Reciprocal Rank Fusion implementation
├── local_client.py             # Async Infinity embedding + reranking client
├── supabase_client.py          # Supabase parse cache + image storage client
├── supabase_schema.sql         # SQL schema for parse cache table
├── config.py                   # Centralized configuration
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Backend container definition
├── docker-compose.yml          # Full-stack local orchestration
├── .env.example                # Environment variable template
└── README.md                   # This file
```

---

## Indexing Pipeline (Deep Dive)

The ingestion pipeline converts a raw PDF into searchable multi-modal vectors stored in Qdrant. The design goal is **idempotency** and **cost efficiency** — re-processing the same document should be nearly free.

### Step 1 — SHA-256 Cache Check

Before calling any external API, the `Indexer` computes a SHA-256 hash of the uploaded file bytes and queries Supabase for a matching `file_hash` in the `parse_cache` table. On a cache hit, the stored JSON parse result and image URLs are returned immediately, bypassing LlamaParse entirely.

```python
file_hash = hashlib.sha256(open(file_path, "rb").read()).hexdigest()
cached = get_parse_cache(file_hash)  # Supabase lookup
```

### Step 2 — LlamaParse (Cache Miss Path)

On a cache miss, LlamaParse is called with `result_type="markdown"`. It returns a JSON structure with:
- `pages[].text` — Markdown-formatted text per page (tables, headers, paragraphs)
- `pages[].images[].name` — Image filenames extractable via the LlamaParse API

```
PDF → LlamaParse → { pages: [{ text: "...", images: [{ name: "img-1.png" }] }] }
```

### Step 3 — Image Extraction & Persistent Storage

LlamaParse image URLs are temporary (they expire). The pipeline downloads each image via `httpx` and uploads it to **Supabase Storage**, where it receives a permanent public URL. This URL is stored in both the parse cache and later as Qdrant payload.

Two modes are configurable:
- `SUPABASE_STORAGE_ENABLED=true` — download + upload to Supabase (default, production-safe)
- `URL_BASED_IMAGE_INDEXING=true` — use LlamaParse URLs directly (fast, but URLs expire)

### Step 4 — Text Chunking

Page-level documents are split using LlamaIndex's `SentenceSplitter` with:
- `CHUNK_SIZE = 512` tokens
- `CHUNK_OVERLAP = 50` tokens

Page numbers and source file metadata are preserved on each chunk node.

### Step 5 — Deduplication Check

Before embedding, the pipeline computes MD5-based UUIDv5 identifiers for every chunk and queries Qdrant for already-existing IDs. Only new chunks proceed to embedding. This makes re-indexing the same document (e.g., after a crash mid-way) resumable without duplicates.

```python
content_hash = hashlib.md5(node.get_content().encode()).hexdigest()
point_id = str(uuid.uuid5(NAMESPACE_UUID, content_hash))
```

### Step 6 — Dual Text Embedding (Dense + Sparse)

New text chunks are embedded in batches of 50:

- **Dense**: `bge-small-en-v1.5` via local Infinity server → 384-dimensional float vector
- **Sparse**: Local BM25 (`HashingVectorizer`) → `{indices: [...], values: [...]}` sparse vector

Both are stored together in a single Qdrant point under the `text-dense` and `text-sparse` named vectors.

### Step 7 — Image Embedding + AI Captioning

For each image (batched per `LOCAL_IMAGE_BATCH_SIZE = 10`):

1. **Visual Embedding**: `jina-clip-v1` via Infinity embeds the image into a 768-dimensional vector stored under `image-visual`.
2. **Groq Vision Captioning**: Llama 4 Scout generates a rich, domain-specific technical caption:
   - All visible text, labels, axis names, formulas
   - Precise technical terminology
   - Ending with `Keywords:` and 5–10 specific terms
3. **Caption Embedding**: The generated caption is embedded with `bge-small` → 384-dim vector stored under `caption-text`.

This dual representation enables both visual-similarity and text-to-image retrieval paths.

### Step 8 — Qdrant Upsert

All points are upserted to their respective collections. The upsert is per-batch and crash-resilient — batches already stored survive restarts.

### Indexing Flow Summary

```
PDF File
  │
  ├─[SHA-256]──► Supabase Parse Cache?
  │               ├─ HIT: Load JSON + image URLs → Step 5
  │               └─ MISS:
  │                   ├─ LlamaParse API → Markdown + image names
  │                   ├─ Download images (httpx)
  │                   ├─ Upload images → Supabase Storage (permanent URLs)
  │                   └─ Save to parse cache
  │
  ├─ SentenceSplitter → text chunks (512 tok, 50 overlap)
  │
  ├─[MD5 dedup]─► Qdrant ID check → filter new-only chunks
  │
  ├─[bge-small] → dense 384-dim vectors
  ├─[BM25]      → sparse indices/values
  └─ Qdrant upsert → text collection
  │
  └─[images]
      ├─[jina-clip] → visual 768-dim vectors
      ├─[Groq Vision] → rich technical caption
      ├─[bge-small]  → caption 384-dim vectors
      └─ Qdrant upsert → image collection
```

---

## Retrieval Pipeline (Deep Dive)

The retrieval pipeline executes on every user query. It is fully async, using `asyncio.gather` for parallelism at every stage.

### Step 1 — Query Expansion (Optional)

Short or vague queries (≤ 8 words) are rewritten by the LLM before search. This increases recall for under-specified questions:

```
"explain the diagram" → "Explain the technical diagram including labels, axes, and key concepts shown"
```

### Step 2 — Parallel Query Embedding

Two embeddings are computed in parallel:
- `bge-small` → 384-dim text query vector
- `jina-clip` → 768-dim image query vector (text prompt encoded in CLIP's shared space)

### Step 3 — Semantic Response Cache Check

The text query embedding is compared against cached responses in Qdrant's `response_cache` collection (cosine similarity ≥ 0.80 threshold). A cache hit streams the cached response immediately, skipping the entire retrieval + generation pipeline.

Cached entries expire after `SEMANTIC_CACHE_TTL_HOURS = 1` and the cache is capped at `SEMANTIC_CACHE_MAX_ENTRIES = 200`.

### Step 4 — 4-Way Parallel Vector Search

Four searches execute simultaneously via `asyncio.gather`:

| Search | Collection | Vector | Top-K |
|---|---|---|---|
| Dense text | `multimodal_text_index` | `text-dense` (384-dim) | 10 |
| Sparse text (BM25) | `multimodal_text_index` | `text-sparse` | 10 |
| Image-visual | `multimodal_image_index` | `image-visual` (768-dim) | 5 |
| Caption-text | `multimodal_image_index` | `caption-text` (384-dim) | 5 |

Image results from caption-text and image-visual are merged with deduplication (caption-text results take priority as they tend to be higher precision).

### Step 5 — RRF Fusion (Dense + Sparse Text)

Dense and sparse text results are fused using Reciprocal Rank Fusion:

```
RRF_score(d) = w_dense / (k + rank_dense(d)) + w_sparse / (k + rank_sparse(d))
```

Where `k = 60` (standard RRF constant), and `w_dense = w_sparse = 1.0` by default. Nodes appearing in both lists receive a boosted combined score. The fused list is rank-sorted by RRF score.

### Step 6 — Unified Cross-Encoder Reranking

Fused text results and image results are combined into a single candidate pool and passed to `bge-reranker-base` (via Infinity). The cross-encoder "reads" each (query, candidate) pair deeply to assign precise relevance scores.

Images are represented by their Groq-generated captions during reranking, enabling the cross-encoder to meaningfully compare image relevance against text relevance.

After reranking, results are slotted:
- `text_slots = FINAL_RERANK_TOP_N - IMAGE_RESULT_SLOTS` (default: 5)
- `image_slots = IMAGE_RESULT_SLOTS` (default: 2)

### Step 7 — Vision LLM Generation

The top-ranked context is assembled into a message:

1. **Text chunks** → formatted as `[Source N]: ...` in the system context string
2. **Images** → downloaded/fetched, resized if > 30M pixels, encoded as base64 JPEG, injected into the multimodal message payload

The message is sent to `llama-4-scout-17b-16e-instruct` via Groq with streaming enabled. Chunks are yielded as WebSocket events in real time.

### Step 8 — Response Cache Store

After generation completes, the query embedding and full response text are stored in the Qdrant response cache for future semantic-similarity lookup.

### Retrieval Flow Summary

```
User Query
  │
  ├─[≤8 words?] → Query Expansion (Groq)
  │
  ├─[Parallel embed]
  │   ├─ bge-small → text query vec (384-dim)
  │   └─ jina-clip → image query vec (768-dim)
  │
  ├─[Cache check] → Qdrant response_cache (cosine ≥ 0.80)
  │   └─ HIT: stream cached response → done
  │
  ├─[Parallel 4-way search]
  │   ├─ dense text   → top 10 text nodes
  │   ├─ sparse BM25  → top 10 text nodes
  │   ├─ image-visual → top 5 image nodes
  │   └─ caption-text → top 5 image nodes
  │
  ├─[Image dedup] → merge caption + visual, prefer caption
  │
  ├─[RRF Fusion] → dense + sparse text → unified text list
  │
  ├─[Unified rerank] → text + images → cross-encoder (bge-reranker-base)
  │   └─ Slot allocation: top 5 text + top 2 images
  │
  ├─[Message build] → text context + base64 images
  │
  ├─[Groq stream] → llama-4-scout-17b-16e-instruct
  │   └─ Yield generation chunks via WebSocket
  │
  └─[Cache store] → save embedding + response to response_cache
```

---

## User Request Flow

### Document Upload Flow

```
1. User selects PDF in Next.js UI
2. Frontend sends multipart/form-data POST to /api/upload
3. FastAPI saves file to temp storage
4. Indexer.index_document(file_path) is called
5. → SHA-256 hash computed
6. → Supabase parse cache queried
7. → [Cache miss] LlamaParse API called (~30–120s for large PDFs)
8. → Images downloaded and uploaded to Supabase Storage
9. → Parse result saved to Supabase cache
10. → Text chunks created via SentenceSplitter
11. → New chunks filtered by Qdrant dedup check
12. → bge-small dense + BM25 sparse embeddings generated
13. → Text points upserted to multimodal_text_index
14. → Images embedded (jina-clip) + captioned (Groq Vision) + caption embedded (bge-small)
15. → Image points upserted to multimodal_image_index
16. FastAPI returns { text_count, image_count, from_cache } to frontend
```

### Query / Chat Flow (WebSocket Streaming)

```
1. User types query and submits
2. Frontend opens WebSocket to /ws/query
3. Backend receives query string
4. GroqGenerator.astream_query_detailed() begins yielding events:

   Phase: "embedding"   → { type: "phase", message: "Creating query embedding..." }
   Phase: "search"      → { type: "phase", message: "Searching dense, sparse & image..." }
   Data:  "chunks_found"→ { text_count, image_count, chunks: [...previews] }
   Phase: "fusion"      → { type: "phase", message: "RRF fused → N unique text nodes" }
   Phase: "reranking"   → { type: "phase", message: "Reranked → N text + M images" }
   Phase: "generation"  → { type: "phase", message: "Generating response..." }
   Tokens:"generation"  → { type: "generation", chunk: "token..." } × many
   Final: "sources"     → { sources: [{content, score, type, metadata}] }
   Final: "done"        → { total_duration_ms }

5. Frontend renders each event type accordingly:
   - Phase events → progress bar / status indicator
   - Generation chunks → streaming chat bubble
   - Sources → collapsible citations panel
```

### WebSocket Event Types Reference

| Event type | Payload fields | Purpose |
|---|---|---|
| `phase` | `phase`, `status`, `message`, `duration_ms?` | Pipeline progress indicator |
| `chunks_found` | `text_count`, `dense_count`, `sparse_count`, `image_count`, `chunks[]` | Search result preview |
| `generation` | `chunk` | LLM token stream |
| `sources` | `sources[]`, `total_duration_ms` | Citation metadata |
| `done` | `total_duration_ms`, `cached?` | Completion signal |
| `error` | `message` | Error notification |

---

## Data Models & Qdrant Schema

### Text Collection — `multimodal_text_index`

Each point represents a 512-token text chunk from a page.

| Field | Type | Description |
|---|---|---|
| `id` | UUIDv5 (string) | Deterministic from MD5 of chunk content |
| `text-dense` | float32[384] | bge-small embedding of chunk text |
| `text-sparse` | sparse vector | BM25 indices and values |
| `payload.text_chunk` | string | Raw text content |
| `payload.source` | string | Source file path |
| `payload.page` | int | 1-indexed page number |
| `payload.file_name` | string | Original filename |

### Image Collection — `multimodal_image_index`

Each point represents one image extracted from a PDF page.

| Field | Type | Description |
|---|---|---|
| `id` | UUIDv5 (string) | Deterministic from MD5 of image name |
| `image-visual` | float32[768] | jina-clip-v1 visual embedding |
| `caption-text` | float32[384] | bge-small embedding of AI-generated caption |
| `payload.image_url` | string | Permanent Supabase Storage URL |
| `payload.caption` | string | Groq-generated technical caption |
| `payload.page` | int | Source page number |
| `payload.file_name` | string | Source filename |
| `payload.original_name` | string | LlamaParse image filename |
| `payload.storage` | string | Storage backend ("supabase" or "url") |

### Response Cache Collection — `response_cache`

Semantic response cache for avoiding redundant LLM calls.

| Field | Type | Description |
|---|---|---|
| `id` | UUID | Random |
| vector | float32[384] | bge-small embedding of query |
| `payload.query_text` | string | Original query string |
| `payload.response` | string | Full LLM response |
| `payload.sources` | JSON | Source nodes used |
| `payload.created_at` | timestamp | For TTL expiry |

### Supabase `parse_cache` Table Schema

```sql
CREATE TABLE parse_cache (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    file_hash    TEXT UNIQUE NOT NULL,   -- SHA-256 of PDF bytes
    file_name    TEXT,
    parsed_json  JSONB,                  -- Full LlamaParse JSON result
    images_data  JSONB,                  -- List of { path, url, name, page }
    job_id       TEXT,                   -- LlamaParse job ID
    created_at   TIMESTAMPTZ DEFAULT now()
);
```

---

## Configuration Reference

All configuration is centralized in `config.py` and overrideable via environment variables.

### Embedding Models

| Parameter | Default | Description |
|---|---|---|
| `LOCAL_TEXT_MODEL` | `BAAI/bge-small-en-v1.5` | Text embedding model (Infinity) |
| `LOCAL_TEXT_DIMENSIONS` | `384` | Output dimensions for text embedder |
| `LOCAL_IMAGE_MODEL` | `jinaai/jina-clip-v1` | Image embedding model (Infinity) |
| `LOCAL_IMAGE_DIMENSIONS` | `768` | Output dimensions for image embedder |
| `LOCAL_RERANK_MODEL` | `BAAI/bge-reranker-base` | Cross-encoder reranker model |
| `LOCAL_EMBED_URL` | `http://localhost:7997` | Infinity server base URL |

### Retrieval Parameters

| Parameter | Default | Description |
|---|---|---|
| `TEXT_SIMILARITY_TOP_K` | `10` | Dense search top-K candidates |
| `SPARSE_TOP_K` | `10` | BM25 sparse search top-K |
| `IMAGE_SIMILARITY_TOP_K` | `5` | Image search top-K per mode |
| `RRF_K` | `60` | RRF constant (standard value) |
| `FINAL_RERANK_TOP_N` | `7` | Total results after unified reranking |
| `IMAGE_RESULT_SLOTS` | `2` | Guaranteed image slots in final output |

### Semantic Cache Parameters

| Parameter | Default | Description |
|---|---|---|
| `SEMANTIC_CACHE_THRESHOLD` | `0.80` | Cosine similarity threshold for cache hit |
| `SEMANTIC_CACHE_TTL_HOURS` | `1` | Hours before cache entries expire |
| `SEMANTIC_CACHE_MAX_ENTRIES` | `200` | Maximum entries in cache collection |

### Text Chunking

| Parameter | Default | Description |
|---|---|---|
| `CHUNK_SIZE` | `512` | Tokens per chunk |
| `CHUNK_OVERLAP` | `50` | Overlap tokens between adjacent chunks |

### LLM

| Parameter | Default | Description |
|---|---|---|
| `GROQ_MODEL_NAME` | `meta-llama/llama-4-scout-17b-16e-instruct` | Groq model for generation + captioning |
| `LLM_MAX_NEW_TOKENS` | `1024` | Maximum generation tokens |
| `LLM_TEMPERATURE` | `0.7` | Sampling temperature |

### Image Handling Flags

| Parameter | Default | Description |
|---|---|---|
| `URL_BASED_IMAGE_INDEXING` | `false` | Use LlamaParse URLs directly (no download) |
| `SUPABASE_STORAGE_ENABLED` | `true` | Upload images to Supabase Storage |
| `LOCAL_IMAGE_BATCH_SIZE` | `10` | Images per embedding batch |
| `LOCAL_BATCH_DELAY_SECONDS` | `0` | Delay between image batches |

---

## Building Locally

### Prerequisites

- Python 3.10+
- Node.js 18+
- Docker + Docker Compose
- API keys: Groq, LlamaParse, Supabase (optional for caching)

### Step 1 — Clone the Repository

```bash
git clone https://github.com/yugborana/ModalMuse.git
cd ModalMuse
```

### Step 2 — Set Up Environment Variables

```bash
cp .env.example .env
```

Edit `.env` with your credentials (see [Environment Variables](#environment-variables) below).

### Step 3 — Start Infrastructure Services

This starts Qdrant (vector DB) and the Infinity embedding server (local model serving):

```bash
docker-compose up -d qdrant infinity
```

Wait for both containers to be healthy. Verify:
```bash
# Qdrant dashboard
curl http://localhost:6333/dashboard

# Infinity health check
curl http://localhost:7997/health
```

### Step 4 — Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Step 5 — Set Up Supabase (Optional — for Parse Cache & Image Storage)

If you want parse caching and persistent image storage, create a Supabase project and run:

```sql
-- Run supabase_schema.sql in your Supabase SQL Editor
```

Then populate `SUPABASE_URL` and `SUPABASE_KEY` in your `.env`.

If you skip Supabase, set:
```env
SUPABASE_STORAGE_ENABLED=false
URL_BASED_IMAGE_INDEXING=false
```
Images will be stored locally in `downloaded_images/`.

### Step 6 — Start the Backend

```bash
cd api
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

### Step 7 — Start the Frontend

```bash
cd frontend
npm install
npm run dev
```

The UI will be available at `http://localhost:3000`.

### Step 8 — Index a Document

Upload a PDF through the UI at `http://localhost:3000`, or via curl:

```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@your_document.pdf"
```

### Step 9 — Query

Use the chat interface at `http://localhost:3000` or query the WebSocket directly:

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/query');
ws.onopen = () => ws.send(JSON.stringify({ query: "Explain the architecture diagram" }));
ws.onmessage = (e) => console.log(JSON.parse(e.data));
```

---

## Docker Setup

The provided `docker-compose.yml` orchestrates all services. A full stack can be launched with:

```bash
docker-compose up --build
```

### Services

| Service | Port | Description |
|---|---|---|
| `backend` | `8000` | FastAPI backend (built from `Dockerfile`) |
| `qdrant` | `6333` | Qdrant vector database |
| `infinity` | `7997` | Local embedding + reranking server |
| `frontend` | `3000` | Next.js UI (if included in compose) |

### Dockerfile Overview

The `Dockerfile` builds the Python backend:
- Base image: `python:3.11-slim`
- Installs system deps (libGL for image processing)
- Copies `requirements.txt` and installs dependencies
- Exposes port 8000
- Entrypoint: `uvicorn api.main:app`

---

## Environment Variables

```env
# ── Required ──────────────────────────────────────────────────────
GROQ_API_KEY=gsk_...                   # Groq API key (LLM + captioning)
LLAMA_PARSE_API_KEY=llx-...            # LlamaParse API key (PDF parsing)

# ── Supabase (Optional) ───────────────────────────────────────────
SUPABASE_URL=https://xxx.supabase.co   # Supabase project URL
SUPABASE_KEY=eyJ...                    # Supabase anon/service key
SUPABASE_STORAGE_ENABLED=true          # Enable image upload to Supabase Storage

# ── Qdrant ────────────────────────────────────────────────────────
QDRANT_URL=http://localhost:6333       # Qdrant URL (local or cloud)
QDRANT_API_KEY=                        # Required only for Qdrant Cloud

# ── Infinity (Local Embedding Server) ─────────────────────────────
LOCAL_EMBED_URL=http://localhost:7997  # Infinity server URL

# ── API Server ────────────────────────────────────────────────────
API_HOST=0.0.0.0
API_PORT=8000
CORS_ORIGINS=http://localhost:3000

# ── Feature Flags ─────────────────────────────────────────────────
URL_BASED_IMAGE_INDEXING=false         # true = skip image download (faster, URLs expire)
```

---

## API Reference

### `POST /upload`

Upload and index a PDF document.

**Request:** `multipart/form-data` with field `file` (PDF).

**Response:**
```json
{
  "status": "success",
  "text_count": 142,
  "image_count": 23,
  "from_cache": false,
  "message": "Indexed 142 text chunks and 23 images"
}
```

### `POST /query`

Query the indexed documents (non-streaming).

**Request:**
```json
{ "query": "Explain the RoPE positional encoding diagram" }
```

**Response:**
```json
{
  "response": "Rotary Positional Encoding (RoPE) ...",
  "sources": [
    { "content": "...", "score": 0.91, "type": "text", "metadata": {...} },
    { "content": "...", "score": 0.87, "type": "image", "metadata": {"image_url": "..."} }
  ]
}
```

### `WebSocket /ws/query`

Real-time streaming query with phase events.

**Send:**
```json
{ "query": "What does Figure 3 show?" }
```

**Receive stream:** (series of JSON events — see [WebSocket Event Types](#websocket-event-types-reference))

### `GET /collections`

Returns Qdrant collection stats (point counts, vector configs).

### `DELETE /collections`

Deletes and recreates all Qdrant collections (full re-index required).

---

## Engineering Decisions & Challenges

### Why Two Separate Embedding Models?

Text chunks and images require fundamentally different embedding models. `bge-small-en-v1.5` is optimized for semantic text similarity at low cost (384-dim). `jina-clip-v1` is a CLIP-aligned model that projects images and text into a shared latent space (768-dim), enabling text queries to retrieve visually relevant images. Using a single model for both would degrade either text or image retrieval quality.

### Why BM25 + Dense Hybrid?

Dense embeddings excel at semantic understanding but can miss exact keyword matches. BM25 is the opposite — great for keyword precision but blind to semantics. RRF fusion gets the best of both: it handles paraphrased queries (dense wins) and precise term lookup (sparse wins) in a single pass. The RRF constant `k=60` provides even, non-discriminating weighting, which is generally robust across domains.

### Why RRF Instead of Score Normalization?

Dense and sparse scores come from different mathematical distributions (cosine similarity vs. BM25 IDF-weighted TF). Directly averaging or weighting raw scores is unreliable. RRF is score-agnostic — it only looks at rank order — making it stable even when score distributions shift between documents or query types.

### Why Unified Cross-Encoder Reranking (Text + Images Together)?

Early versions reranked text and images separately, then merged by weight. This created an arbitrary boundary: a highly relevant image could be crowded out by mediocre text results. By feeding the cross-encoder the combined candidate pool (images represented by their Groq captions), the reranker can make fair apples-to-apples comparisons and surface the single most relevant context regardless of modality.

### Why AI-Generated Captions for Images?

Raw image embeddings (jina-clip) handle visual similarity well, but a text query like "how does the attention mechanism work?" may not retrieve an attention diagram through visual features alone. By generating dense technical captions with Groq Vision and embedding those captions with bge-small, we add a high-precision text→image retrieval path that understands domain terminology.

### Why Content-Hash Deduplication?

Re-indexing the same document (e.g., after adding new pages) is common in production. Without deduplication, existing chunks would be duplicated in Qdrant, inflating storage and degrading retrieval quality. UUIDv5 from MD5 content hashes makes deduplication deterministic and cheap (a batch Qdrant ID lookup before any embedding).

### Async Everywhere

The retrieval pipeline uses `asyncio.gather` at two levels: (1) parallel text + image query embedding, and (2) 4-way parallel Qdrant search. This reduces wall-clock latency by ~40% compared to sequential execution, as all four searches are IO-bound (network calls to Qdrant).

### Graceful Degradation

Every external dependency has a fallback:
- Supabase cache unavailable → parse every time, log warning
- Image captioning fails → store image without caption, use filename proxy for reranking
- Response cache store fails → generation still completes, just not cached
- Embedding batch fails → log error, skip batch, continue with next batch (partial indexing)

---

## Performance Characteristics

These are approximate figures on typical hardware (4-core VM, 8GB RAM). Actual performance varies by document size and hardware.

| Operation | Typical Latency |
|---|---|
| Parse cache hit (Supabase) | < 500ms |
| LlamaParse (20-page PDF) | 30–90s |
| Text embedding (50 chunks, local Infinity) | ~1–2s |
| Image embedding + captioning (10 images) | ~15–25s (captioning dominates) |
| Query embedding (2 models, parallel) | ~50–150ms |
| 4-way Qdrant search (parallel) | ~50–200ms |
| Cross-encoder reranking (7–15 candidates) | ~100–300ms |
| Groq LLM (streaming, 512 output tokens) | ~1–3s to first token |
| End-to-end query (cache miss) | ~500ms–1.5s |
| End-to-end query (semantic cache hit) | < 100ms |

---

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make your changes with tests if applicable
4. Commit: `git commit -m 'feat: add my feature'`
5. Push: `git push origin feature/my-feature`
6. Open a Pull Request

### Development Tips

- Run `python -m pytest` for backend tests
- Use `python retriever.py` as a quick CLI test of the full pipeline
- The `config.py` validation function (`validate_config()`) helps catch missing API keys early
- Qdrant's dashboard at `http://localhost:6333/dashboard` is invaluable for inspecting indexed data

---

## Acknowledgements

- [LlamaIndex](https://www.llamaindex.ai/) — Document parsing and indexing framework
- [Jina AI](https://jina.ai/) — `jina-clip-v1` for cross-modal embeddings
- [BAAI](https://huggingface.co/BAAI) — `bge-small-en-v1.5` and `bge-reranker-base`
- [Groq](https://groq.com/) — Ultra-fast LLM inference for Llama 4
- [Qdrant](https://qdrant.tech/) — High-performance vector database with multi-vector support
- [Supabase](https://supabase.com/) — Open-source backend for caching and storage
- [Infinity](https://github.com/michaelfeil/infinity) — Self-hosted embedding and reranking server

---

## License

Distributed under the MIT License.

## Contact

**Yug Borana** — [yugborana000@gmail.com](mailto:yugborana000@gmail.com)

Project: [https://github.com/yugborana/ModalMuse](https://github.com/yugborana/ModalMuse)
