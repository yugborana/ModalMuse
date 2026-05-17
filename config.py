# config.py - Centralized Configuration for ModalMuse
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Base Directories ---
BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "data"

# --- LLM Configuration (Groq) ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL_NAME = "meta-llama/llama-4-scout-17b-16e-instruct"

# LLM Parameters
LLM_MAX_NEW_TOKENS = 1024
LLM_TEMPERATURE = 0.7

# --- Local Embedding Server (Infinity via Docker) ---
LOCAL_EMBED_URL = os.getenv("LOCAL_EMBED_URL", "http://localhost:7997")

# Text Embeddings (bge-small — fast, 384-dim, text-only)
LOCAL_TEXT_MODEL = "BAAI/bge-small-en-v1.5"
LOCAL_TEXT_DIMENSIONS = 384

# Image Embeddings (jina-clip — multimodal, 768-dim, images + text→image)
LOCAL_IMAGE_MODEL = "jinaai/jina-clip-v1"
LOCAL_IMAGE_DIMENSIONS = 768

# Reranker
LOCAL_RERANK_MODEL = "BAAI/bge-reranker-base"

# Image embedding batch configuration (no rate limits locally)
LOCAL_IMAGE_BATCH_SIZE = 10     
LOCAL_BATCH_DELAY_SECONDS = 0   


# --- Qdrant Configuration ---
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)  # Required for Qdrant Cloud
TEXT_COLLECTION_NAME = "multimodal_text_index"
IMAGE_COLLECTION_NAME = "multimodal_image_index"

# --- LlamaParse Configuration ---
LLAMA_PARSE_API_KEY = os.getenv("LLAMA_PARSE_API_KEY", "")

# --- API Configuration ---
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")

# --- Retrieval Configuration ---
TEXT_SIMILARITY_TOP_K = 10  # Top-K from dense retriever (more candidates for reranker)
SPARSE_TOP_K = 10           # Top-K from sparse (BM25) retriever
IMAGE_SIMILARITY_TOP_K = 5
RRF_K = 60                  # RRF constant (standard value)
FINAL_RERANK_TOP_N = 7      # Total results after reranking (text + images)
IMAGE_RESULT_SLOTS = 2      # Guaranteed image slots in final results (rest = text)

# --- Semantic Response Cache ---
RESPONSE_CACHE_COLLECTION = "response_cache"
SEMANTIC_CACHE_THRESHOLD = 0.80  # Cosine similarity — matches paraphrased questions
SEMANTIC_CACHE_TTL_HOURS = 1     # Cached responses expire after this
SEMANTIC_CACHE_MAX_ENTRIES = 200  # Max cached responses in Qdrant

# --- Image Indexing Configuration ---
URL_BASED_IMAGE_INDEXING = os.getenv("URL_BASED_IMAGE_INDEXING", "false").lower() == "true"
# When True: Embed images directly from LlamaParse URLs (faster, no storage)
# When False: Download images and store them (Supabase or local)

SUPABASE_STORAGE_ENABLED = os.getenv("SUPABASE_STORAGE_ENABLED", "true").lower() == "true"
# When True: Upload downloaded images to Supabase storage bucket
# When False: Save images to local downloaded_images/ folder

# --- Text Chunking Configuration ---
CHUNK_SIZE = 512            # Tokens per chunk (smaller = more precise retrieval with bge-small)
CHUNK_OVERLAP = 50          # Token overlap between adjacent chunks

# --- Validation ---
def validate_config():
    """Validate that required configuration is present."""
    errors = []
    
    if not LLAMA_PARSE_API_KEY or LLAMA_PARSE_API_KEY.startswith("llx-..."):
        errors.append("LLAMA_PARSE_API_KEY is not set. Set it in .env file.")
    
    if not GROQ_API_KEY:
        errors.append("GROQ_API_KEY is not set. Set it in .env file.")
    
    if errors:
        print("[WARN]  Configuration Errors:")
        for err in errors:
            print(f"   - {err}")
        return False
    
    return True
