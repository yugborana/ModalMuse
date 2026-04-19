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

# --- Jina AI Embeddings Configuration ---
JINA_API_KEY = os.getenv("JINA_API_KEY", "")
JINA_EMBED_MODEL = "jina-embeddings-v4"
JINA_EMBED_DIMENSIONS = 1024  # Can be 256-2048, lower = faster/cheaper
JINA_RERANK_MODEL = "jina-reranker-v2-base-multilingual"

# Image embedding batch configuration (Jina free tier: 100K tokens/min)
JINA_IMAGE_BATCH_SIZE = 2      # Images per API batch (keep small — images are token-heavy)
JINA_BATCH_DELAY_SECONDS = 20   # Delay between batches to avoid rate limit (60s window + buffer)


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
TEXT_SIMILARITY_TOP_K = 7   # Top-K from dense (Jina) retriever
SPARSE_TOP_K = 7            # Top-K from sparse (BM25) retriever
IMAGE_SIMILARITY_TOP_K = 5
RRF_K = 60                  # RRF constant (standard value)
FINAL_RERANK_TOP_N = 5      # Top-N after reranking fused results

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

# --- Validation ---
def validate_config():
    """Validate that required configuration is present."""
    errors = []
    
    if not LLAMA_PARSE_API_KEY or LLAMA_PARSE_API_KEY.startswith("llx-..."):
        errors.append("LLAMA_PARSE_API_KEY is not set. Set it in .env file.")
    
    if not GROQ_API_KEY:
        errors.append("GROQ_API_KEY is not set. Set it in .env file.")
    
    if not JINA_API_KEY:
        errors.append("JINA_API_KEY is not set. Get one at jina.ai/embeddings")
    
    if errors:
        print("[WARN]  Configuration Errors:")
        for err in errors:
            print(f"   - {err}")
        return False
    
    return True
