# config.py - Centralized Configuration for ModalMuse
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Base Directories ---
BASE_DIR = Path(__file__).parent.resolve()
LLMS_DIR = BASE_DIR / "LLMs"
DATA_DIR = BASE_DIR / "data"

# --- LLM Configuration (Groq) ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL_NAME = "meta-llama/llama-4-maverick-17b-128e-instruct"

# LLM Parameters
LLM_MAX_NEW_TOKENS = 1024
LLM_TEMPERATURE = 0.7

# --- Jina AI Embeddings Configuration ---
JINA_API_KEY = os.getenv("JINA_API_KEY", "")
JINA_EMBED_MODEL = "jina-embeddings-v4"
JINA_EMBED_DIMENSIONS = 1024  # Can be 256-2048, lower = faster/cheaper
JINA_RERANK_MODEL = "jina-reranker-v2-base-multilingual"

# --- Legacy Local Models (kept for reference) ---
# BGE_MODEL_NAME = "BAAI/bge-small-en-v1.5" 
# SPLADE_MODEL_NAME = "prithivida/Splade_PP_en_v1"
# CLIP_EMBED_MODEL_NAME = "openai/clip-vit-base-patch32"
# RERANKER_MODEL = "BAAI/bge-reranker-base"

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
TEXT_SIMILARITY_TOP_K = 7  # Reduced from 10 for faster reranking
TEXT_SPARSE_TOP_K = 7
TEXT_RERANK_TOP_N = 4  # Reduced from 5
IMAGE_SIMILARITY_TOP_K = 3
FINAL_TOP_K = 8  # Reduced from 10

# --- RRF (Reciprocal Rank Fusion) Configuration ---
RRF_K = 60  # Standard RRF constant (higher = more weight to lower-ranked items)
RRF_WEIGHTS = [0.6, 0.4]  # Weights for [dense, sparse] retrieval

# --- Multi-Modal RRF Configuration ---
MULTIMODAL_RRF_ENABLED = True  # Enable RRF fusion between text and images
MULTIMODAL_RRF_WEIGHTS = [0.7, 0.3]  # Weights for [text, images]

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
