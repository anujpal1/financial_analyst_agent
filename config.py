import os

# Target Execution Engine
MODEL_NAME = "llama3.1:8b"

# Hardware Constraints for i5-12500H & 8GB RAM
OLLAMA_NUM_THREADS = 4       # Commit to 4 primary execution threads
MODEL_TEMPERATURE = 0.0      # Lock output distribution to absolute factual correctness
MAX_CONTEXT_TOKENS = 4096    # Restrict token depth to secure system RAM headroom

# Storage References
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VECTOR_DB_DIR = os.path.join(BASE_DIR, "memory", "chroma_db")

# Add these lines to your existing config.py file

EMBEDDING_MODEL_NAME = "nomic-embed-text"
VECTOR_DB_DIR = "chroma_db"