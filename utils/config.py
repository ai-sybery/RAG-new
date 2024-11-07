# utils/config.py

from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class Config:
    # Document Processing
    SUPPORTED_FORMATS = ['.pdf', '.txt']
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    
    # Embeddings
    EMBEDDING_MODEL = "text-multilingual-embedding-002"
    MAX_CHUNK_SIZE = 2048
    CHUNK_OVERLAP = 200
    
    # Vector Store
    CHROMA_PERSIST_DIR = "./data/chroma"
    COLLECTION_NAME = "documents"
    
    # Neo4j
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "password"
    
    # Gemini
    GEMINI_CONFIG: Dict[str, Any] = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8192
    }
    
    # Retrieval
    TOP_K_VECTORS = 10
    RERANKING_THRESHOLD = 0.7

config = Config()