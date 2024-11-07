# processors/embedding_processor.py

from typing import List, Dict
import google.generativeai as genai
from utils.config import config
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingProcessor:
    def __init__(self):
        self.model = config.EMBEDDING_MODEL
        self.max_chunk_size = config.MAX_CHUNK_SIZE
        self.chunk_overlap = config.CHUNK_OVERLAP
        
        # Initialize Gemini
        genai.configure(api_key='YOUR_GEMINI_API_KEY')
        
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into semantic chunks with overlap
        """
        if len(text) <= self.max_chunk_size:
            return [text]
            
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.max_chunk_size
            
            # Find the nearest sentence end
            if end < len(text):
                while end > start and text[end] not in '.!?':
                    end -= 1
                if end == start:
                    end = start + self.max_chunk_size
                    
            chunks.append(text[start:end])
            start = end - self.chunk_overlap
            
        return chunks
        
    async def create_embeddings(self, elements: List[Dict]) -> List[Dict]:
        """
        Create embeddings for document elements
        """
        try:
            processed_elements = []
            
            for element in elements:
                content = element['content']
                chunks = self.chunk_text(content)
                
                chunk_embeddings = []
                for chunk in chunks:
                    embedding = genai.embed_content(
                        model=self.model,
                        content=chunk,
                        task_type="retrieval_document"
                    )
                    chunk_embeddings.append({
                        "content": chunk,
                        "embedding": embedding,
                        "position": element['position']
                    })
                
                element['chunks'] = chunk_embeddings
                processed_elements.append(element)
                
            return processed_elements
            
        except Exception as e:
            logger.error(f"Error creating embeddings: {str(e)}")
            raise