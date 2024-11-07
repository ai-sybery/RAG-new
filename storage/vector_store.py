# storage/vector_store.py

import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional
import numpy as np
from utils.config import config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self):
        self.client = chromadb.Client(Settings(
            persist_directory=config.CHROMA_PERSIST_DIR,
            anonymized_telemetry=False
        ))
        
        # Создаем или получаем коллекцию
        self.collection = self.client.get_or_create_collection(
            name=config.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )

    async def add_documents(self, processed_elements: List[Dict]) -> None:
        """
        Добавляет документы в векторное хранилище
        """
        try:
            for element in processed_elements:
                for chunk in element['chunks']:
                    self.collection.add(
                        embeddings=chunk['embedding'].tolist(),
                        documents=chunk['content'],
                        metadatas={
                            "source_file": element['metadata']['source_file'],
                            "position": chunk['position'],
                            "element_type": element['metadata']['element_type']
                        },
                        ids=[f"{element['metadata']['source_file']}_{chunk['position']}"]
                    )
            
            logger.info(f"Successfully added {len(processed_elements)} documents to vector store")
            
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {str(e)}")
            raise

    async def search(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Dict]:
        """
        Поиск похожих документов
        """
        try:
            results = self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=top_k
            )
            
            return [
                {
                    "content": doc,
                    "metadata": metadata,
                    "distance": distance
                }
                for doc, metadata, distance in zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )
            ]
            
        except Exception as e:
            logger.error(f"Error searching vector store: {str(e)}")
            raise