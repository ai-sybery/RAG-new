# retrieval/hybrid_retriever.py

from typing import List, Dict, Tuple
import asyncio
from utils.config import config
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HybridRetriever:
    def __init__(self, vector_store, graph_store, embedding_processor):
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.embedding_processor = embedding_processor

    async def retrieve(self, query: str, entities: List[str]) -> List[Dict]:
        """
        Выполняет гибридный поиск, комбинируя векторный и графовый поиск
        """
        try:
            # Параллельный запуск поисков
            vector_results, graph_results = await asyncio.gather(
                self._vector_search(query),
                self._graph_search(entities)
            )

            # Объединение и ранжирование результатов
            combined_results = await self._merge_results(
                vector_results,
                graph_results,
                query
            )

            return combined_results

        except Exception as e:
            logger.error(f"Error in hybrid retrieval: {str(e)}")
            raise

    async def _vector_search(self, query: str) -> List[Dict]:
        """
        Выполняет векторный поиск
        """
        query_embedding = await self.embedding_processor.create_embeddings([{"content": query}])
        return await self.vector_store.search(
            query_embedding[0]['chunks'][0]['embedding'],
            top_k=config.TOP_K_VECTORS
        )

    async def _graph_search(self, entities: List[str]) -> List[Dict]:
        """
        Выполняет поиск по графу
        """
        return await self.graph_store.search_graph(entities)

    async def _merge_results(
        self,
        vector_results: List[Dict],
        graph_results: List[Dict],
        query: str
    ) -> List[Dict]:
        """
        Объединяет и ранжирует результаты поиска
        """
        # Здесь можно добавить более сложную логику ранжирования
        merged = []
        
        # Добавляем векторные результаты
        for vr in vector_results:
            merged.append({
                "content": vr["content"],
                "metadata": vr["metadata"],
                "score": 1 - vr["distance"],  # Конвертируем дистанцию в score
                "source": "vector"
            })

        # Добавляем графовые результаты
        for gr in graph_results:
            merged.append({
                "content": str(gr["entities"]),
                "metadata": {"relations": gr["relations"]},
                "score": 0.5,  # Базовый скор для графовых результатов
                "source": "graph"
            })

        # Сортируем по score
        merged.sort(key=lambda x: x["score"], reverse=True)
        
        return merged