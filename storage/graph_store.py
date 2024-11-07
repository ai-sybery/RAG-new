# storage/graph_store.py

from neo4j import GraphDatabase
from typing import List, Dict, Optional
from utils.config import config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GraphStore:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            config.NEO4J_URI,
            auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
        )

    def close(self):
        self.driver.close()

    async def create_knowledge_graph(self, entities: List[Dict], relations: List[Dict]) -> None:
        """
        Создает граф знаний из извлеченных сущностей и отношений
        """
        try:
            with self.driver.session() as session:
                # Создаем сущности
                for entity in entities:
                    session.run("""
                        MERGE (e:Entity {id: $id})
                        SET e.type = $type,
                            e.value = $value,
                            e.source = $source,
                            e.position = $position
                    """, entity)

                # Создаем отношения
                for relation in relations:
                    session.run("""
                        MATCH (source:Entity {id: $source_id})
                        MATCH (target:Entity {id: $target_id})
                        MERGE (source)-[r:RELATES {type: $relation_type}]->(target)
                        SET r.confidence = $confidence,
                            r.context = $context
                    """, relation)

            logger.info("Successfully created knowledge graph")

        except Exception as e:
            logger.error(f"Error creating knowledge graph: {str(e)}")
            raise

    async def search_graph(self, query_entities: List[str], max_depth: int = 2) -> List[Dict]:
        """
        Поиск в графе по сущностям
        """
        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH path = (start:Entity)-[*1..{max_depth}]-(connected:Entity)
                    WHERE start.value IN $query_entities
                    RETURN path, 
                           [node in nodes(path) | node.value] as entity_values,
                           [rel in relationships(path) | type(rel)] as relation_types
                    LIMIT 10
                """, {"query_entities": query_entities, "max_depth": max_depth})

                paths = []
                for record in result:
                    paths.append({
                        "entities": record["entity_values"],
                        "relations": record["relation_types"]
                    })

                return paths

        except Exception as e:
            logger.error(f"Error searching graph: {str(e)}")
            raise