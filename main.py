# main.py

import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Optional
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from processors.document_processor import DocumentProcessor
from processors.embedding_processor import EmbeddingProcessor
from storage.vector_store import VectorStore
from storage.graph_store import GraphStore
from retrieval.hybrid_retriever import HybridRetriever
from qa.gemini_handler import GeminiHandler
from qa.chat_manager import ChatManager

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG Chatbot API")

# Модели данных
class Message(BaseModel):
    session_id: str
    query: str
    entities: Optional[List[str]] = None

class DocumentUpload(BaseModel):
    file_path: str

# Инициализация компонентов
async def init_components():
    try:
        doc_processor = DocumentProcessor()
        embedding_processor = EmbeddingProcessor()
        vector_store = VectorStore()
        graph_store = GraphStore()
        
        retriever = HybridRetriever(
            vector_store=vector_store,
            graph_store=graph_store,
            embedding_processor=embedding_processor
        )
        
        gemini_handler = GeminiHandler()
        chat_manager = ChatManager(
            retriever=retriever,
            gemini_handler=gemini_handler
        )
        
        return {
            "doc_processor": doc_processor,
            "embedding_processor": embedding_processor,
            "vector_store": vector_store,
            "graph_store": graph_store,
            "retriever": retriever,
            "chat_manager": chat_manager
        }
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        raise

# Глобальные компоненты
components = {}

@app.on_event("startup")
async def startup_event():
    """Инициализация при запуске"""
    global components
    components = await init_components()

@app.post("/upload")
async def upload_document(doc: DocumentUpload):
    """Загрузка и обработка документа"""
    try:
        # Проверяем существование файла
        file_path = Path(doc.file_path)
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")

        # Обрабатываем документ
        processed = await components["doc_processor"].process_document(str(file_path))
        processed = await components["embedding_processor"].create_embeddings(processed)
        
        # Сохраняем в хранилища
        await components["vector_store"].add_documents(processed)
        
        return {"status": "success", "message": "Document processed successfully"}
        
    except Exception as e:
        logger.error(f"Error processing document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat(message: Message):
    """Обработка сообщений чата"""
    try:
        response = await components["chat_manager"].process_message(
            session_id=message.session_id,
            query=message.query,
            entities=message.entities
        )
        
        return {
            "status