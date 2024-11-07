# qa/chat_manager.py

from typing import List, Dict, Optional
from dataclasses import dataclass
import asyncio
from datetime import datetime
import json
import logging
from pathlib import Path

from .gemini_handler import GeminiHandler, ChatMessage
from retrieval.hybrid_retriever import HybridRetriever

logger = logging.getLogger(__name__)

@dataclass
class ChatSession:
    """Сессия чата"""
    session_id: str
    start_time: datetime
    messages: List[ChatMessage]
    metadata: Dict

class ChatManager:
    """Управление чат-сессиями и взаимодействием"""
    
    def __init__(
        self,
        retriever: HybridRetriever,
        gemini_handler: GeminiHandler,
        session_dir: str = "chat_sessions"
    ):
        self.retriever = retriever
        self.gemini_handler = gemini_handler
        self.session_dir = Path(session_dir)
        self.session_dir.mkdir(exist_ok=True)
        self.active_sessions: Dict[str, ChatSession] = {}

    async def create_session(self, session_id: str, metadata: Dict = None) -> ChatSession:
        """Создает новую сессию чата"""
        session = ChatSession(
            session_id=session_id,
            start_time=datetime.now(),
            messages=[],
            metadata=metadata or {}
        )
        self.active_sessions[session_id] = session
        await self._save_session(session)
        return session

    async def process_message(
        self,
        session_id: str,
        query: str,
        entities: List[str] = None
    ) -> str:
        """
        Обрабатывает сообщение пользователя
        
        Args:
            session_id: ID сессии
            query: Вопрос пользователя
            entities: Список сущностей для поиска
            
        Returns:
            str: Ответ системы
        """
        try:
            session = self.active_sessions.get(session_id)
            if not session:
                session = await self.create_session(session_id)

            # Получаем релевантный контекст
            context = await self.retriever.retrieve(query, entities or [])

            # Генерируем ответ
            response = await self.gemini_handler.generate_response(
                query=query,
                context=context,
                chat_history=session.messages
            )

            # Валидируем ответ
            if not self.gemini_handler.validate_response(response):
                response = "Извините, не удалось сгенерировать качественный ответ. Попробуйте переформулировать вопрос."

            # Сохраняем сообщения
            session.messages.extend([
                ChatMessage(role="user", content=query, context={"entities": entities}),
                ChatMessage(role="assistant", content=response, context={"retrieved": context})
            ])

            await self._save_session(session)
            return response

        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return "Произошла ошибка при обработке сообщения. Попробуйте позже."

    async def _save_session(self, session: ChatSession) -> None:
        """Сохраняет сессию в файл"""
        try:
            session_path = self.session_dir / f"{session.session_id}.json"
            session_data = {
                "session_id": session.session_id,
                "start_time": session.start_time.isoformat(),
                "messages": [
                    {
                        "role": msg.role,
                        "content": msg.content,
                        "context": msg.context
                    }
                    for msg in session.messages
                ],
                "metadata": session.metadata
            }
            
            async with aiofiles.open(session_path, 'w') as f:
                await f.write(json.dumps(session_data, indent=2, ensure_ascii=False))
                
        except Exception as e:
            logger.error(f"Error saving session: {e}")