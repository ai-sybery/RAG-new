# qa/gemini_handler.py

from typing import List, Dict, Optional
from dataclasses import dataclass
import google.generativeai as genai
from utils.config import config
import asyncio
import logging
from functools import lru_cache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ChatMessage:
    """Data class для сообщений чата"""
    role: str
    content: str
    context: Optional[Dict] = None

class GeminiHandler:
    """Обработчик взаимодействия с Gemini API"""
    
    def __init__(self):
        try:
            genai.configure(api_key='YOUR_GEMINI_API_KEY')
            self.model = genai.GenerativeModel(
                model_name="gemini-pro",
                generation_config=config.GEMINI_CONFIG
            )
            self.chat = self.model.start_chat(history=[])
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {e}")
            raise

    @lru_cache(maxsize=1000)
    def _format_context(self, context: List[Dict]) -> str:
        """
        Форматирует контекст для модели
        
        Args:
            context: Список документов с контекстом
            
        Returns:
            str: Отформатированный контекст
        """
        formatted_context = []
        for doc in context:
            formatted_context.append(
                f"Source: {doc['metadata'].get('source_file', 'Unknown')}\n"
                f"Content: {doc['content']}\n"
                f"Relevance: {doc.get('score', 0):.2f}\n"
                "---"
            )
        return "\n".join(formatted_context)

    async def generate_response(
        self,
        query: str,
        context: List[Dict],
        chat_history: List[ChatMessage]
    ) -> str:
        """
        Генерирует ответ на основе запроса и контекста
        
        Args:
            query: Вопрос пользователя
            context: Релевантный контекст
            chat_history: История чата
            
        Returns:
            str: Сгенерированный ответ
        """
        try:
            # Формируем промпт
            formatted_context = self._format_context(context)
            system_prompt = (
                "Ты - помощник по документации. Используй предоставленный контекст "
                "для ответа на вопросы. Если информации недостаточно, честно "
                "признай это. Отвечай структурированно и по существу."
            )
            
            # Добавляем историю чата и контекст
            messages = []
            messages.append({"role": "system", "content": system_prompt})
            for msg in chat_history[-5:]:  # Берем последние 5 сообщений
                messages.append({"role": msg.role, "content": msg.content})
            
            # Формируем финальный промпт
            full_prompt = (
                f"Context:\n{formatted_context}\n\n"
                f"Question: {query}\n\n"
                "Please provide a clear and structured answer based on the context above."
            )
            
            # Генерируем ответ
            response = await asyncio.to_thread(
                self.chat.send_message,
                full_prompt
            )
            
            return response.text

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "Извините, произошла ошибка при генерации ответа. Попробуйте позже."

    def validate_response(self, response: str) -> bool:
        """
        Проверяет качество ответа
        
        Args:
            response: Сгенерированный ответ
            
        Returns:
            bool: Результат валидации
        """
        if not response or len(response) < 10:
            return False
        if "извините" in response.lower() and "ошибка" in response.lower():
            return False
        return True