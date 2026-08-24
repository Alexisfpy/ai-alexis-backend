from pydantic import BaseModel
from typing import Optional, List, Dict

class ChatMessage(BaseModel):
    role: str  # "user" o "assistant"
    content: str
    image: Optional[str] = None  # Cadena Base64 opcional

class AssistantRequest(BaseModel):
    message: str
    history: Optional[List[ChatMessage]] = []
    groq_api_key: Optional[str] = None
    user_id: Optional[str] = "guest_user"
    conversation_id: Optional[str] = None  # Identificador de sesión/chat activo
    cv_text: Optional[str] = "" # Añadido campo para la transición
    image: Optional[str] = None  # Imagen en formato Base64 enviada desde el frontend
class AssistantResponse(BaseModel):
    intent: str
    response: str
    conversation_id: Optional[str] = None  # Devuelve el ID (creado o mantenido)