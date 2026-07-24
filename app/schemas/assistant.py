from pydantic import BaseModel
from typing import Optional, List, Dict

class ChatMessage(BaseModel):
    role: str  # "user" o "assistant"
    content: str

class AssistantRequest(BaseModel):
    message: str
    history: Optional[List[ChatMessage]] = []
    groq_api_key: Optional[str] = None
    user_id: Optional[str] = "alexis_perez_123"
    cv_text: Optional[str] = "" # Añadido campo para la transición
class AssistantResponse(BaseModel):
    intent: str
    response: str