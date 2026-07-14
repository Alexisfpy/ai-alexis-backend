from pydantic import BaseModel
from typing import Optional, List, Dict

class ChatMessage(BaseModel):
    role: str  # "user" o "assistant"
    content: str

class AssistantRequest(BaseModel):
    message: str
    history: Optional[List[ChatMessage]] = []
    groq_api_key: Optional[str] = None

class AssistantResponse(BaseModel):
    intent: str
    response: str