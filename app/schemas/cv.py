from pydantic import BaseModel
from typing import Optional

class CVUpdateRequest(BaseModel):
    cv_text: str
    job_description: str
    groq_api_key: Optional[str] = None  # Si se deja vacío, usará la del servidor

class CVUpdateResponse(BaseModel):
    optimized_cv: str