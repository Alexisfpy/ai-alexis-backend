from dotenv import load_dotenv
# 1. Cargamos las variables de entorno lo primero de todo
load_dotenv()

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.patches.litellm_patch import apply_litellm_patch
from app.routers import cv, assistant

# 2. Aplicamos el parche de litellm
apply_litellm_patch()

# 3. Inicializamos la aplicación de FastAPI (Esto es lo que busca Uvicorn)
app = FastAPI(
    title="AI Alexis API",
    description="El cerebro asíncrono y distribuido de tu asistente personal multiplataforma.",
    version="0.1.0"
)

# 4. Configuración de CORS para acceso desde Móvil, Tablet y Web
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción cambia esto por tu dominio
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 5. Registro de las Rutas de los Agentes
app.include_router(cv.router, prefix="/api/v1")
app.include_router(assistant.router, prefix="/api/v1")
@app.get("/")
def read_root():
    return {
        "status": "online",
        "assistant": "AI Alexis",
        "system_time": "Operational",
        "message": "Sistemas listos y en línea, Alexis. Esperando comandos."
    }