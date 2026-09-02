import os
from pathlib import Path
from pymongo import MongoClient
from dotenv import load_dotenv

# Al estar dentro de app/core/, subimos 3 niveles para llegar a la raíz donde está el .env
ruta_raiz = Path(__file__).resolve().parent.parent.parent
ruta_env = ruta_raiz / ".env"
load_dotenv(dotenv_path=ruta_env)

MONGODB_URI = os.getenv("MONGODB_URI")
if not MONGODB_URI:
    MONGODB_URI = "mongodb://localhost:27017"

# Conexión única compartida a MongoDB Atlas
client = MongoClient(MONGODB_URI)
db = client["ai_alexis_db"]

# Colecciones
profiles_collection = db["profiles"]
history_collection = db["chat_history"]
conversations_collection = db["conversations"]
google_tokens_collection = db["google_user_tokens"]