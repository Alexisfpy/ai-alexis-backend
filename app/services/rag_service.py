import os
import io
from pypdf import PdfReader
from pymongo import MongoClient
from google import genai

# --- CONEXIÓN MONGODB ATLAS ---
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
client = MongoClient(MONGODB_URI)
db = client["ai_alexis_db"]
knowledge_collection = db["knowledge_base"]

EMBEDDING_MODEL = "gemini-embedding-001"


def get_gemini_client():
    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        raise ValueError("GEMINI_API_KEY no configurada en las variables de entorno.")
    return genai.Client(api_key=gemini_key)


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 40) -> list[str]:
    """Divide el texto en fragmentos con solapamiento."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks if chunks else [text]


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extrae texto de un archivo PDF."""
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        full_text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                full_text += extracted + "\n"
        return full_text.strip()
    except Exception as e:
        print(f"❌ Error al extraer PDF: {e}")
        return ""


def index_document_content(user_id: str, filename: str, text: str) -> int:
    """Fragmenta, genera embeddings en la nube de Google y guarda en Atlas."""
    try:
        chunks = chunk_text(text)
        if not chunks:
            return 0

        ai_client = get_gemini_client()
        docs = []

        for chunk in chunks:
            res = ai_client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=chunk
            )
            
            # Extraer vector de 768 dimensiones
            if hasattr(res, "embedding") and res.embedding:
                vector = list(res.embedding.values)
            elif hasattr(res, "embeddings") and res.embeddings:
                vector = list(res.embeddings[0].values)
            else:
                continue

            docs.append({
                "user_id": user_id,
                "filename": filename,
                "text": chunk,
                "embedding": vector
            })

        if docs:
            knowledge_collection.delete_many({"user_id": user_id, "filename": filename})
            knowledge_collection.insert_many(docs)

        return len(docs)
    except Exception as e:
        print(f"❌ Error al indexar documento con Gemini API: {e}")
        return 0


def vector_search(user_id: str, query: str, limit: int = 3) -> str:
    """Realiza la búsqueda semántica en Atlas usando el embedding de la consulta."""
    try:
        ai_client = get_gemini_client()
        res = ai_client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=query
        )

        if hasattr(res, "embedding") and res.embedding:
            query_vector = list(res.embedding.values)
        elif hasattr(res, "embeddings") and res.embeddings:
            query_vector = list(res.embeddings[0].values)
        else:
            return ""

        pipeline = [
            {
                "$vectorSearch": {
                    "index": "vector_index",
                    "path": "embedding",
                    "queryVector": query_vector,
                    "numCandidates": limit * 10,
                    "limit": limit,
                    "filter": {
                        "user_id": {"$eq": user_id}
                    }
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "text": 1,
                    "filename": 1,
                    "score": {"$meta": "vectorSearchScore"}
                }
            }
        ]

        results = list(knowledge_collection.aggregate(pipeline))
        if not results:
            return ""

        return "\n\n".join([f"📄 [Doc: {r.get('filename', 'Archivo')}]: {r['text']}" for r in results])
    except Exception as e:
        print(f"⚠️ Error en vector_search: {e}")
        return ""