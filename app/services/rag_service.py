import io
import os
from pypdf import PdfReader
from fastembed import TextEmbedding
from pymongo import MongoClient

# Cliente global de MongoDB Atlas (reutiliza el pool de conexiones)
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
client = MongoClient(MONGODB_URI)
db = client["ai_alexis_db"]
knowledge_collection = db["knowledge_base"]

# Modelo de embeddings de 384 dimensiones
embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")


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
    """Extrae texto de documentos PDF."""
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        full_text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                full_text += extracted + "\n"
        return full_text.strip()
    except Exception as e:
        print(f"❌ Error extrayendo PDF: {e}")
        return ""


def index_document_content(user_id: str, filename: str, text: str) -> int:
    """Fragmenta, vectoriza y almacena el contenido en Atlas."""
    try:
        chunks = chunk_text(text)
        if not chunks:
            return 0

        embeddings = list(embedding_model.embed(chunks))
        docs = [
            {
                "user_id": user_id,
                "filename": filename,
                "text": chunk,
                "embedding": vector.tolist()
            }
            for chunk, vector in zip(chunks, embeddings)
        ]
        
        # Elimina versiones anteriores del mismo archivo para el usuario
        knowledge_collection.delete_many({"user_id": user_id, "filename": filename})
        knowledge_collection.insert_many(docs)
        return len(docs)
    except Exception as e:
        print(f"❌ Error indexando contenido: {e}")
        return 0


def vector_search(user_id: str, query: str, limit: int = 3) -> str:
    """Ejecuta búsqueda vectorial semántica en Atlas."""
    try:
        query_vector = list(embedding_model.embed([query]))[0].tolist()

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
        print(f"⚠️ Error en búsqueda vectorial: {e}")
        return ""