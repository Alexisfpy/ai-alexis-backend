import io
import os
from pypdf import PdfReader
from fastembed import TextEmbedding
from pymongo import MongoClient

# Modelo optimizado para CPU que genera embeddings de 384 dimensiones
embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

def get_knowledge_collection():
    mongodb_uri = os.getenv("MONGODB_URI")
    client = MongoClient(mongodb_uri)
    db = client["ai_alexis_db"]
    return db["knowledge_base"]

def chunk_text(text: str, chunk_size: int = 400, overlap: int = 40) -> list[str]:
    """Divide el texto en fragmentos con solapamiento para no perder contexto."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extrae el contenido de texto de un archivo PDF."""
    reader = PdfReader(io.BytesIO(file_bytes))
    full_text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            full_text += extracted + "\n"
    return full_text

def index_document_content(user_id: str, filename: str, text: str) -> int:
    """Fragmenta, vectoriza y guarda el documento en MongoDB."""
    collection = get_knowledge_collection()
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
    collection.insert_many(docs)
    return len(docs)

def vector_search(user_id: str, query: str, limit: int = 3) -> str:
    """Realiza la búsqueda por similitud semántica en Atlas Vector Search."""
    collection = get_knowledge_collection()
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

    try:
        results = list(collection.aggregate(pipeline))
        if not results:
            return ""
        return "\n\n".join([f"📄 [Doc: {r.get('filename', 'Archivo')}]: {r['text']}" for r in results])
    except Exception as e:
        print(f"Error en búsqueda vectorial: {e}")
        return ""