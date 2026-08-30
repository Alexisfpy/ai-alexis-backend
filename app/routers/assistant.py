import os
import io
import litellm
import re
import json
import time
import uuid
import base64
import asyncio
from typing import Optional, Any
from datetime import datetime
from pathlib import Path

import httpx
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Query
from fastapi.responses import StreamingResponse, Response, PlainTextResponse
from pydantic import BaseModel
from pymongo import MongoClient
from dotenv import load_dotenv
from duckduckgo_search import DDGS
from crewai import Agent, Task, Crew, Process, LLM
from pypdf import PdfReader
from tavily import TavilyClient
from google import genai
from google.genai import types

from app.schemas.assistant import AssistantRequest, AssistantResponse
from app.services.weather_service import consultar_clima_open_meteo
from app.services.news_service import buscar_noticias
from app.services.rag_service import extract_text_from_pdf, index_document_content, vector_search
from app.services.tts_service import sintetizar_voz_neural

# --- CONFIGURACIÓN DE MODELOS ---
TEXT_MODEL = "groq/openai/gpt-oss-120b"
VISION_MODEL = "gemini-3.6-flash"

# RUTA ABSOLUTA AL ARCHIVO .env
ruta_raiz = Path(__file__).resolve().parent.parent.parent
ruta_env = ruta_raiz / ".env"
load_dotenv(dotenv_path=ruta_env)

router = APIRouter(prefix="/assistant", tags=["AI Alexis Assistant"])

# --- CONEXIÓN A MONGODB ATLAS ---
MONGODB_URI = os.getenv("MONGODB_URI")
if not MONGODB_URI:
    MONGODB_URI = "mongodb://localhost:27017"

client = MongoClient(MONGODB_URI)
db = client["ai_alexis_db"]
profiles_collection = db["profiles"]
history_collection = db["chat_history"]
conversations_collection = db["conversations"]

# --- CACHÉ EN MEMORIA CON TTL ---
_cache = {}
CACHE_TTL = {
    "weather": 600,   # 10 minutos
    "news": 300,      # 5 minutos
    "web": 1800       # 30 minutos
}

def get_cached(key: str, cache_type: str) -> Optional[Any]:
    """Obtiene un valor de la caché si no ha expirado."""
    if key in _cache:
        data, timestamp = _cache[key]
        if time.time() - timestamp < CACHE_TTL.get(cache_type, 300):
            return data
        else:
            del _cache[key]
    return None

def set_cached(key: str, value: Any, cache_type: str):
    """Guarda un valor en caché con timestamp."""
    _cache[key] = (value, time.time())

# --- HELPERS: GESTIÓN DE HISTORIALES Y SESIONES ---
def guardar_en_historial(user_id: str, user_text: str, bot_response: str, intent: str):
    """Guarda mensajes en la colección general chat_history."""
    try:
        history_collection.update_one(
            {"_id": user_id},
            {
                "$push": {
                    "messages": {
                        "$each": [
                            {"role": "user", "content": user_text},
                            {"role": "assistant", "content": bot_response, "intent": intent}
                        ]
                    }
                }
            },
            upsert=True
        )
    except Exception as e:
        print(f"⚠️ Error guardando en chat_history: {e}")

def guardar_en_conversacion(user_id: str, conversation_id: str, user_text: str, bot_response: str, intent: str) -> str:
    """Guarda o actualiza mensajes dentro de una sesión de chat específica en Atlas."""
    if not conversation_id:
        conversation_id = str(uuid.uuid4())

    titulo_inicial = user_text[:30] + ("..." if len(user_text) > 30 else "")

    conversations_collection.update_one(
        {"_id": conversation_id},
        {
            "$setOnInsert": {
                "user_id": user_id,
                "title": titulo_inicial,
                "created_at": datetime.utcnow()
            },
            "$set": {"updated_at": datetime.utcnow()},
            "$push": {
                "messages": {
                    "$each": [
                        {"role": "user", "content": user_text},
                        {"role": "assistant", "content": bot_response, "intent": intent}
                    ]
                }
            }
        },
        upsert=True
    )
    return conversation_id

async def generar_titulo_inteligente(user_text: str, bot_response: str, api_key: str) -> str:
    """Genera un título conciso de 3 a 5 palabras mediante el LLM."""
    try:
        prompt = (
            "Genera un título descriptivo y conciso de 3 a 5 palabras en español "
            "para identificar esta conversación en una lista lateral.\n"
            "REGLA ESTRICTA: Responde ÚNICAMENTE con el título, sin comillas, sin puntos finales y sin explicaciones.\n\n"
            f"Usuario: {user_text[:250]}\n"
            f"Asistente: {bot_response[:250]}"
        )
        res = await litellm.acompletion(
            model=TEXT_MODEL,
            api_key=api_key,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=25
        )
        titulo = res.choices[0].message.content.strip().replace('"', '').replace("'", "")
        return titulo[:45] if titulo else user_text[:30]
    except Exception:
        return user_text[:30] + ("..." if len(user_text) > 30 else "")

# --- PERFIL PROFESIONAL BASE ---
BASE_CV = """
Nombre Completo: Alexis Fernando Pérez Yamasque
Ubicación: Vigo, Galicia
Perfil Profesional: 
Ingeniero de IA y Estudiante de Big Data con sólida formación en desarrollo multiplataforma y administración de sistemas. Especializado en el diseño, entrenamiento y despliegue de modelos de Machine Learning y Deep Learning.
"""

def extraer_texto_pdf(contenido_bytes: bytes) -> str:
    try:
        fichero_pdf = io.BytesIO(contenido_bytes)
        lector = PdfReader(fichero_pdf)
        texto = ""
        for pagina in lector.pages:
            texto_pagina = pagina.extract_text()
            if texto_pagina:
                texto += texto_pagina + "\n"
        return texto.strip()
    except Exception as e:
        raise Exception(f"No se pudo decodificar el PDF: {str(e)}")

def optimizar_query_busqueda(mensaje_usuario: str, api_key: str) -> str:
    try:
        prompt_optimizador = (
            "Tu único trabajo es convertir un mensaje conversacional en una consulta de búsqueda "
            "ultra-eficiente de 2 a 4 palabras clave para un buscador web.\n"
            "REGLA ESTRICTA: Responde ÚNICAMENTE con las palabras clave sugeridas, sin introducciones ni comillas."
        )
        response = litellm.completion(
            model=TEXT_MODEL,
            api_key=api_key,
            messages=[
                {"role": "system", "content": prompt_optimizador},
                {"role": "user", "content": mensaje_usuario}
            ],
            temperature=0.0
        )
        return response.choices[0].message.content.strip().replace('"', '').replace("'", "")
    except Exception:
        return mensaje_usuario

def buscar_en_internet(query: str) -> str:
    tavily_key = os.getenv("TAVILY_API_KEY")
    if tavily_key:
        try:
            tavily = TavilyClient(api_key=tavily_key)
            response = tavily.search(query=query, search_depth="basic", max_results=3)
            resultados = response.get("results", [])
            if resultados:
                contexto = ""
                for i, r in enumerate(resultados, 1):
                    contexto += f"[{i}] Fuente: {r.get('title')}\nInformación: {r.get('content')}\n\n"
                return contexto
        except Exception as e:
            print(f"⚠️ Error en Tavily API: {e}")

    try:
        with DDGS() as ddgs:
            resultados = list(ddgs.text(query, max_results=3))
            if resultados:
                contexto = ""
                for i, r in enumerate(resultados, 1):
                    titulo = r.get("title", "Sin título")
                    cuerpo = r.get("body", r.get("snippet", ""))
                    contexto += f"[{i}] Fuente: {titulo}\nInformación: {cuerpo}\n\n"
                return contexto
    except Exception as e:
        print(f"⚠️ Error en DuckDuckGo: {e}")

    return "No se encontraron resultados actualizados en la web."

# --- GENERADOR ASÍNCRONO DE STREAMING (SSE) ---
async def generar_stream_alexis(
    message: str,
    cv_texto: str,
    api_key: str,
    user_id: str = "guest_user",
    conversation_id: str = None,
    image_base64: str = None,
    rag_context: str = ""
):
    os.environ["GROQ_API_KEY"] = api_key
    os.environ["OPENAI_API_KEY"] = api_key

    conv_id = conversation_id or str(uuid.uuid4())
    texto_acumulado = ""
    intent_detectado = "GENERAL_CHAT"

    doc_previo = conversations_collection.find_one({"_id": conv_id})
    es_primer_mensaje = doc_previo is None or len(doc_previo.get("messages", [])) == 0

    try:
        # 1. ANÁLISIS DE IMAGEN CON GEMINI SDK
        if image_base64:
            intent_detectado = "IMAGE_ANALYSIS"
            yield f"data: {json.dumps({'intent': intent_detectado, 'conversation_id': conv_id, 'token': ''})}\n\n"

            gemini_key = os.getenv("GEMINI_API_KEY")
            if not gemini_key:
                err_msg = "⚠️ Falta configurar la variable GEMINI_API_KEY en el servidor."
                yield f"data: {json.dumps({'token': err_msg})}\n\n"
                guardar_en_conversacion(user_id, conv_id, message or "📸 [Imagen]", err_msg, intent_detectado)
                yield "data: [DONE]\n\n"
                return

            client = genai.Client(api_key=gemini_key)
            image_bytes = base64.b64decode(image_base64)
            texto_usuario = message.strip() if message and message.strip() else "Analiza y describe esta imagen en detalle."

            prompt_vision = (
                "Eres AI Alexis, un asistente virtual directo, técnico y elegante.\n"
                f"Consulta del usuario: {texto_usuario}\n"
                "Responde de forma concisa, profesional y en español."
            )
            response = client.models.generate_content(
                model=VISION_MODEL,
                contents=[prompt_vision, types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")]
            )
            texto_acumulado = response.text.strip()
            yield f"data: {json.dumps({'token': texto_acumulado})}\n\n"
            guardar_en_conversacion(user_id, conv_id, message or "📸 [Imagen]", texto_acumulado, intent_detectado)

            if es_primer_mensaje:
                titulo_nuevo = await generar_titulo_inteligente(message or "Análisis de Imagen", texto_acumulado, api_key)
                conversations_collection.update_one({"_id": conv_id}, {"$set": {"title": titulo_nuevo}})
                yield f"data: {json.dumps({'new_title': titulo_nuevo})}\n\n"

            yield "data: [DONE]\n\n"
            return

        # 2. CLASIFICACIÓN DE INTENCIONES
        mensaje_lower = message.lower()
        palabras_clima = ["tiempo", "clima", "temperatura", "grados", "lluvia", "meteorológico", "soleado", "nublado", "viento", "presión", "humedad"]

        if any(p in mensaje_lower for p in palabras_clima):
            intent_detectado = "WEATHER"
        else:
            system_prompt = (
                "Eres el clasificador de intenciones de AI Alexis.\n"
                "Responde ÚNICAMENTE con una de estas cinco palabras en mayúsculas:\n"
                "- 'WEATHER', 'SEARCH', 'CV_OPTIMIZATION', 'DOMOTICS_CONTROL', 'GENERAL_CHAT'."
            )
            clasif = await litellm.acompletion(
                model=TEXT_MODEL,
                api_key=api_key,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": message}],
                temperature=0.0
            )
            intent_detectado = re.sub(r'[^A-Z_]', '', clasif.choices[0].message.content.strip().upper()) or "GENERAL_CHAT"

        yield f"data: {json.dumps({'intent': intent_detectado, 'conversation_id': conv_id, 'token': ''})}\n\n"

        # 3. PREPARACIÓN DE CONTEXTO
        mensajes_para_llm = []

        if "WEATHER" in intent_detectado:
            res_loc = await litellm.acompletion(
                model=TEXT_MODEL,
                api_key=api_key,
                messages=[
                    {"role": "system", "content": "Extrae SOLO la ciudad/municipio. Si no hay, responde 'Vigo'."},
                    {"role": "user", "content": message}
                ],
                temperature=0.0
            )
            ubicacion = re.sub(r'[^a-zA-ZáéíóúÁÉÍÓÚñÑ\s]', '', res_loc.choices[0].message.content.strip()) or "Vigo"
            dias = 1 if ("mañana" in mensaje_lower or "manana" in mensaje_lower) else 0

            cache_key = f"weather_{ubicacion.lower()}_{dias}"
            datos_clima = get_cached(cache_key, "weather")
            if not datos_clima:
                datos_clima = await consultar_clima_open_meteo(ubicacion, dias=dias)
                set_cached(cache_key, datos_clima, "weather")

            prompt_clima = (
                "Eres AI Alexis (J.A.R.V.I.S.). Responde de forma clara y directa con los datos meteorológicos.\n"
                "Presenta la información en lista con viñetas (*), negritas en valores clave y sin enlaces externos.\n\n"
                f"DATOS OPEN-METEO (JSON):\n{json.dumps(datos_clima, ensure_ascii=False)}\n\n"
                f"Consulta: {message}"
            )
            mensajes_para_llm = [{"role": "user", "content": prompt_clima}]

        elif "SEARCH" in intent_detectado:
            palabras_noticias = ["noticia", "titular", "periódico", "actualidad", "última hora"]
            es_noticias = any(p in mensaje_lower for p in palabras_noticias)
            tema = optimizar_query_busqueda(message, api_key)

            if es_noticias:
                cache_key = f"news_{tema.lower()}"
                noticias_raw = get_cached(cache_key, "news")
                if not noticias_raw:
                    noticias_raw = await buscar_noticias(query=tema)
                    if not noticias_raw:
                        noticias_raw = buscar_en_internet(f"noticias actualidad {tema} 2026")
                    set_cached(cache_key, noticias_raw, "news")

                prompt_search = (
                    "Eres AI Alexis (J.A.R.V.I.S.). FECHA ACTUAL: Año 2026.\n"
                    "Presenta los titulares con viñetas (*), fuentes entre paréntesis y enlaces [Leer más](url).\n\n"
                    f"NOTICIAS:\n{noticias_raw}\n\nConsulta: {message}"
                )
            else:
                cache_key = f"web_{tema.lower()}"
                contexto_web = get_cached(cache_key, "web")
                if not contexto_web:
                    contexto_web = buscar_en_internet(tema)
                    set_cached(cache_key, contexto_web, "web")

                prompt_search = (
                    "Eres AI Alexis (J.A.R.V.I.S.). FECHA ACTUAL: Año 2026.\n"
                    "Responde basándote en estos datos en formato de lista con viñetas (*):\n\n"
                    f"INFORMACIÓN WEB:\n{contexto_web}\n\nConsulta: {message}"
                )
            mensajes_para_llm = [{"role": "user", "content": prompt_search}]

        elif "CV_OPTIMIZATION" in intent_detectado:
            extraction = await litellm.acompletion(
                model=TEXT_MODEL,
                api_key=api_key,
                messages=[{"role": "system", "content": "Extrae exclusivamente la descripción del puesto o requisitos."}, {"role": "user", "content": message}],
                temperature=0.1
            )
            oferta_laboral = extraction.choices[0].message.content.strip()

            llm_groq = LLM(model=TEXT_MODEL, api_key=api_key)
            reclutador = Agent(role="Reclutador Técnico", goal="Extraer requisitos clave ATS.", backstory="Experto en screening de perfiles tech.", verbose=False, llm=llm_groq)
            redactor = Agent(role="Redactor Técnico", goal="Adaptar perfil y CV del usuario.", backstory="Consultor y redactor de perfiles de ingeniería.", verbose=False, llm=llm_groq)

            tarea_analisis = Task(description=f"Oferta laboral:\n{oferta_laboral}", expected_output="Palabras clave y requisitos ATS.", agent=reclutador)
            tarea_adaptacion = Task(description=f"CV Base:\n{cv_texto}\nAdapta al puesto.", expected_output="Currículum adaptado en Markdown profesional.", agent=redactor)

            orquestador = Crew(agents=[reclutador, redactor], tasks=[tarea_analisis, tarea_adaptacion], process=Process.sequential, memory=False)
            cv_res = await asyncio.to_thread(orquestador.kickoff)

            texto_acumulado = f"He optimizado el perfil profesional con éxito:\n\n{str(cv_res)}"
            yield f"data: {json.dumps({'token': texto_acumulado})}\n\n"
            guardar_en_conversacion(user_id, conv_id, message, texto_acumulado, intent_detectado)

            if es_primer_mensaje:
                titulo_nuevo = await generar_titulo_inteligente(message, texto_acumulado, api_key)
                conversations_collection.update_one({"_id": conv_id}, {"$set": {"title": titulo_nuevo}})
                yield f"data: {json.dumps({'new_title': titulo_nuevo})}\n\n"

            yield "data: [DONE]\n\n"
            return

        elif "DOMOTICS_CONTROL" in intent_detectado:
            texto_acumulado = "Entendido, Alexis. Conectando con los sistemas de domótica... (Módulo IoT en desarrollo)."
            yield f"data: {json.dumps({'token': texto_acumulado})}\n\n"
            guardar_en_conversacion(user_id, conv_id, message, texto_acumulado, intent_detectado)
            yield "data: [DONE]\n\n"
            return

        else:
            historial_previo = []
            if doc_previo and "messages" in doc_previo:
                for m in doc_previo["messages"][-6:]:
                    historial_previo.append({"role": m["role"], "content": m["content"]})

            contexto_perfil = cv_texto[:1500] if cv_texto else "Sin perfil."
            contexto_doc = f"\n--- BASE DE CONOCIMIENTO (RAG) ---\n{rag_context}\n---------------------------------\n" if rag_context else ""

            prompt_sistema = {
                "role": "system",
                "content": (
                    f"Eres AI Alexis (J.A.R.V.I.S.). Asistente leal, inteligente y técnico.\n"
                    f"Perfil del usuario:\n{contexto_perfil}\n{contexto_doc}\n"
                    "REGLAS OBLIGATORIAS:\n"
                    "1. Formato visual: Listas con viñetas (*), negritas en conceptos clave y sin tablas Markdown (|).\n"
                    "2. Fórmulas matemáticas: Usa SIEMPRE formato LaTeX estándar con delimitadores $$ para ecuaciones en bloque y $ para fórmulas en línea (nunca uses corchetes \\[ \\] ni paréntesis \\( \\)).\n"
                    "3. Responde siempre en español, de forma concisa y elegante."
                )
            }
            mensajes_para_llm = [prompt_sistema] + historial_previo + [{"role": "user", "content": message}]

        # 4. TRANSMISIÓN DE TOKENS (STREAMING)
        response_stream = await litellm.acompletion(
            model=TEXT_MODEL,
            api_key=api_key,
            messages=mensajes_para_llm,
            stream=True
        )

        async for chunk in response_stream:
            token = chunk.choices[0].delta.content or ""
            if token:
                texto_acumulado += token
                yield f"data: {json.dumps({'token': token})}\n\n"

        # 5. PERSISTENCIA FINAL
        guardar_en_conversacion(
            user_id=user_id,
            conversation_id=conv_id,
            user_text=message or "📸 [Imagen enviada]",
            bot_response=texto_acumulado,
            intent=intent_detectado
        )

        if es_primer_mensaje and (message or texto_acumulado):
            titulo_nuevo = await generar_titulo_inteligente(message, texto_acumulado, api_key)
            conversations_collection.update_one({"_id": conv_id}, {"$set": {"title": titulo_nuevo}})
            yield f"data: {json.dumps({'new_title': titulo_nuevo})}\n\n"

        yield "data: [DONE]\n\n"

    except Exception as e:
        err_msg = f"⚠️ Error en streaming: {str(e)}"
        yield f"data: {json.dumps({'token': err_msg, 'intent': 'ERROR'})}\n\n"
        yield "data: [DONE]\n\n"


# --- PROCESADOR SÍNCRONO (FALLBACK Y LLAMADAS DIRECTAS) ---
async def procesar_mensaje_alexis(
    message: str,
    cv_texto: str,
    api_key: str,
    user_id: str = "guest_user",
    image_base64: str = None,
    rag_context: str = ""
) -> AssistantResponse:
    os.environ["GROQ_API_KEY"] = api_key
    os.environ["OPENAI_API_KEY"] = api_key

    if image_base64:
        try:
            gemini_key = os.getenv("GEMINI_API_KEY")
            if not gemini_key:
                return AssistantResponse(intent="IMAGE_ANALYSIS", response="⚠️ Falta configurar GEMINI_API_KEY.")
            client = genai.Client(api_key=gemini_key)
            image_bytes = base64.b64decode(image_base64)
            response = client.models.generate_content(
                model=VISION_MODEL,
                contents=["Analiza y describe la imagen en español.", types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"), message or ""]
            )
            return AssistantResponse(intent="IMAGE_ANALYSIS", response=response.text.strip())
        except Exception as e:
            return AssistantResponse(intent="IMAGE_ANALYSIS", response=f"Error en visión: {str(e)}")

    mensaje_lower = message.lower()
    palabras_clima = ["tiempo", "clima", "temperatura", "grados", "lluvia", "meteorológico", "viento"]
    if any(palabra in mensaje_lower for palabra in palabras_clima):
        user_intent = "WEATHER"
    else:
        system_prompt = "Responde SOLO con: 'WEATHER', 'SEARCH', 'CV_OPTIMIZATION', 'DOMOTICS_CONTROL', 'GENERAL_CHAT'."
        classification = litellm.completion(
            model=TEXT_MODEL,
            api_key=api_key,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": message}],
            temperature=0.0
        )
        user_intent = re.sub(r'[^A-Z_]', '', classification.choices[0].message.content.strip().upper()) or "GENERAL_CHAT"

    if "WEATHER" in user_intent:
        res_loc = litellm.completion(
            model=TEXT_MODEL,
            api_key=api_key,
            messages=[{"role": "system", "content": "Extrae SOLO la ciudad. Si no hay, 'Vigo'."}, {"role": "user", "content": message}],
            temperature=0.0
        )
        ubicacion = re.sub(r'[^a-zA-ZáéíóúÁÉÍÓÚñÑ\s]', '', res_loc.choices[0].message.content.strip()) or "Vigo"
        dias = 1 if "mañana" in message.lower() else 0
        datos_clima = await consultar_clima_open_meteo(ubicacion, dias=dias)
        prompt_clima = f"Eres AI Alexis. Responde con estos datos en viñetas (*):\n{json.dumps(datos_clima, ensure_ascii=False)}\nConsulta: {message}"
        chat_res = litellm.completion(model=TEXT_MODEL, api_key=api_key, messages=[{"role": "user", "content": prompt_clima}])
        return AssistantResponse(intent="WEATHER", response=chat_res.choices[0].message.content)

    elif "SEARCH" in user_intent:
        tema = optimizar_query_busqueda(message, api_key)
        contexto_web = buscar_en_internet(tema)
        prompt_search = f"Eres AI Alexis (J.A.R.V.I.S.). FECHA: 2026. Responde con viñetas (*):\n{contexto_web}\nConsulta: {message}"
        chat_res = litellm.completion(model=TEXT_MODEL, api_key=api_key, messages=[{"role": "user", "content": prompt_search}])
        return AssistantResponse(intent="SEARCH", response=chat_res.choices[0].message.content)

    else:
        prompt_sistema = {
            "role": "system",
            "content": (
                f"Eres AI Alexis (J.A.R.V.I.S.).\nPerfil:\n{cv_texto[:1500]}\nRAG:\n{rag_context}\n"
                "Para matemáticas usa siempre delimitadores LaTeX $ y $$."
            )
        }
        chat_res = litellm.completion(model=TEXT_MODEL, api_key=api_key, messages=[prompt_sistema, {"role": "user", "content": message}])
        return AssistantResponse(intent="GENERAL_CHAT", response=chat_res.choices[0].message.content)


# --- ENDPOINTS PRINCIPALES ---

@router.post("/chat-stream")
async def handle_assistant_chat_stream(payload: AssistantRequest):
    """Endpoint principal de chat con respuestas en streaming (SSE)."""
    api_key = payload.groq_api_key
    if not api_key or str(api_key).strip().lower() in ["string", "null", "none", "", "undefined"]:
        api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise HTTPException(status_code=400, detail="API Key de Groq no configurada en el servidor.")

    user_id = payload.user_id or "guest_user"
    rag_context = vector_search(user_id=user_id, query=payload.message, limit=3) if payload.message else ""
    db_profile = profiles_collection.find_one({"_id": user_id})
    cv_a_procesar = db_profile.get("cv_text", BASE_CV) if db_profile else BASE_CV

    return StreamingResponse(
        generar_stream_alexis(
            message=payload.message,
            cv_texto=cv_a_procesar,
            api_key=api_key,
            user_id=user_id,
            conversation_id=payload.conversation_id,
            image_base64=payload.image,
            rag_context=rag_context
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

@router.post("/chat", response_model=AssistantResponse)
async def handle_assistant_chat(payload: AssistantRequest):
    """Endpoint síncrono estándar de chat."""
    api_key = payload.groq_api_key or os.getenv("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(status_code=400, detail="API Key de Groq no configurada.")

    user_id = payload.user_id or "guest_user"
    rag_context = vector_search(user_id=user_id, query=payload.message, limit=3) if payload.message else ""
    db_profile = profiles_collection.find_one({"_id": user_id})
    cv_a_procesar = db_profile.get("cv_text", BASE_CV) if db_profile else BASE_CV

    resultado = await procesar_mensaje_alexis(
        message=payload.message,
        cv_texto=cv_a_procesar,
        api_key=api_key,
        user_id=user_id,
        image_base64=payload.image,
        rag_context=rag_context
    )

    conv_id = guardar_en_conversacion(
        user_id=user_id,
        conversation_id=payload.conversation_id,
        user_text=payload.message or "📸 [Imagen enviada]",
        bot_response=resultado.response,
        intent=resultado.intent
    )
    resultado.conversation_id = conv_id
    return resultado

@router.post("/voice", response_model=AssistantResponse)
async def handle_assistant_voice(
    file: UploadFile = File(...),
    groq_api_key: str = Form(None),
    user_id: str = Form("guest_user"),
    conversation_id: Optional[str] = Form(None),
    cv_text: str = Form("")
):
    """Transcribe audio con Whisper, procesa la intención y guarda en la sesión activa."""
    api_key = groq_api_key or os.getenv("GROQ_API_KEY")
    if not api_key or str(api_key).strip().lower() in ["string", "null", "none", "", "undefined"]:
        api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise HTTPException(status_code=400, detail="API Key de Groq no configurada.")

    nombre_temp = f"temp_{uuid.uuid4().hex}_{file.filename}"
    try:
        with open(nombre_temp, "wb") as f:
            f.write(await file.read())

        with open(nombre_temp, "rb") as audio_file:
            transcription = litellm.transcription(
                model="groq/whisper-large-v3",
                file=audio_file,
                api_key=api_key
            )

        if os.path.exists(nombre_temp):
            os.remove(nombre_temp)

        texto_transcrito = transcription.get("text", "").strip()
        if not texto_transcrito:
            raise HTTPException(status_code=400, detail="No se pudo entender el audio.")

        db_profile = profiles_collection.find_one({"_id": user_id})
        cv_a_procesar = db_profile.get("cv_text", BASE_CV) if db_profile else (cv_text or BASE_CV)
        rag_context = vector_search(user_id=user_id, query=texto_transcrito, limit=3)

        resultado = await procesar_mensaje_alexis(
            message=texto_transcrito,
            cv_texto=cv_a_procesar,
            api_key=api_key,
            user_id=user_id,
            rag_context=rag_context
        )

        respuesta_con_transcripcion = f"*(Entendí: \"{texto_transcrito}\")*\n\n{resultado.response}"

        conv_id = guardar_en_conversacion(
            user_id=user_id,
            conversation_id=conversation_id,
            user_text=f"🎙️ {texto_transcrito}",
            bot_response=resultado.response,
            intent=resultado.intent
        )
        guardar_en_historial(user_id, f"🎙️ {texto_transcrito}", resultado.response, resultado.intent)

        resultado.response = respuesta_con_transcripcion
        resultado.conversation_id = conv_id
        return resultado

    except Exception as e:
        if os.path.exists(nombre_temp):
            os.remove(nombre_temp)
        raise HTTPException(status_code=500, detail=f"Error en nota de voz: {str(e)}")

# --- ENDPOINTS DE UTILIDADES AVANZADAS (BÚSQUEDA Y EXPORTACIÓN) ---

@router.get("/conversations/search/{user_id}")
async def search_user_conversations(user_id: str, q: str = Query("", description="Término a buscar")):
    """Búsqueda en tiempo real por palabras clave en los títulos y mensajes de las conversaciones."""
    try:
        query_limpia = q.strip()
        if not query_limpia:
            cursor = conversations_collection.find(
                {"user_id": user_id},
                {"_id": 1, "title": 1, "updated_at": 1}
            ).sort("updated_at", -1)
            chats = [{"id": str(c["_id"]), "title": c.get("title", "Conversación"), "updated_at": c.get("updated_at")} for c in cursor]
            return {"conversations": chats}

        regex_pattern = {"$regex": re.escape(query_limpia), "$options": "i"}
        cursor = conversations_collection.find(
            {
                "user_id": user_id,
                "$or": [
                    {"title": regex_pattern},
                    {"messages.content": regex_pattern}
                ]
            },
            {"_id": 1, "title": 1, "updated_at": 1}
        ).sort("updated_at", -1)

        chats = [{"id": str(c["_id"]), "title": c.get("title", "Conversación"), "updated_at": c.get("updated_at")} for c in cursor]
        return {"conversations": chats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al buscar conversaciones: {str(e)}")

@router.get("/conversation/{conversation_id}/export/markdown")
async def export_conversation_markdown(conversation_id: str):
    """Devuelve la conversación completa formateada en un archivo descargable Markdown (.md)."""
    try:
        doc = conversations_collection.find_one({"_id": conversation_id})
        if not doc or "messages" not in doc:
            raise HTTPException(status_code=404, detail="Conversación no encontrada.")

        titulo = doc.get("title", "Conversacion").replace(" ", "_")
        fecha_creacion = doc.get("created_at", datetime.utcnow()).strftime("%Y-%m-%d %H:%M")

        md_content = [
            f"# 💬 Conversación: {doc.get('title', 'AI Alexis Chat')}",
            f"**Fecha:** {fecha_creacion} | **ID:** `{conversation_id}`\n",
            "---\n"
        ]

        for m in doc["messages"]:
            rol = "🧑 **Usuario**" if m["role"] == "user" else "🤖 **AI Alexis**"
            intent_info = f" *(Intención: {m['intent']})*" if m.get("intent") else ""
            md_content.append(f"### {rol}{intent_info}\n")
            md_content.append(f"{m['content']}\n")
            md_content.append("---\n")

        resultado_texto = "\n".join(md_content)
        headers = {
            "Content-Disposition": f"attachment; filename={titulo}_{conversation_id[:6]}.md"
        }
        return PlainTextResponse(content=resultado_texto, media_type="text/markdown", headers=headers)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error exportando conversación: {str(e)}")

# --- ENDPOINTS DE SESIONES Y GESTIÓN DE CHATS ---

@router.get("/conversations/{user_id}")
async def get_user_conversations(user_id: str):
    """Devuelve todas las sesiones de chat del usuario ordenadas por fecha."""
    try:
        cursor = conversations_collection.find(
            {"user_id": user_id},
            {"_id": 1, "title": 1, "updated_at": 1}
        ).sort("updated_at", -1)
        chats = [{"id": str(c["_id"]), "title": c.get("title", "Conversación"), "updated_at": c.get("updated_at")} for c in cursor]
        return {"conversations": chats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al listar conversaciones: {str(e)}")

@router.get("/conversation/{conversation_id}")
async def get_conversation_messages(conversation_id: str):
    """Recupera los mensajes de un chat específico."""
    try:
        doc = conversations_collection.find_one({"_id": conversation_id})
        if doc and "messages" in doc:
            return {"messages": doc["messages"]}
        return {"messages": []}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al obtener mensajes: {str(e)}")

@router.delete("/conversation/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Elimina una conversación y sus mensajes en Atlas."""
    try:
        conversations_collection.delete_one({"_id": conversation_id})
        return {"status": "deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al eliminar conversación: {str(e)}")

@router.get("/history/{user_id}")
async def get_user_chat_history(user_id: str):
    """Recupera el historial general de chat."""
    try:
        doc = history_collection.find_one({"_id": user_id})
        if doc and "messages" in doc:
            return {"messages": doc["messages"]}
        return {"messages": []}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al recuperar historial: {str(e)}")

# --- ENDPOINTS DOCUMENTAL (RAG) Y CV ---

@router.post("/upload-cv")
async def handle_upload_cv(file: UploadFile = File(...), user_id: str = Form("guest_user")):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Sube un documento PDF.")

    try:
        contenido = await file.read()
        texto_extraido = extraer_texto_pdf(contenido)
        if not texto_extraido:
            raise HTTPException(status_code=400, detail="No se pudo extraer texto del PDF.")

        profiles_collection.update_one(
            {"_id": user_id},
            {"$set": {"filename": file.filename, "cv_text": texto_extraido}},
            upsert=True
        )
        return {"user_id": user_id, "filename": file.filename, "status": "success_saved_to_atlas"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al guardar CV: {str(e)}")

@router.post("/upload-document")
async def upload_document(file: UploadFile = File(...), user_id: str = Form(...)):
    try:
        content_bytes = await file.read()
        texto = extract_text_from_pdf(content_bytes) if file.filename.lower().endswith(".pdf") else content_bytes.decode("utf-8", errors="ignore")
        if not texto.strip():
            return {"status": "error", "message": "Documento vacío o no legible."}

        num_chunks = index_document_content(user_id=user_id, filename=file.filename, text=texto)
        return {"status": "success", "filename": file.filename, "chunks_indexed": num_chunks}
    except Exception as e:
        return {"status": "error", "message": f"Error al indexar documento: {str(e)}"}

# --- ENDPOINT TTS (TEXT-TO-SPEECH) ---
class TTSRequest(BaseModel):
    text: str
    voice: Optional[str] = "es-ES-AlvaroNeural"

@router.post("/tts")
async def generar_audio_tts(payload: TTSRequest):
    """Genera audio binario MP3 usando Edge TTS."""
    try:
        audio_bytes = await sintetizar_voz_neural(payload.text, voice=payload.voice or "es-ES-AlvaroNeural")
        return Response(content=audio_bytes, media_type="audio/mpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generando audio TTS: {str(e)}")