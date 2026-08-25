import os
import io
import litellm
import re
import json
import time
from typing import Optional, Any
import httpx
from litellm import completion
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from app.schemas.assistant import AssistantRequest, AssistantResponse
from duckduckgo_search import DDGS
from crewai import Agent, Task, Crew, Process, LLM
from pypdf import PdfReader
from pymongo import MongoClient
from dotenv import load_dotenv
from pathlib import Path
from app.services.weather_service import consultar_clima_open_meteo
from tavily import TavilyClient
from app.services.rag_service import extract_text_from_pdf, index_document_content, vector_search
import base64
from google import genai
from google.genai import types
from app.services.news_service import buscar_noticias
import uuid
from datetime import datetime
from fastapi.responses import Response
from pydantic import BaseModel
from app.services.tts_service import sintetizar_voz_neural

# --- CONFIGURACIÓN DE MODELOS ---
TEXT_MODEL = "groq/openai/gpt-oss-120b"  # Groq (Texto, RAG, Búsqueda, Clima, Agentes)
VISION_MODEL = "gemini-3.6-flash"        # Google AI (Visión y análisis de imágenes)

# RUTA ABSOLUTA AL ARCHIVO .env
ruta_raiz = Path(__file__).resolve().parent.parent.parent
ruta_env = ruta_raiz / ".env"
load_dotenv(dotenv_path=ruta_env)

router = APIRouter(prefix="/assistant", tags=["AI Alexis Assistant"])

# --- CONEXIÓN A MONGODB ATLAS ---
MONGODB_URI = os.getenv("MONGODB_URI")
if not MONGODB_URI:
    print("❌ ERROR CRÍTICO: MONGODB_URI no se está leyendo del archivo .env")
    MONGODB_URI = "mongodb://localhost:27017"

client = MongoClient(MONGODB_URI)
db = client["ai_alexis_db"]
profiles_collection = db["profiles"]
history_collection = db["chat_history"]

# --- CACHÉ SIMPLE CON TTL ---
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
    _cache[key] = (value, time.time())

# --- HELPER: GUARDAR MENSAJES EN MONGO ATLAS ---
def guardar_en_historial(user_id: str, user_text: str, bot_response: str, intent: str):
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
        print(f"⚠️ Error guardando historial en Atlas: {e}")

# COLECCION DEDICADA A MULTIPLES SESIONES
conversations_collection = db["conversations"]

# --- HELPER: GUARDAR / CREAR SESIÓN DE CHAT ---
def guardar_en_conversacion(user_id: str, conversation_id: str, user_text: str, bot_response: str, intent: str) -> str:
    if not conversation_id:
        conversation_id = str(uuid.uuid4())
    
    # Generar título inicial a partir del primer mensaje
    titulo = user_text[:30] + ("..." if len(user_text) > 30 else "")
    
    conversations_collection.update_one(
        {"_id": conversation_id},
        {
            "$setOnInsert": {
                "user_id": user_id,
                "title": titulo,
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

# --- PERFIL PROFESIONAL BASE ---
BASE_CV = """
Nombre Completo: Alexis Fernando Pérez Yamasque
Ubicación: Vigo, Galicia
Perfil Profesional: 
Ingeniero de IA y Estudiante de Big Data con sólida formación en desarrollo multiplataforma y administración de sistemas. Especializado en el diseño, entrenamiento y despliegue de modelos de Machine Learning y Deep Learning.
"""

# --- UTILERÍA: EXTRACTOR DE TEXTO DE PDF ---
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

# --- OPTIMIZADOR INTELIGENTE DE BÚSQUEDAS ---
def optimizar_query_busqueda(mensaje_usuario: str, api_key: str) -> str:
    try:
        prompt_optimizador = (
            "Tu único trabajo es convertir un mensaje conversacional en una consulta de búsqueda "
            "ultra-eficiente de 2 a 4 palabras clave para un buscador web.\n"
            "Ejemplo: '¿Sabes quién ganó el partido de ayer del Real Madrid?' -> 'resultado Real Madrid ayer'\n\n"
            "REGLA ESTRICTA: Responde ÚNICAMENTE con las palabras clave sugeridas, sin introducciones, sin comillas y sin explicaciones."
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
        query_limpia = response.choices[0].message.content.strip().replace('"', '').replace("'", "")
        return query_limpia
    except Exception:
        return mensaje_usuario

# --- EJECUTOR DE BÚSQUEDAS EN TIEMPO REAL ---
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

# --- PROCESADOR CENTRAL DE INTENCIONES ---
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

    # --- 0. PROCESAMIENTO MULTIMODAL CON VISIÓN (GOOGLE GENAI SDK) ---
    if image_base64:
        try:
            gemini_key = os.getenv("GEMINI_API_KEY")
            if not gemini_key:
                return AssistantResponse(
                    intent="IMAGE_ANALYSIS",
                    response="⚠️ Falta configurar la variable GEMINI_API_KEY en el servidor."
                )

            client = genai.Client(api_key=gemini_key)

            texto_usuario = message.strip() if message and message.strip() else "Analiza y describe esta imagen en detalle."
            prompt_vision = (
                "Eres AI Alexis, un asistente virtual directo, técnico y elegante.\n"
                f"Consulta del usuario: {texto_usuario}\n"
                "Responde de forma concisa, profesional y en español."
            )

            image_bytes = base64.b64decode(image_base64)

            response = client.models.generate_content(
                model=VISION_MODEL,
                contents=[
                    prompt_vision,
                    types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
                ]
            )

            return AssistantResponse(
                intent="IMAGE_ANALYSIS",
                response=response.text.strip()
            )
        except Exception as e:
            return AssistantResponse(
                intent="IMAGE_ANALYSIS",
                response=f"Error al analizar la imagen: {str(e)}"
            )

    # --- 1. DETECCIÓN DE INTENCIÓN (CON FALLBACK POR PALABRAS CLAVE) ---
    mensaje_lower = message.lower()
    palabras_clima = ["tiempo", "clima", "temperatura", "grados", "lluvia", "meteorológico", "soleado", "nublado", "viento", "presión", "humedad"]
    
    if any(palabra in mensaje_lower for palabra in palabras_clima):
        user_intent = "WEATHER"
        print(f"🔍 FALLBACK: detectado WEATHER por palabras clave en: {message}")
    else:
        # --- CLASIFICACIÓN CON LLM ---
        system_prompt = (
            "Eres el clasificador de intenciones de AI Alexis.\n"
            "Tu único trabajo es leer el mensaje del usuario y responder ÚNICAMENTE con una de estas cinco palabras:\n"
            "- 'WEATHER' (si pregunta por el tiempo, clima, temperatura, grados, predicción meteorológica o lluvia).\n"
            "  Ejemplos: '¿Qué tiempo hace en Madrid?' -> WEATHER, 'Temperatura para mañana en Vigo' -> WEATHER\n"
            "- 'SEARCH' (si pide noticias, titulares, información de actualidad, búsquedas web generales, resultados deportivos).\n"
            "  Ejemplos: 'Últimas noticias de OpenAI' -> SEARCH, '¿Quién ganó el partido de ayer?' -> SEARCH\n"
            "- 'CV_OPTIMIZATION' (si quiere adaptar su CV, currículum o habla de ofertas de empleo).\n"
            "  Ejemplo: 'Optimiza mi CV para un puesto de Data Scientist' -> CV_OPTIMIZATION\n"
            "- 'DOMOTICS_CONTROL' (si habla de controlar luces, sensores, domótica o IoT).\n"
            "  Ejemplo: 'Enciende la luz del salón' -> DOMOTICS_CONTROL\n"
            "- 'GENERAL_CHAT' (para saludos, charla casual, dudas generales, programación o análisis de documentos).\n"
            "  Ejemplo: 'Hola, ¿cómo estás?' -> GENERAL_CHAT\n\n"
            "REGLA ESTRICTA: Responde SOLO con la palabra exacta en mayúsculas, sin comillas, introducciones ni explicaciones."
        )

        classification = litellm.completion(
            model=TEXT_MODEL,
            api_key=api_key,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message}
            ],
            temperature=0.0
        )

        user_intent = classification.choices[0].message.content.strip().upper()
        user_intent = re.sub(r'[^A-Z_]', '', user_intent)  # Limpieza adicional
        print(f"🔍 INTENT DETECTED (LLM): {user_intent}")

    # --- 2. EJECUCIÓN POR INTENCIÓN ---

    # --- 2.1 INTENCIÓN: METEOROLOGÍA / CLIMA (OPEN-METEO) ---
    if "WEATHER" in user_intent:
        try:
            # Extraer ubicación
            prompt_ubicacion = (
                "Extrae ÚNICAMENTE el nombre de la ciudad, isla o municipio mencionado en el mensaje del usuario.\n"
                "Ejemplo: 'Dime los grados de temperatura para mañana en Fuerteventura' -> 'Fuerteventura'\n"
                "Ejemplo: '¿Va a llover en Sevilla hoy?' -> 'Sevilla'\n"
                "Si no detectas ninguna ubicación explícita, responde 'Vigo'.\n"
                "REGLA ESTRICTA: Responde SOLO con el nombre de la ubicación sin comillas ni signos."
            )
            res_loc = litellm.completion(
                model=TEXT_MODEL,
                api_key=api_key,
                messages=[
                    {"role": "system", "content": prompt_ubicacion},
                    {"role": "user", "content": message}
                ],
                temperature=0.0
            )
            ubicacion_raw = res_loc.choices[0].message.content.strip()
            ubicacion = re.sub(r'[^a-zA-ZáéíóúÁÉÍÓÚñÑ\s]', '', ubicacion_raw).strip()
            print(f"📍 Ubicación extraída: {ubicacion}")

            # Detectar si es hoy o mañana
            es_manana = "mañana" in message.lower() or "manana" in message.lower()
            dias = 1 if es_manana else 0
            print(f"📅 Días: {'mañana' if es_manana else 'hoy'}")

            # Consultar clima con caché
            cache_key = f"weather_{ubicacion.lower()}_{dias}"
            datos_clima = get_cached(cache_key, "weather")
            if not datos_clima:
                try:
                    datos_clima = await consultar_clima_open_meteo(ubicacion, dias=dias)
                    set_cached(cache_key, datos_clima, "weather")
                except httpx.HTTPStatusError as e:
                    if e.response.status_code == 429:
                        return AssistantResponse(
                            intent="WEATHER",
                            response="⏳ Límite de peticiones a Open-Meteo alcanzado. Intenta de nuevo en unos minutos."
                        )
                    else:
                        raise
                except json.JSONDecodeError:
                    return AssistantResponse(
                        intent="WEATHER",
                        response="⚠️ La respuesta del servicio meteorológico no tiene un formato válido."
                    )
            print(f"🌤️ Datos de clima: {datos_clima}")

            # Si hay error en la respuesta
            if "error" in datos_clima:
                return AssistantResponse(
                    intent="WEATHER",
                    response=f"⚠️ {datos_clima['error']}"
                )

            # Convertir a JSON para el prompt
            datos_clima_str = json.dumps(datos_clima, ensure_ascii=False, indent=2)

            prompt_respuesta_clima = (
                "Eres AI Alexis, un asistente virtual directo, inteligente y elegante (inspirado en J.A.R.V.I.S.).\n\n"
                "REGLAS OBLIGATORIAS DE RESPUESTA:\n"
                "1. Responde de forma clara y directa con los datos meteorológicos proporcionados.\n"
                "2. NUNCA sugieras buscar en webs externas ni incluyas enlaces.\n"
                "3. Presenta la información en formato de lista con viñetas (*), usando negritas para los conceptos clave (temperatura, lluvia).\n"
                "4. Sé natural, conversacional y conciso.\n\n"
                f"DATOS OFICIALES OPEN-METEO (JSON):\n{datos_clima_str}\n\n"
                f"Consulta del usuario: {message}"
            )

            chat_response = litellm.completion(
                model=TEXT_MODEL,
                api_key=api_key,
                messages=[{"role": "user", "content": prompt_respuesta_clima}]
            )

            return AssistantResponse(
                intent="WEATHER",
                response=chat_response.choices[0].message.content
            )

        except Exception as e:
            return AssistantResponse(
                intent="WEATHER",
                response=f"No pude consultar el clima: {str(e)}"
            )

    # --- 2.2 INTENCIÓN: BÚSQUEDA UNIFICADA (NEWS + WEB) ---
    elif "SEARCH" in user_intent:
        try:
            prompt_tema = (
                "Extrae el tema, personaje, país o sector sobre el que el usuario quiere buscar información.\n"
                "Ejemplo: '¿Cuáles son las últimas noticias de OpenAI?' -> 'OpenAI'\n"
                "Ejemplo: 'Dame los titulares de hoy en España' -> 'actualidad'\n"
                "Ejemplo: '¿Quién ganó el partido de ayer?' -> 'resultado partido'\n"
                "Si es una consulta general sin temática concreta, responde 'actualidad'.\n"
                "REGLA ESTRICTA: Responde SOLO con la palabra o término clave, sin comillas ni explicaciones."
            )
            res_tema = litellm.completion(
                model=TEXT_MODEL,
                api_key=api_key,
                messages=[
                    {"role": "system", "content": prompt_tema},
                    {"role": "user", "content": message}
                ],
                temperature=0.0
            )
            tema_raw = res_tema.choices[0].message.content.strip()
            tema = re.sub(r'[^a-zA-ZáéíóúÁÉÍÓÚñÑ0-9\s]', '', tema_raw).strip()

            # Decidir si es noticias o búsqueda general basado en palabras clave
            palabras_clave_noticias = ["noticia", "titular", "periódico", "actualidad", "última hora", "información", "evento", "partido"]
            es_noticias = any(palabra in message.lower() for palabra in palabras_clave_noticias)

            if es_noticias:
                # --- BÚSQUEDA DE NOTICIAS (NewsAPI) ---
                cache_key = f"news_{tema.lower()}"
                noticias_raw = get_cached(cache_key, "news")
                if not noticias_raw:
                    try:
                        noticias_raw = await buscar_noticias(query=tema)
                        set_cached(cache_key, noticias_raw, "news")
                    except httpx.HTTPStatusError as e:
                        if e.response.status_code == 429:
                            return AssistantResponse(
                                intent="SEARCH",
                                response="⏳ Límite de peticiones a NewsAPI alcanzado. Intenta de nuevo en unos minutos."
                            )
                        elif e.response.status_code == 404:
                            return AssistantResponse(
                                intent="SEARCH",
                                response="No se encontraron noticias para ese tema."
                            )
                        else:
                            raise
                    except json.JSONDecodeError:
                        return AssistantResponse(
                            intent="SEARCH",
                            response="⚠️ La respuesta del servicio de noticias no tiene un formato válido."
                        )

                prompt_respuesta = (
                    "Eres AI Alexis, un asistente informativo de alta precisión (inspirado en J.A.R.V.I.S.).\n"
                    "Presenta los titulares de forma estructurada:\n"
                    "- Usa viñetas (*) para cada noticia.\n"
                    "- Incluye el nombre de la fuente entre paréntesis.\n"
                    "- Si hay enlace, añade ' [Leer más](url)'.\n"
                    "- Usa negritas para los temas principales.\n"
                    "Sé ejecutivo, claro y neutral.\n\n"
                    f"TITULARES RECUPERADOS:\n{noticias_raw}\n\n"
                    f"Consulta del usuario: {message}"
                )
            else:
                # --- BÚSQUEDA WEB GENERAL (Tavily / DuckDuckGo) ---
                query_optima = optimizar_query_busqueda(message, api_key)
                cache_key = f"web_{query_optima.lower()}"
                contexto_web = get_cached(cache_key, "web")
                if not contexto_web:
                    contexto_web = buscar_en_internet(query_optima)
                    set_cached(cache_key, contexto_web, "web")

                prompt_respuesta = (
                    f"Eres AI Alexis, un asistente de inteligencia artificial leal, directo e inteligente (inspirado en J.A.R.V.I.S.).\n"
                    f"FECHA ACTUAL DEL SISTEMA: Año 2026.\n\n"
                    f"Usa la siguiente información extraída de internet para responder a la pregunta del usuario.\n"
                    f"REGLAS:\n"
                    f"1. Responde de forma clara, directa y precisa basándote en los datos web.\n"
                    f"2. Presenta la información en formato de lista con viñetas (*), usando negritas para los conceptos clave.\n"
                    f"3. NUNCA digas que no tienes información si los datos web la contienen.\n"
                    f"4. NUNCA le pidas al usuario que navegue en la web por su cuenta ni incluyas URLs o corchetes de fuentes [1].\n\n"
                    f"--- INFORMACIÓN RECUPERADA DE INTERNET ---\n"
                    f"{contexto_web}\n"
                    f"------------------------------------------\n\n"
                    f"Consulta del usuario: {message}"
                )

            chat_response = litellm.completion(
                model=TEXT_MODEL,
                api_key=api_key,
                messages=[{"role": "user", "content": prompt_respuesta}]
            )
            return AssistantResponse(
                intent="SEARCH",
                response=chat_response.choices[0].message.content
            )

        except Exception as e:
            return AssistantResponse(
                intent="SEARCH",
                response=f"No pude realizar la búsqueda en este momento: {str(e)}"
            )

    # --- 2.3 INTENCIÓN: OPTIMIZACIÓN DE CV ---
    elif "CV_OPTIMIZATION" in user_intent:
        try:
            extraction = litellm.completion(
                model=TEXT_MODEL,
                api_key=api_key,
                messages=[
                    {"role": "system", "content": "Extrae exclusivamente la descripción detallada del puesto de trabajo o los requisitos del mensaje del usuario. Elimina saludos o textos irrelevantes."},
                    {"role": "user", "content": message}
                ],
                temperature=0.1
            )
            oferta_laboral = extraction.choices[0].message.content.strip()

            llm_groq = LLM(
                model=TEXT_MODEL,
                api_key=api_key
            )

            reclutador = Agent(
                role="Reclutador Técnico Senior",
                goal="Analizar ofertas de empleo y extraer requisitos técnicos esenciales y palabras clave para filtros ATS.",
                backstory="Eres un reclutador experto en el sector tecnológico. Sabes aislar perfectamente qué competencias indexan los algoritmos de selección automatizados.",
                verbose=True,
                llm=llm_groq
            )

            redactor = Agent(
                role="Consultor de Carreras y Redactor Técnico",
                goal="Adaptar el perfil profesional del usuario para alinearlo con la oferta de trabajo, destacando su experiencia real de forma atractiva.",
                backstory="Eres un redactor experto en currículums de ingeniería. Sabes pulir la presentación de habilidades técnicas para hacerlas destacar ante los mánagers de contratación.",
                verbose=True,
                llm=llm_groq
            )

            tarea_analisis = Task(
                description=f"Analiza detalladamente esta oferta de empleo e identifica los requisitos críticos:\n\n{oferta_laboral}",
                expected_output="Un reporte ejecutivo con las palabras clave, tecnologías obligatorias y competencias clave deseadas.",
                agent=reclutador
            )

            tarea_adaptacion = Task(
                description=f"Toma el currículum base proporcionado:\n\n{cv_texto}\n\nUtilizando el reporte de la tarea de análisis, reestructura el perfil profesional, la experiencia y la sección de habilidades para encajar de forma óptima con la vacante. Genera el resultado final en un Markdown limpio y elegante.",
                expected_output="Un currículum optimizado estructurado en Markdown profesional.",
                agent=redactor
            )

            orquestador = Crew(
                agents=[reclutador, redactor],
                tasks=[tarea_analisis, tarea_adaptacion],
                process=Process.sequential,
                memory=False,
                cache=False
            )

            cv_optimizado = orquestador.kickoff()

            return AssistantResponse(
                intent="CV_OPTIMIZATION",
                response=f"He activado a mis agentes de selección en segundo plano. Analicé la oferta y reestructuré el perfil profesional con éxito.\n\nAquí tienes el resultado de la optimización:\n\n{str(cv_optimizado)}"
            )
        except Exception as e:
            return AssistantResponse(
                intent="CV_OPTIMIZATION",
                response=f"Hubo un contratiempo al coordinar la Crew de agentes: {str(e)}"
            )

    # --- 2.4 INTENCIÓN: CONTROL DOMÓTICO ---
    elif "DOMOTICS_CONTROL" in user_intent:
        return AssistantResponse(
            intent="DOMOTICS_CONTROL",
            response="Entendido, Alexis. Conectando con los sistemas de domótica... (Módulo IoT en desarrollo)."
        )

    # --- 2.5 INTENCIÓN: CHARLA GENERAL Y RAG DOCUMENTAL ---
    else:
        historial_previo = []
        doc = history_collection.find_one({"_id": user_id})
        if doc and "messages" in doc:
            for m in doc["messages"][-6:]:
                historial_previo.append({"role": m["role"], "content": m["content"]})

        contexto_perfil = cv_texto[:1500] if cv_texto else "Sin perfil registrado."

        contexto_documentos = ""
        if rag_context and rag_context.strip():
            contexto_documentos = (
                f"\n--- INFORMACIÓN DE LA BASE DE CONOCIMIENTO (DOCUMENTOS INDEXADOS) ---\n"
                f"{rag_context}\n"
                f"--------------------------------------------------------------------\n"
            )

        prompt_sistema = {
            "role": "system",
            "content": (
                f"Eres AI Alexis, un asistente de IA leal, altamente inteligente y con un toque de ingenio técnico, inspirado en J.A.R.V.I.S.\n"
                f"Estás hablando con el usuario registrado (ID de sesión: {user_id}).\n"
                f"--- DATOS DEL PERFIL DEL USUARIO ---\n"
                f"{contexto_perfil}\n"
                f"------------------------------------\n"
                f"{contexto_documentos}\n"
                f"REGLAS OBLIGATORIAS:\n"
                f"1. Si se proporciona información de la BASE DE CONOCIMIENTO, úsala como fuente principal y obligatoria de verdad técnica.\n"
                f"2. Usa los datos del perfil para responder dudas sobre identidad y experiencia.\n"
                f"3. FORMATO VISUAL: NUNCA generes tablas con barras Markdown (|). Presenta siempre la información en listas con viñetas (*), títulos en negrita y saltos de línea limpios entre elementos.\n"
                f"4. Responde siempre de forma clara, concisa y en español."
            )
        }

        mensajes_para_llm = [prompt_sistema] + historial_previo + [{"role": "user", "content": message}]

        chat_response = litellm.completion(
            model=TEXT_MODEL,
            api_key=api_key,
            messages=mensajes_para_llm
        )
        return AssistantResponse(
            intent="GENERAL_CHAT",
            response=chat_response.choices[0].message.content
        )

# --- ENDPOINTS ---

@router.get("/history/{user_id}")
async def get_user_chat_history(user_id: str):
    try:
        doc = history_collection.find_one({"_id": user_id})
        if doc and "messages" in doc:
            return {"messages": doc["messages"]}
        return {"messages": []}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al recuperar historial: {str(e)}")

@router.post("/chat", response_model=AssistantResponse)
async def handle_assistant_chat(payload: AssistantRequest):
    api_key = payload.groq_api_key
    if not api_key or str(api_key).strip().lower() in ["string", "null", "none", "", "undefined"]:
        api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise HTTPException(status_code=400, detail="API Key de Groq no configurada en el servidor.")

    user_id = payload.user_id or "guest_user"

    # Contexto RAG
    rag_context = ""
    if payload.message:
        try:
            rag_context = vector_search(user_id=user_id, query=payload.message, limit=3)
        except Exception as e:
            rag_context = ""

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

    texto_guardado = payload.message or "📸 [Imagen enviada]"
    
    # Persistencia en la sesión correspondiente
    conv_id = guardar_en_conversacion(
        user_id=user_id,
        conversation_id=payload.conversation_id,
        user_text=texto_guardado,
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
    cv_text: str = Form("")
):
    api_key = groq_api_key or os.getenv("GROQ_API_KEY")
    if not api_key or api_key in ["string", "null", ""]:
        api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise HTTPException(status_code=400, detail="API Key de Groq no configurada.")

    nombre_archivo_temporal = f"temp_{file.filename}"
    try:
        with open(nombre_archivo_temporal, "wb") as f:
            f.write(await file.read())

        with open(nombre_archivo_temporal, "rb") as audio_file:
            transcription_response = litellm.transcription(
                model="groq/whisper-large-v3",
                file=audio_file,
                api_key=api_key
            )

        os.remove(nombre_archivo_temporal)
        texto_transcrito = transcription_response.get("text", "").strip()

        if not texto_transcrito:
            raise HTTPException(status_code=400, detail="No se pudo entender el audio.")

        db_profile = profiles_collection.find_one({"_id": user_id})
        if db_profile and "cv_text" in db_profile:
            cv_a_procesar = db_profile["cv_text"]
        else:
            cv_a_procesar = cv_text or BASE_CV

        # Búsqueda semántica RAG para mensajes por voz
        rag_context = vector_search(user_id=user_id, query=texto_transcrito, limit=3)

        resultado = await procesar_mensaje_alexis(
            message=texto_transcrito,
            cv_texto=cv_a_procesar,
            api_key=api_key,
            user_id=user_id,
            rag_context=rag_context
        )
        respuesta_formateada = f"*(Entendí: \"{texto_transcrito}\")*\n\n{resultado.response}"

        guardar_en_historial(user_id, f"🎙️ [Nota de voz]: {texto_transcrito}", resultado.response, resultado.intent)

        resultado.response = respuesta_formateada
        return resultado

    except Exception as e:
        if os.path.exists(nombre_archivo_temporal):
            os.remove(nombre_archivo_temporal)
        raise HTTPException(status_code=500, detail=f"Error al procesar la nota de voz: {str(e)}")

@router.post("/upload-cv")
async def handle_upload_cv(
    file: UploadFile = File(...),
    user_id: str = Form("guest_user")
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Formato de archivo inválido. Sube un documento PDF.")

    try:
        contenido = await file.read()
        texto_extraido = extraer_texto_pdf(contenido)

        if not texto_extraido:
            raise HTTPException(status_code=400, detail="No se pudo extraer texto del PDF.")

        profiles_collection.update_one(
            {"_id": user_id},
            {"$set": {
                "filename": file.filename,
                "cv_text": texto_extraido
            }},
            upsert=True
        )

        return {
            "user_id": user_id,
            "filename": file.filename,
            "status": "success_saved_to_atlas",
            "extracted_text_preview": texto_extraido[:200] + "..."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar y guardar: {str(e)}")

@router.post("/upload-document")
async def upload_document(
    file: UploadFile = File(...),
    user_id: str = Form(...)
):
    try:
        content_bytes = await file.read()
        if file.filename.lower().endswith(".pdf"):
            texto = extract_text_from_pdf(content_bytes)
        else:
            texto = content_bytes.decode("utf-8", errors="ignore")

        if not texto.strip():
            return {"status": "error", "message": "El documento está vacío o no contiene texto legible."}

        num_chunks = index_document_content(user_id=user_id, filename=file.filename, text=texto)
        return {
            "status": "success",
            "filename": file.filename,
            "chunks_indexed": num_chunks,
            "message": f"Documento indexado con éxito ({num_chunks} fragmentos vectorizados)."
        }
    except Exception as e:
        return {"status": "error", "message": f"Error al procesar el documento: {str(e)}"}


# --- ENDPOINTS DE GESTIÓN DE SESIONES ---

@router.get("/conversations/{user_id}")
async def get_user_conversations(user_id: str):
    """Devuelve la lista de todos los chats del usuario ordenados por fecha."""
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
    """Devuelve los mensajes de una conversación específica."""
    try:
        doc = conversations_collection.find_one({"_id": conversation_id})
        if doc and "messages" in doc:
            return {"messages": doc["messages"]}
        return {"messages": []}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al obtener mensajes: {str(e)}")

@router.delete("/conversation/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Elimina una conversación."""
    try:
        conversations_collection.delete_one({"_id": conversation_id})
        return {"status": "deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al eliminar conversación: {str(e)}")

# --- ENDPOINTS Audio en streaming/binario ---
class TTSRequest(BaseModel):
    text: str
    voice: Optional[str] = "es-ES-AlvaroNeural"

@router.post("/tts")
async def generar_audio_tts(payload: TTSRequest):
    """Genera audio MP3 neural a partir de un texto."""
    try:
        audio_bytes = await sintetizar_voz_neural(payload.text, voice=payload.voice or "es-ES-AlvaroNeural")
        return Response(content=audio_bytes, media_type="audio/mpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generando audio TTS: {str(e)}")