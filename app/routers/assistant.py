import os
import io
import litellm
import re
from litellm import completion
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from app.schemas.assistant import AssistantRequest, AssistantResponse
from duckduckgo_search import DDGS
from crewai import Agent, Task, Crew, Process, LLM
from pypdf import PdfReader
from pymongo import MongoClient
from dotenv import load_dotenv
from pathlib import Path
from app.services.weather_service import obtener_clima_open_meteo
from tavily import TavilyClient
from app.services.rag_service import extract_text_from_pdf, index_document_content, vector_search

# --- CONFIGURACIÓN DE MODELOS ---
TEXT_MODEL = "openai/gpt-oss-20b"
VISION_MODEL = "openai/qwen/qwen3.6-27b"

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
            base_url="https://api.groq.com/openai/v1",
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

    # --- 0. PROCESAMIENTO MULTIMODAL CON VISIÓN ---
    if image_base64:
        try:
            texto_usuario = message.strip() if message and message.strip() else "Analiza y describe esta imagen en detalle."
            
            prompt_vision = (
                "Eres AI Alexis, un asistente virtual de IA directo, técnico y elegante.\n"
                "Analiza la imagen y responde con precisión a la consulta del usuario.\n\n"
                "REGLAS:\n"
                "1. NUNCA incluyas etiquetas de pensamiento interno (<think>).\n"
                "2. Ve directamente al grano: resume lo principal y luego detalla textos y elementos clave.\n"
                "3. Mantén una redacción concisa, profesional y en español."
            )

            mensajes_vision = [
                {"role": "system", "content": prompt_vision},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": texto_usuario},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                        }
                    ]
                }
            ]

            vision_response = litellm.completion(
                model=VISION_MODEL,
                api_key=api_key,
                base_url="https://api.groq.com/openai/v1",
                messages=mensajes_vision
            )

            contenido_bruto = vision_response.choices[0].message.content or ""
            contenido_limpio = re.sub(r'<think>.*?</think>', '', contenido_bruto, flags=re.DOTALL).strip()

            return AssistantResponse(
                intent="IMAGE_ANALYSIS",
                response=contenido_limpio
            )
        except Exception as e:
            return AssistantResponse(
                intent="IMAGE_ANALYSIS",
                response=f"Ocurrió un error al procesar la imagen con el modelo de visión: {str(e)}"
            )

    # --- CLASIFICACIÓN DE INTENCIÓN DE TEXTO ---
    system_prompt = (
        "Eres el clasificador de intenciones de AI Alexis.\n"
        "Tu único trabajo es leer el mensaje del usuario y responder ÚNICAMENTE con una de estas cinco palabras:\n"
        "- 'WEATHER' (si pregunta por el tiempo, clima, temperatura, grados, predicción meteorológica o lluvia)\n"
        "- 'CV_OPTIMIZATION' (si quiere adaptar su CV, currículum o habla de ofertas de empleo)\n"
        "- 'DOMOTICS_CONTROL' (si habla de controlar luces, sensores, domótica o IoT)\n"
        "- 'WEB_SEARCH' (noticias, resultados deportivos, eventos o datos en tiempo real que NO sean el clima)\n"
        "- 'GENERAL_CHAT' (para saludos, charla casual, preguntas sobre documentos subidos o programación)\n\n"
        "REGLA ESTRICTA: Responde SOLO con la palabra exacta en mayúsculas, sin introducciones ni explicaciones."
    )

    classification = litellm.completion(
        model=TEXT_MODEL,
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message}
        ],
        temperature=0.0
    )

    user_intent = classification.choices[0].message.content.strip().upper()
    user_intent = user_intent.replace("'", "").replace('"', "").replace(".", "").strip()

    # --- 1. INTENCIÓN: METEOROLOGÍA / CLIMA ---
    if "WEATHER" in user_intent:
        try:
            prompt_ubicacion = (
                "Extrae ÚNICAMENTE el nombre de la ciudad, isla o municipio mencionado en el mensaje del usuario.\n"
                "Ejemplo: 'Dime los grados de temperatura para mañana en Fuerteventura' -> 'Fuerteventura'\n"
                "Ejemplo: '¿Va a llover en Sevilla hoy?' -> 'Sevilla'\n"
                "Si no detectas ninguna ubicación explícita, responde 'Fuerteventura'.\n"
                "REGLA ESTRICTA: Responde SOLO con el nombre de la ubicación sin comillas ni signos."
            )
            res_loc = litellm.completion(
                model=TEXT_MODEL,
                api_key=api_key,
                base_url="https://api.groq.com/openai/v1",
                messages=[
                    {"role": "system", "content": prompt_ubicacion},
                    {"role": "user", "content": message}
                ],
                temperature=0.0
            )
            ubicacion = res_loc.choices[0].message.content.strip().replace('"', '').replace("'", "")
            
            datos_clima = await obtener_clima_open_meteo(ubicacion)

            if datos_clima:
                contexto_clima = (
                    f"DATOS CLIMÁTICOS OFICIALES EN TIEMPO REAL PARA {datos_clima['ubicacion']}:\n"
                    f"- HOY: Temp. Máxima {datos_clima['hoy']['max']}°C, Temp. Mínima {datos_clima['hoy']['min']}°C, Probabilidad de lluvia {datos_clima['hoy']['prob_lluvia']}%\n"
                    f"- MAÑANA: Temp. Máxima {datos_clima['manana']['max']}°C, Temp. Mínima {datos_clima['manana']['min']}°C, Probabilidad de lluvia {datos_clima['manana']['prob_lluvia']}%\n"
                )
            else:
                contexto_clima = f"No se han podido consultar los datos automáticos para la ubicación {ubicacion}."

            prompt_respuesta_clima = (
                "Eres AI Alexis, un asistente virtual directo, inteligente y elegante (inspirado en J.A.R.V.I.S.).\n\n"
                "REGLAS OBLIGATORIAS DE RESPUESTA:\n"
                "1. Responde de forma clara y directa con las temperaturas exactas en °C (máxima y mínima) y la probabilidad de lluvia.\n"
                "2. Si la consulta menciona 'mañana', prioriza dar los datos del pronóstico de mañana.\n"
                "3. NUNCA sugieras al usuario buscar en webs externas ni le expliques cómo usar un buscador.\n"
                "4. NUNCA incluyas enlaces, URLs ni corchetes [1], [2] de fuentes de internet.\n"
                "5. Sé natural, conversacional y conciso.\n\n"
                f"{contexto_clima}\n\n"
                f"Consulta del usuario: {message}"
            )

            chat_response = litellm.completion(
                model=TEXT_MODEL,
                api_key=api_key,
                base_url="https://api.groq.com/openai/v1",
                messages=[{"role": "user", "content": prompt_respuesta_clima}]
            )

            return AssistantResponse(
                intent="WEATHER",
                response=chat_response.choices[0].message.content
            )

        except Exception as e:
            return AssistantResponse(
                intent="WEATHER",
                response=f"No pude consultar la información meteorológica en este momento: {str(e)}"
            )

    # --- 2. INTENCIÓN: OPTIMIZACIÓN DE CV ---
    elif "CV_OPTIMIZATION" in user_intent:
        try:
            extraction = litellm.completion(
                model=TEXT_MODEL,
                api_key=api_key,
                base_url="https://api.groq.com/openai/v1",
                messages=[
                    {"role": "system", "content": "Extrae exclusivamente la descripción detallada del puesto de trabajo o los requisitos del mensaje del usuario. Elimina saludos o textos irrelevantes."},
                    {"role": "user", "content": message}
                ],
                temperature=0.1
            )
            oferta_laboral = extraction.choices[0].message.content.strip()

            llm_groq = LLM(
                model=TEXT_MODEL,  
                base_url="https://api.groq.com/openai/v1", 
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
        
    # --- 3. INTENCIÓN: CONTROL DOMÓTICO ---
    elif "DOMOTICS_CONTROL" in user_intent:
        return AssistantResponse(
            intent="DOMOTICS_CONTROL",
            response="Entendido, Alexis. Conectando con los sistemas de domótica... (Módulo IoT en desarrollo)."
        )
        
    # --- 4. INTENCIÓN: BÚSQUEDA WEB GENERAL ---
    elif "WEB_SEARCH" in user_intent:
        query_optima = optimizar_query_busqueda(message, api_key)
        contexto_web = buscar_en_internet(query_optima)
        
        prompt_final = (
            f"Eres AI Alexis, un asistente de inteligencia artificial leal, directo e inteligente (inspirado en J.A.R.V.I.S.).\n"
            f"FECHA ACTUAL DEL SISTEMA: Año 2026.\n\n"
            f"Usa la siguiente información extraída de internet para responder a la pregunta del usuario.\n"
            f"REGLAS:\n"
            f"1. Responde de forma clara, directa y precisa basándote en los datos web.\n"
            f"2. NUNCA digas que no tienes información si los datos web la contienen.\n"
            f"3. NUNCA le pidas al usuario que navegue en la web por su cuenta ni incluyas URLs o corchetes de fuentes [1].\n\n"
            f"--- INFORMACIÓN RECUPERADA DE INTERNET ---\n"
            f"{contexto_web}\n"
            f"------------------------------------------\n\n"
            f"Consulta del usuario: {message}"
        )
        
        chat_response = litellm.completion(
            model=TEXT_MODEL,
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
            messages=[{"role": "user", "content": prompt_final}]
        )
        return AssistantResponse(
            intent="WEB_SEARCH",
            response=chat_response.choices[0].message.content
        )
        
    # --- 5. INTENCIÓN: CHARLA GENERAL Y RAG DOCUMENTAL ---
    else:
        historial_previo = []
        doc = history_collection.find_one({"_id": user_id})
        if doc and "messages" in doc:
            for m in doc["messages"][-6:]:
                historial_previo.append({"role": m["role"], "content": m["content"]})

        contexto_perfil = cv_texto[:1500] if cv_texto else "Sin perfil registrado."
        
        # Inyección de contexto documental recuperado por búsqueda vectorial (RAG)
        contexto_documentos = ""
        if rag_context:
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
                f"REGLAS:\n"
                f"1. Si se proporciona información de la BASE DE CONOCIMIENTO, úsala como fuente principal de verdad técnica.\n"
                f"2. Usa los datos del perfil para responder dudas sobre identidad y experiencia.\n"
                f"3. Responde siempre de forma clara, concisa y en español."
            )
        }
        
        mensajes_para_llm = [prompt_sistema] + historial_previo + [{"role": "user", "content": message}]

        chat_response = litellm.completion(
            model=TEXT_MODEL,
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
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
    api_key = payload.groq_api_key or os.getenv("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(status_code=400, detail="API Key de Groq no configurada.")
    
    user_id = payload.user_id or "guest_user"
    
    db_profile = profiles_collection.find_one({"_id": user_id})
    if db_profile and "cv_text" in db_profile:
        cv_a_procesar = db_profile["cv_text"]
    else:
        cv_a_procesar = payload.cv_text or BASE_CV

    # Inyección semántica RAG
    rag_context = ""
    if payload.message:
        rag_context = vector_search(user_id=user_id, query=payload.message, limit=3)
        
    resultado = await procesar_mensaje_alexis(
        message=payload.message, 
        cv_texto=cv_a_procesar, 
        api_key=api_key, 
        user_id=user_id,
        image_base64=payload.image,
        rag_context=rag_context
    )

    texto_guardado = payload.message or "📸 [Imagen enviada]"
    if payload.image and not payload.message:
        texto_guardado = "📸 [Análisis de imagen]"

    guardar_en_historial(user_id, texto_guardado, resultado.response, resultado.intent)

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