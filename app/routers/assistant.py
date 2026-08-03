import os
import io
import litellm
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from app.schemas.assistant import AssistantRequest, AssistantResponse
from ddgs import DDGS
from crewai import Agent, Task, Crew, Process, LLM
from pypdf import PdfReader
from pymongo import MongoClient  # Driver de MongoDB
from dotenv import load_dotenv
from pathlib import Path
from app.services.weather_service import obtener_clima_open_meteo # Servicio de Clima Open-Meteo

# FORZAMOS LA RUTA ABSOLUTA AL ARCHIVO .env (subiendo desde app/routers/ hasta la raíz)
ruta_raiz = Path(__file__).resolve().parent.parent.parent
ruta_env = ruta_raiz / ".env"

# Cargamos el archivo apuntando directamente a su ubicación real
load_dotenv(dotenv_path=ruta_env)

router = APIRouter(prefix="/assistant", tags=["AI Alexis Assistant"])

# --- CONEXIÓN A MONGODB ATLAS ---
MONGODB_URI = os.getenv("MONGODB_URI")
if not MONGODB_URI:
    print("❌ ERROR CRÍTICO: MONGODB_URI no se está leyendo del archivo .env")
    print(f"👉 Busqué el archivo en la ruta física: {ruta_env}")
    MONGODB_URI = "mongodb://localhost:27017"

client = MongoClient(MONGODB_URI)
db = client["ai_alexis_db"]
profiles_collection = db["profiles"]
history_collection = db["chat_history"]  # 💾 COLECCIÓN PARA HISTORIAL DE CONVERSACIONES


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


# --- PERFIL PROFESIONAL BASE DE ALEXIS (Respaldo por si Atlas está vacío) ---
BASE_CV = """
Nombre Completo: Alexis Fernando Pérez Yamasque
Ubicación: Vigo, Galicia
Perfil Profesional: 
Ingeniero de IA y Estudiante de Big Data con sólida formación en desarrollo multiplataforma y administración de sistemas. Especializado en el diseño, entrenamiento y despliegue de modelos de Machine Learning y Deep Learning, así como en la estructuración de flujos de datos a gran escala.

Experiencia Laboral:
- Programador Técnico Industrial en ESYCONTROL (Mayo 2022 – Septiembre 2023).
  Desarrollo de aplicaciones de extremo a extremo (.NET), gestión de lógica de negocio en backend y administración de bases de datos.

Educación y Certificaciones:
- Especialidad en Inteligencia Artificial y Big Data (En curso, 2026).
- Curso FUNDAE Nivel 6: Backend Avanzado (Completado en Enero 2025).
- Técnico Superior en Desarrollo de Aplicaciones Multiplataforma (DAM) - Colegio Vivas (2020 – 2022).
- Técnico Superior en Administración de Sistemas Informáticos en Red (ASIR) - Daniel Castelao (2018 – 2020).

Habilidades Técnicas:
- IA & Machine Learning: Python, PyTorch, Keras, TensorFlow, scikit-learn.
- Big Data: Apache Spark, Hadoop, HDFS, Hive, Sqoop, PySpark.
- Desarrollo & Backend: .NET, Python, desarrollo de APIs robustas.
- IoT & Domótica: Configuración de sensores Zigbee, despliegue de brokers MQTT, automatización de dispositivos.
- Herramientas: Control de versiones con GitHub, gestión de dependencias con 'uv' package manager.

Idiomas:
- Español (Nativo)
- Gallego (Competencia profesional)
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
            model="openai/llama-3.1-8b-instant",
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
    try:
        with DDGS() as ddgs:
            resultados = list(ddgs.text(query, max_results=3))
            if not resultados:
                return "No se encontraron resultados relevantes."
            contexto = ""
            for i, r in enumerate(resultados, 1):
                titulo = r.get("title", "Sin título")
                cuerpo = r.get("body", r.get("snippet", "Sin descripción"))
                contexto += f"[{i}] Fuente: {titulo}\nInformación: {cuerpo}\n\n"
            return contexto
    except Exception as e:
        return f"Error de conexión con el motor de búsqueda: {str(e)}"


# --- PROCESADOR CENTRAL DE INTENCIONES (ASÍNCRONO CON CLIMA Y MONGO) ---
async def procesar_mensaje_alexis(message: str, cv_texto: str, api_key: str, user_id: str = "guest_user") -> AssistantResponse:
    os.environ["GROQ_API_KEY"] = api_key
    os.environ["OPENAI_API_KEY"] = api_key

    system_prompt = (
        "Eres el clasificador de intenciones de AI Alexis.\n"
        "Tu único trabajo es leer el mensaje del usuario y responder ÚNICAMENTE con una de estas cinco palabras:\n"
        "- 'WEATHER' (si pregunta por el tiempo, clima, temperatura, grados, predicción meteorológica o lluvia)\n"
        "- 'CV_OPTIMIZATION' (si quiere adaptar su CV, currículum o habla de ofertas de empleo)\n"
        "- 'DOMOTICS_CONTROL' (si habla de controlar luces, sensores, domótica o IoT)\n"
        "- 'WEB_SEARCH' (noticias, resultados deportivos, eventos o datos en tiempo real que NO sean el clima)\n"
        "- 'GENERAL_CHAT' (para saludos, charla casual, programación o dudas de conocimiento estático)\n\n"
        "REGLA ESTRICTA: Responde SOLO con la palabra exacta en mayúsculas, sin introducciones ni explicaciones."
    )

    classification = litellm.completion(
        model="openai/llama-3.1-8b-instant",
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

    # --- 1. INTENCIÓN: METEOROLOGÍA / CLIMA (OPEN-METEO) ---
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
                model="openai/llama-3.1-8b-instant",
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
                model="openai/llama-3.1-8b-instant",
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
                model="openai/llama-3.1-8b-instant",
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
                model="openai/llama-3.1-8b-instant",  
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
            f"Eres AI Alexis, un asistente de inteligencia artificial leal y brillante inspirado en J.A.R.V.I.S.\n"
            f"Sintetiza la información relevante para responder al usuario directamente.\n"
            f"REGLAS: Sé preciso, directo y profesional. NUNCA le pidas al usuario que navegue en la web por su cuenta.\n\n"
            f"Resultados de búsqueda:\n{contexto_web}\n\n"
            f"Mensaje original del usuario: {message}"
        )
        
        chat_response = litellm.completion(
            model="openai/llama-3.1-8b-instant",
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
            messages=[{"role": "user", "content": prompt_final}]
        )
        return AssistantResponse(
            intent="WEB_SEARCH",
            response=chat_response.choices[0].message.content
        )
        
    # --- 5. INTENCIÓN: CHARLA GENERAL (CON MEMORIA CONVERSACIONAL) ---
    else:
        # Recuperamos los últimos 6 mensajes guardados en Atlas
        historial_previo = []
        doc = history_collection.find_one({"_id": user_id})
        if doc and "messages" in doc:
            for m in doc["messages"][-6:]:
                historial_previo.append({"role": m["role"], "content": m["content"]})

        # Extraemos información de tu perfil/CV guardado en Atlas
        contexto_perfil = cv_texto[:1500] if cv_texto else "Sin perfil registrado."

        prompt_sistema = {
            "role": "system",
            "content": (
                f"Eres AI Alexis, un asistente de IA leal, altamente inteligente y con un toque de ingenio técnico, inspirado en J.A.R.V.I.S.\n"
                f"Estás hablando con el usuario registrado (ID de sesión: {user_id}).\n"
                f"--- DATOS DEL PERFIL DEL USUARIO ---\n"
                f"{contexto_perfil}\n"
                f"------------------------------------\n\n"
                f"REGLA: Usa esta información para saber exactamente con quién hablas cuando te pregunte por su identidad o perfil. Responde siempre de forma clara, directa y en español."
            )
        }
        
        mensajes_para_llm = [prompt_sistema] + historial_previo + [{"role": "user", "content": message}]

        chat_response = litellm.completion(
            model="openai/llama-3.1-8b-instant",
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
            messages=mensajes_para_llm
        )
        return AssistantResponse(
            intent="GENERAL_CHAT",
            response=chat_response.choices[0].message.content
        )


# --- ENDPOINTS ---

# 📜 RECUPERAR HISTORIAL DE CHAT DESDE MONGO ATLAS
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
    
    # 🔍 Buscamos en MongoDB Atlas si existe el perfil del usuario
    db_profile = profiles_collection.find_one({"_id": user_id})
    
    if db_profile and "cv_text" in db_profile:
        cv_a_procesar = db_profile["cv_text"]
    else:
        cv_a_procesar = payload.cv_text or BASE_CV
        
    resultado = await procesar_mensaje_alexis(payload.message, cv_a_procesar, api_key, user_id)

    # 💾 Guardamos la interacción en MongoDB Atlas automáticamente
    guardar_en_historial(user_id, payload.message, resultado.response, resultado.intent)

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

    try:
        nombre_archivo_temporal = f"temp_{file.filename}"
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

        # 🔍 Buscamos en MongoDB Atlas para la inyección de voz
        db_profile = profiles_collection.find_one({"_id": user_id})
        
        if db_profile and "cv_text" in db_profile:
            cv_a_procesar = db_profile["cv_text"]
        else:
            cv_a_procesar = cv_text or BASE_CV

        resultado = await procesar_mensaje_alexis(texto_transcrito, cv_a_procesar, api_key, user_id)
        respuesta_formateada = f"*(Entendí: \"{texto_transcrito}\")*\n\n{resultado.response}"
        
        # 💾 Guardamos también las notas de voz en el historial de Atlas
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
        
        # 💾 Guardamos o Actualizamos en MongoDB Atlas en tiempo real asociando al userId de Clerk
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