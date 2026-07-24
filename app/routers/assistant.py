import os
import io
import litellm
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from app.schemas.assistant import AssistantResponse
from ddgs import DDGS
from crewai import Agent, Task, Crew, Process, LLM
from pypdf import PdfReader
from pymongo import MongoClient  # <-- Importamos el driver de MongoDB
from dotenv import load_dotenv  # <-- 1. Importamos el cargador
from pathlib import Path

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
            "Ejemplo: 'Dime el tiempo de Fuerteventura para mañana.' -> 'clima Fuerteventura mañana'\n"
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


# --- PROCESADOR CENTRAL DE INTENCIONES (CREWAI CON MONGO) ---
def procesar_mensaje_alexis(message: str, cv_texto: str, api_key: str) -> AssistantResponse:
    os.environ["GROQ_API_KEY"] = api_key
    os.environ["OPENAI_API_KEY"] = api_key

    system_prompt = (
        "Eres el clasificador de intenciones de AI Alexis.\n"
        "Tu único trabajo es leer el mensaje del usuario y responder ÚNICAMENTE con una de estas cuatro palabras:\n"
        "- 'CV_OPTIMIZATION' (si quiere adaptar su CV, currículum o habla de ofertas de empleo)\n"
        "- 'DOMOTICS_CONTROL' (si habla de controlar luces, sensores, domótica o IoT)\n"
        "- 'WEB_SEARCH' (si pregunta por el clima, previsiones, noticias, resultados, eventos o cualquier dato que requiera internet)\n"
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

    if "CV_OPTIMIZATION" in user_intent:
        try:
            # 1. Extracción de la oferta
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

            # 2. Instanciamos el modelo para CrewAI
            llm_groq = LLM(
                model="openai/llama-3.1-8b-instant",  
                base_url="https://api.groq.com/openai/v1", 
                api_key=api_key
            )

            # 3. Definición de los agentes
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

            # 4. Definición de las tareas
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
        
    elif "DOMOTICS_CONTROL" in user_intent:
        return AssistantResponse(
            intent="DOMOTICS_CONTROL",
            response="Entendido, Alexis. Conectando con los sistemas de domótica... (Módulo IoT en desarrollo)."
        )
        
    elif "WEB_SEARCH" in user_intent:
        query_optima = optimizar_query_busqueda(message, api_key)
        contexto_web = buscar_en_internet(query_optima)
        
        prompt_final = (
            f"Eres AI Alexis, un asistente de inteligencia artificial leal y brillante inspirado en J.A.R.V.I.S.\n"
            f"Responde detalladamente a la solicitud del usuario utilizando como única fuente de verdad la información extraída de internet.\n"
            f"Sé preciso, directo y amigable. Nota: Estamos en el año 2026.\n\n"
            f"Resultados de búsqueda en tiempo real para '{query_optima}':\n{contexto_web}\n\n"
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
        
    else:
        chat_response = litellm.completion(
            model="openai/llama-3.1-8b-instant",
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
            messages=[
                {"role": "system", "content": "Eres AI Alexis, un asistente de IA leal, altamente inteligente y con un toque de ingenio técnico, inspirado en J.A.R.V.I.S. Responde siempre de forma clara, directa y en español."},
                {"role": "user", "content": message}
            ]
        )
        return AssistantResponse(
            intent="GENERAL_CHAT",
            response=chat_response.choices[0].message.content
        )


# --- ENDPOINTS ---
from app.schemas.assistant import AssistantRequest

@router.post("/chat", response_model=AssistantResponse)
def handle_assistant_chat(payload: AssistantRequest):
    api_key = payload.groq_api_key or os.getenv("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(status_code=400, detail="API Key de Groq no configurada.")
    
    user_id = payload.user_id or "alexis_perez_123"
    
    # 🔍 Buscamos en MongoDB Atlas si existe el perfil del usuario
    db_profile = profiles_collection.find_one({"_id": user_id})
    
    if db_profile and "cv_text" in db_profile:
        cv_a_procesar = db_profile["cv_text"]
    else:
        # Fallback si no hay registro aún
        cv_a_procesar = payload.cv_text or BASE_CV
        
    return procesar_mensaje_alexis(payload.message, cv_a_procesar, api_key)


@router.post("/voice", response_model=AssistantResponse)
async def handle_assistant_voice(
    file: UploadFile = File(...), 
    groq_api_key: str = Form(None),
    user_id: str = Form("alexis_perez_123"),
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

        resultado = procesar_mensaje_alexis(texto_transcrito, cv_a_procesar, api_key)
        resultado.response = f"*(Entendí: \"{texto_transcrito}\")*\n\n{resultado.response}"
        return resultado

    except Exception as e:
        if os.path.exists(nombre_archivo_temporal):
            os.remove(nombre_archivo_temporal)
        raise HTTPException(status_code=500, detail=f"Error al procesar la nota de voz: {str(e)}")


# 🌟 ENDPOINT MEJORADO: SUBIDA, EXTRACCIÓN Y GUARDADO EN ATLAS 🌟
@router.post("/upload-cv")
async def handle_upload_cv(
    file: UploadFile = File(...),
    user_id: str = Form("alexis_perez_123")  # ID predeterminado para pruebas
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Formato de archivo inválido. Sube un documento PDF.")
    
    try:
        contenido = await file.read()
        texto_extraido = extraer_texto_pdf(contenido)
        
        if not texto_extraido:
            raise HTTPException(status_code=400, detail="No se pudo extraer texto del PDF.")
        
        # 💾 ¡Guardamos o Actualizamos en MongoDB Atlas en tiempo real!
        profiles_collection.update_one(
            {"_id": user_id},
            {"$set": {
                "filename": file.filename,
                "cv_text": texto_extraido
            }},
            upsert=True  # Si el usuario no existe, lo crea automáticamente (upsert)
        )
        
        return {
            "user_id": user_id,
            "filename": file.filename,
            "status": "success_saved_to_atlas",
            "extracted_text_preview": texto_extraido[:200] + "..."  # Una vista previa
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar y guardar: {str(e)}")