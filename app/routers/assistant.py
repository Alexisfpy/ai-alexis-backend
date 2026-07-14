import os
import litellm
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from app.schemas.assistant import AssistantResponse
from ddgs import DDGS

router = APIRouter(prefix="/assistant", tags=["AI Alexis Assistant"])

# --- 1. OPTIMIZADOR INTELIGENTE DE BÚSQUEDAS ---
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
        # Si algo falla en la optimización rápida, devolvemos el mensaje original como plan de respaldo
        return mensaje_usuario

# --- 2. EJECUTOR DE BÚSQUEDAS EN TIEMPO REAL ---
def buscar_en_internet(query: str) -> str:
    try:
        with DDGS() as ddgs:
            resultados = list(ddgs.text(query, max_results=3))
            
            if not resultados:
                return "No se encontraron resultados relevantes en internet para la previsión meteorológica."
            
            contexto = ""
            for i, r in enumerate(resultados, 1):
                titulo = r.get("title", "Sin título")
                cuerpo = r.get("body", r.get("snippet", "Sin descripción"))
                contexto += f"[{i}] Fuente: {titulo}\nInformación: {cuerpo}\n\n"
            return contexto
    except Exception as e:
        return f"Error de conexión con el motor de búsqueda: {str(e)}"

# --- 3. PROCESADOR CENTRAL DE INTENCIONES ---
def procesar_mensaje_alexis(message: str, api_key: str) -> AssistantResponse:
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

    # Clasificación rápida de intención
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
        return AssistantResponse(
            intent="CV_OPTIMIZATION",
            response="He detectado que deseas optimizar tu CV. Por favor, envíame los detalles estructurados a través de mi módulo de optimización."
        )
        
    elif "DOMOTICS_CONTROL" in user_intent:
        return AssistantResponse(
            intent="DOMOTICS_CONTROL",
            response="Entendido, Alexis. Conectando con los sistemas de domótica... (Módulo IoT en desarrollo)."
        )
        
    elif "WEB_SEARCH" in user_intent:
        # A. Optimizamos la query (Dime el tiempo... -> clima Fuerteventura mañana)
        query_optima = optimizar_query_busqueda(message, api_key)
        
        # B. Buscamos la previsión meteorológica real en internet
        contexto_web = buscar_en_internet(query_optima)
        
        # C. El LLM redacta la respuesta definitiva basándose en los datos verídicos obtenidos
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
            messages=[
                {"role": "user", "content": prompt_final}
            ]
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
    return procesar_mensaje_alexis(payload.message, api_key)

@router.post("/voice", response_model=AssistantResponse)
async def handle_assistant_voice(
    file: UploadFile = File(...), 
    groq_api_key: str = Form(None)
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

        resultado = procesar_mensaje_alexis(texto_transcrito, api_key)
        resultado.response = f"*(Entendí: \"{texto_transcrito}\")*\n\n{resultado.response}"
        return resultado

    except Exception as e:
        if os.path.exists(nombre_archivo_temporal):
            os.remove(nombre_archivo_temporal)
        raise HTTPException(status_code=500, detail=f"Error al procesar la nota de voz: {str(e)}")