import os
from fastapi import APIRouter, HTTPException
from crewai import Agent, Task, Crew, Process, LLM
from app.schemas.cv import CVUpdateRequest, CVUpdateResponse

router = APIRouter(prefix="/cv", tags=["CV Optimization"])

@router.post("/optimize", response_model=CVUpdateResponse)
def optimize_cv(payload: CVUpdateRequest):
    # 1. Limpieza de la API Key enviada
    api_key = payload.groq_api_key
    
    # Si viene el marcador por defecto de Swagger ("string") o vacío, lo ignoramos para usar el .env
    if not api_key or api_key.strip() in ["", "string", "null"]:
        api_key = os.getenv("GROQ_API_KEY")

    # Limpiamos posibles comillas dobles o simples que se queden al copiar del .env
    if api_key:
        api_key = api_key.strip().replace('"', '').replace("'", "")

    # Validación de seguridad para la API Key
    if not api_key or api_key.startswith("gsk_tu_clave") or api_key == "string":
        raise HTTPException(
            status_code=400, 
            detail="Falta una API Key de Groq válida. Configúrala en tu archivo .env o envíala en el campo 'groq_api_key'."
        )

    # 2. Validación de los textos de entrada
    cv_text = payload.cv_text
    job_desc = payload.job_description

    if cv_text.strip() in ["", "string"] or job_desc.strip() in ["", "string"]:
        raise HTTPException(
            status_code=400,
            detail="Por favor, rellena los campos 'cv_text' y 'job_description' con información real en lugar de los valores por defecto."
        )

    try:
        # Mapeamos la clave a ambas variables de entorno para evitar confusiones de litellm
        os.environ["GROQ_API_KEY"] = api_key
        os.environ["OPENAI_API_KEY"] = api_key

        # Configuración del LLM
        llm_groq = LLM(
            model="openai/llama-3.1-8b-instant",  
            base_url="https://api.groq.com/openai/v1", 
            api_key=api_key
        )

        # Definición de Agentes
        reclutador = Agent(
            role="Reclutador Técnico Senior",
            goal="Analizar ofertas de empleo y extraer requisitos técnicos esenciales y palabras clave para filtros ATS.",
            backstory="Eres un reclutador experto en el sector tecnológico en España. Sabes identificar qué indexan los filtros automáticos.",
            verbose=True,
            llm=llm_groq
        )

        redactor = Agent(
            role="Consultor de Carreras y Experto en CVs",
            goal="Adaptar el CV del usuario para alinearlo perfectamente con la oferta de trabajo destacando el valor real.",
            backstory="Eres un redactor profesional de currículums técnicos. Sabes alinear perfiles sin inventar jamás información nueva.",
            verbose=True,
            llm=llm_groq
        )

        # Definición de Tareas
        tarea_analisis = Task(
            description=f"Analiza detalladamente esta oferta de empleo:\n\n{job_desc}\n\nExtrae las 5 habilidades clave obligatorias y los puntos extra valorables.",
            expected_output="Un reporte ejecutivo detallando las keywords y necesidades críticas de la empresa.",
            agent=reclutador
        )

        tarea_adaptacion = Task(
            description=f"Coge el CV original del candidato:\n\n{cv_text}\n\nUtilizando el reporte del Reclutador, adapta las secciones de Perfil y Experiencia. Mantén un formato estructurado en Markdown profesional.",
            expected_output="Un currículum adaptado estructurado en Markdown impecable.",
            agent=redactor
        )

        # Orquestación de la Crew
        crew = Crew(
            agents=[reclutador, redactor],
            tasks=[tarea_analisis, tarea_adaptacion],
            process=Process.sequential,
            memory=False,
            cache=False
        )

        resultado = crew.kickoff()
        return CVUpdateResponse(optimized_cv=str(resultado))

    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error durante el análisis del agente: {str(e)}"
        )