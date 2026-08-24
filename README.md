# 🤖 AI Alexis - Asistente Virtual Inteligente Multimodal

**AI Alexis** es un ecosistema de asistencia virtual e inteligencia artificial multimodal de nueva generación, inspirado en sistemas avanzados como J.A.R.V.I.S. Permite interacción conversacional por texto y voz, consulta de información meteorológica y web en tiempo real, análisis y optimización automatizada de currículums mediante agentes autónomos, ingesta y búsqueda vectorial semántica sobre documentos propios (RAG), autenticación multiusuario y memoria conversacional persistente en la nube.

---

## 🚀 Características Principales

* 📚 **RAG Documental y Búsqueda Vectorial Semántica (MongoDB Atlas Vector Search + FastEmbed):**
  * Ingesta e indexación directa de documentos PDF y archivos de texto (`.txt`) a la base de conocimiento (`knowledge_base`).
  * Extracción con `pypdf`, fragmentación semántica (*chunking* con solapamiento) y generación de embeddings densos de 384 dimensiones mediante `BAAI/bge-small-en-v1.5` con `fastembed`.
  * Búsqueda por similitud del coseno e inyección dinámica del contexto documental relevante en las consultas de chat y notas de voz.
  * Aislamiento multitenant: filtrado vectorial estricto por usuario (`user_id`).

* 👁️ **Visión Multimodal y Análisis de Imágenes:**
  * Subida de imágenes y capturas directamente desde la interfaz (botón clip 📎).
  * Conversión automática a Base64 y previsualización interactiva antes del envío.
  * Análisis visual de alta precisión: OCR, lectura de tablas, esquemas, diagramas y descripción técnica de elementos visuales.
  * Filtro por expresiones regulares para eliminar reflexiones internas (`<think>`) y renderizado Markdown limpio con `react-markdown`.

* 🔐 **Autenticación Multiusuario y Control de Acceso (Clerk):**
  * Inicio de sesión seguro mediante cuentas de Google u otros proveedores OAuth / Email.
  * Gestión de tokens y sesiones protegidas con `@clerk/nextjs` y Middleware en Next.js.
  * Aislamiento total de datos, historiales, perfiles y vectores por ID único de usuario (`userId`).

* 🧠 **Memoria Conversacional Persistente (MongoDB Atlas Cloud):**
  * Guardado automático de todas las interacciones (texto, imágenes, voz e intenciones) en la nube.
  * Recuperación dinámica del historial al iniciar sesión o recargar la pantalla desde cualquier dispositivo.
  * Inyección del contexto histórico conversacional y del perfil del usuario en las respuestas del asistente.

* 🎙️ **Procesamiento de Voz en Tiempo Real (Whisper Large v3):**
  * Botón *Push-to-Talk* ("Mantener para hablar") integrado directamente en la interfaz.
  * Transcripción de audio de alta precisión mediante `groq/whisper-large-v3`.
  * Conversión automática de notas de voz a texto, búsqueda RAG y registro en la memoria del asistente.

* 📄 **Indexación y Optimización de CV con Agentes Autónomos (CrewAI):**
  * Subida y extracción directa de documentos PDF.
  * Almacenamiento permanente del texto del CV en la colección `profiles` de MongoDB Atlas.
  * Orquestación de una **Crew de Agentes de IA** (*Reclutador Técnico Senior* + *Consultor y Redactor Técnico*) para adaptar el currículum del usuario según ofertas laborales específicas y filtros ATS.

* 🌤️ **Servicio Meteorológico en Tiempo Real (Open-Meteo):**
  * Detección automática de ubicaciones mediante LLM.
  * Consulta directa a la API de Open-Meteo para obtener temperaturas (máxima y mínima) y probabilidad de precipitación sin depender de búsquedas web genéricas.

* 🌐 **Búsqueda Web en Tiempo Real (Tavily Search API + Optimización LLM):**
  * Clasificador inteligente de intenciones (`IMAGE_ANALYSIS`, `WEATHER`, `CV_OPTIMIZATION`, `DOMOTICS_CONTROL`, `WEB_SEARCH`, `GENERAL_CHAT`).
  * Integración con **Tavily Search API** para la extracción directa y precisa de información en tiempo real sin bloqueos en la nube.
  * Optimizador de consultas para transformar lenguaje natural en palabras clave eficientes de búsqueda y síntesis de resultados actualizados.

---

## 🛠️ Tecnologías Utilizadas

### **Frontend**
* **Framework:** Next.js (React / App Router).
* **Renderizado de Contenido:** `react-markdown` (formato enriquecido con negritas, listas y bloques de código).
* **Autenticación:** Clerk (`@clerk/nextjs`).
* **Estilos:** Tailwind CSS con tema oscuro (*Dark Mode*).
* **Audio & Multimedia:** MediaRecorder API (grabación WebM) y FileReader API (procesamiento de imágenes en Base64).

### **Backend**
* **Framework:** FastAPI (Python).
* **Modelos de Inferencia (Groq / LiteLLM):**
  * **Texto, Clasificación y Chat:** `groq/openai/gpt-oss-120b` (vía Groq API).
  * **Visión Multimodal (OCR & Análisis de Imágenes):** Modelo de visión activo (vía Groq API).
  * **Voz a Texto:** `groq/whisper-large-v3`.
* **Pipeline RAG & Búsqueda Vectorial:**
  * **Motor Vectorial:** MongoDB Atlas Vector Search (índice `vector_index`, similitud coseno).
  * **Generación de Embeddings:** `fastembed` con modelo `BAAI/bge-small-en-v1.5` (384 dimensiones).
  * **Extracción Documental:** `pypdf`.
* **Orquestación Multi-Agente:** CrewAI.
* **Base de Datos:** MongoDB Atlas Cloud (PyMongo).
* **Búsquedas & Clima:** Tavily Search API (`tavily-python`), Open-Meteo API, DuckDuckGo Search (`duckduckgo_search`).

### **Infraestructura & Cloud**
* **Hosting App & API:** Render Cloud Platform.
* **Base de Datos y Motor Vectorial:** MongoDB Atlas.
* **Control de Versiones:** Git & GitHub.

---

## 📁 Estructura del Proyecto

```text
ai-alexis/
├── ai-alexis-backend/               # Servidor FastAPI
│   ├── app/
│   │   ├── routers/
│   │   │   └── assistant.py         # Endpoints de chat, visión, voz, RAG, CV e historial
│   │   ├── schemas/
│   │   │   └── assistant.py         # Esquemas Pydantic (Request con soporte multimodal y Response)
│   │   ├── services/
│   │   │   ├── rag_service.py       # Chunking, embeddings con FastEmbed y Vector Search en Atlas
│   │   │   └── weather_service.py   # Integración con Open-Meteo API
│   │   └── main.py                  # Punto de entrada de FastAPI
│   ├── .env                         # Variables de entorno del backend
│   └── requirements.txt             # Dependencias de Python (FastAPI, PyMongo, FastEmbed, PyPDF, CrewAI...)
│
└── ai-alexis-frontend/              # Aplicación Next.js
    ├── src/
    │   ├── app/
    │   │   ├── layout.js            # Provider de Clerk y estilos globales
    │   │   └── page.js              # Interfaz de chat, visor de adjuntos, botón RAG y voz
    │   └── middleware.js            # Interceptor de seguridad Clerk
    ├── .env.local                   # Variables de entorno del frontend
    └── package.json
```

---

## ⚙️ Variables de Entorno

### **Backend (`.env`)**
```ini
GROQ_API_KEY=tu_api_key_de_groq
MONGODB_URI=mongodb+srv://<usuario>:<password>@cluster0.mongodb.net/?retryWrites=true&w=majority
TAVILY_API_KEY=tu_api_key_de_tavily
```

### **Frontend (`.env.local`)**
```ini
NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=pk_test_...
CLERK_SECRET_KEY=sk_test_...
NEXT_PUBLIC_API_URL=[https://ai-alexis-backend.onrender.com/api/v1](https://ai-alexis-backend.onrender.com/api/v1)
```

---

## 📌 Endpoints Principales de la API

| Método | Ruta | Descripción |
| :--- | :--- | :--- |
| `POST` | `/api/v1/assistant/chat` | Procesa mensajes de texto e interactúa con el LLM. |
| `POST` | `/api/v1/assistant/voice` | Recibe audio WebM, transcribe con Whisper, consulta RAG y responde. |
| `POST` | `/api/v1/assistant/upload-document` | Indexa PDFs o TXTs en MongoDB Atlas generando embeddings vectoriales (RAG). |
| `POST` | `/api/v1/assistant/upload-cv` | Extrae y guarda el PDF del CV asociado al usuario en MongoDB Atlas. |
| `GET`  | `/api/v1/assistant/history/{user_id}` | Obtiene el historial de mensajes de un usuario desde MongoDB Atlas. |

---

## 💻 Ejecución en Desarrollo Local

1. **Iniciar Backend:**
   ```bash
   cd ai-alexis-backend
   uvicorn app.main:app --reload
   ```

2. **Iniciar Frontend:**
   ```bash
   cd ai-alexis-frontend
   npm run dev
   ```

   Abre [http://localhost:3000](http://localhost:3000) en el navegador.

---

## ☁️ Despliegue en Producción (Render)

1. **Backend (Web Service):** Conectar repositorio GitHub en Render, configurar entorno Python 3.11+, comando de inicio `uvicorn app.main:app --host 0.0.0.0 --port $PORT` y añadir variables `GROQ_API_KEY`, `MONGODB_URI` y `TAVILY_API_KEY`.
2. **Frontend (Web Service / Static Site):** Conectar repositorio de Next.js, añadir variables `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY`, `CLERK_SECRET_KEY` y `NEXT_PUBLIC_API_URL`.

---

## 🏆 Resumen: Arquitectura en Producción

* 🔐 **Autenticación:** Clerk (Google / Email).
* 🧠 **Memoria Persistente & RAG:** MongoDB Atlas Cloud (colecciones `chat_history`, `profiles` y `knowledge_base` con Atlas Vector Search).
* ⚡ **Inferencia LLM:** Groq (`groq/openai/gpt-oss-120b`).
* 📊 **Vectorización: FastEmbed (`BAAI/bge-small-en-v1.5`, 384 dimensiones).**
* 🌐 **Búsqueda Web en Tiempo Real:** Tavily Search API.
* 🎙️ **Transcripción de Voz:** Whisper Large v3.
* 🤖 **Agentes Autónomos:** CrewAI para optimización de CVs.
* 🌤️ **Servicio Meteorológico:** Open-Meteo API.