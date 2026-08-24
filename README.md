# 🤖 AI Alexis - Asistente Virtual Inteligente Multimodal[cite: 4]

**AI Alexis** es un ecosistema de asistencia virtual e inteligencia artificial multimodal de nueva generación, inspirado en sistemas avanzados como J.A.R.V.I.S[cite: 4]. Permite interacción conversacional por texto y voz[cite: 4], gestión de múltiples sesiones de chat con persistencia independiente[cite: 1, 2], consulta de información meteorológica y de actualidad en tiempo real con sistema de caché inteligente[cite: 2], análisis visual avanzado con Google Gemini[cite: 2], análisis y optimización automatizada de currículums mediante agentes autónomos[cite: 4], ingesta y búsqueda vectorial semántica sobre documentos propios (RAG)[cite: 4], autenticación multiusuario y memoria conversacional persistente en la nube[cite: 4].

---

## 🚀 Características Principales

* 💬 **Gestión de Múltiples Conversaciones y Sesiones Persistentes:**
  * Soporte completo para hilos y sesiones de chat independientes asociados al usuario (`conversation_id`)[cite: 2].
  * Barra lateral (*sidebar*) interactiva y responsive con selector de chats activos, botón **"+ Nuevo Chat"** y opción de eliminación[cite: 1].
  * Generación dinámica de títulos por conversación a partir del primer mensaje enviado[cite: 2].
  * Persistencia en tiempo real en la colección `conversations` de MongoDB Atlas[cite: 2].

* 📚 **RAG Documental y Búsqueda Vectorial Semántica (MongoDB Atlas Vector Search + FastEmbed):**
  * Ingesta e indexación directa de documentos PDF y archivos de texto (`.txt`) a la base de conocimiento (`knowledge_base`)[cite: 4].
  * Extracción con `pypdf`, fragmentación semántica (*chunking* con solapamiento) y generación de embeddings densos de 384 dimensiones mediante `BAAI/bge-small-en-v1.5` con `fastembed`[cite: 4].
  * Búsqueda por similitud del coseno e inyección dinámica del contexto documental relevante en las consultas de chat y notas de voz[cite: 4].
  * Aislamiento multitenant: filtrado vectorial estricto por usuario (`user_id`)[cite: 4].

* 👁️ **Visión Multimodal y Análisis de Imágenes (Google GenAI SDK):**
  * Subida de imágenes y capturas directamente desde la interfaz (botón clip 📎) con previsualización interactiva y compresión automática[cite: 1].
  * Procesamiento visual mediante **Gemini 3.6 Flash** a través del SDK oficial de Google GenAI (`google-genai`)[cite: 2].
  * Análisis visual de alta precisión: OCR, lectura de tablas, esquemas, diagramas y descripción técnica de elementos visuales[cite: 4].

* 🔐 **Autenticación Multiusuario y Control de Acceso (Clerk):**
  * Inicio de sesión seguro mediante cuentas de Google u otros proveedores OAuth / Email[cite: 4].
  * Gestión de tokens y sesiones protegidas con `@clerk/nextjs` y Middleware en Next.js[cite: 4].
  * Aislamiento total de datos, historiales, perfiles y vectores por ID único de usuario (`userId`)[cite: 4].

* ⚡ **Caché en Memoria con TTL (Time-To-Live):**
  * Sistema de almacenamiento temporal en memoria para optimizar latencias y evitar consumos innecesarios de cuotas API[cite: 2].
  * Clima: 10 minutos de expiración (`600s`)[cite: 2].
  * Noticias: 5 minutos de expiración (`300s`)[cite: 2].
  * Búsquedas web: 30 minutos de expiración (`1800s`)[cite: 2].

* 🌤️ **Servicio Meteorológico en Tiempo Real (Open-Meteo):**
  * Geocodificación automática de municipios y ciudades mediante la API de Geocoding de Open-Meteo[cite: 2].
  * Consulta de temperaturas (máximas y mínimas), sensación térmica, humedad, viento y probabilidad de precipitación para el día actual o pronóstico de mañana[cite: 2].
  * Detección por palabras clave y clasificación precisa[cite: 2].

* 📰 **Búsqueda de Noticias y Actualidad (NewsAPI + Fallback Web):**
  * Extracción de titulares de última hora y noticias por temática o sector mediante **NewsAPI**[cite: 2, 3].
  * Sistema de contingencia (*fallback*) automático: si la API no está disponible o no tiene clave configurada, redirige la búsqueda hacia DuckDuckGo / Tavily sin interrumpir el servicio[cite: 2].

* 🌐 **Búsqueda Web en Tiempo Real (Tavily Search API + DuckDuckGo):**
  * Clasificador inteligente de intenciones unificado (`WEATHER`, `SEARCH`, `CV_OPTIMIZATION`, `DOMOTICS_CONTROL`, `GENERAL_CHAT`)[cite: 2].
  * Integración con **Tavily Search API** y **DuckDuckGo Search** para la extracción directa y precisa de información actualizada en internet[cite: 2, 4].
  * Optimizador de consultas para transformar lenguaje natural en palabras clave eficientes[cite: 2, 4].

* 📄 **Indexación y Optimización de CV con Agentes Autónomos (CrewAI):**
  * Subida y extracción directa de documentos PDF a la colección `profiles` de MongoDB Atlas[cite: 2, 4].
  * Orquestación de una **Crew de Agentes de IA** (*Reclutador Técnico Senior* + *Consultor y Redactor Técnico*) para adaptar el perfil profesional según ofertas laborales específicas y filtros ATS[cite: 2, 4].

* 🎙️ **Procesamiento de Voz en Tiempo Real (Whisper Large v3):**
  * Botón *Push-to-Talk* integrado en la interfaz móvil y de escritorio[cite: 1, 4].
  * Transcripción de audio mediante `groq/whisper-large-v3`[cite: 2, 4].

---

## 🛠️ Tecnologías Utilizadas

### **Frontend**
* **Framework:** Next.js (React / App Router)[cite: 4].
* **Renderizado de Contenido:** `react-markdown` con soporte completo de **`remark-gfm`** (tablas Markdown, listas estructuradas y bloques de código)[cite: 1].
* **Autenticación:** Clerk (`@clerk/nextjs`)[cite: 4].
* **Estilos:** Tailwind CSS con tema oscuro (*Dark Mode*)[cite: 4].
* **Audio & Multimedia:** MediaRecorder API (audio WebM) y FileReader / HTML Canvas (redimensionamiento y compresión en Base64)[cite: 1, 4].

### **Backend**
* **Framework:** FastAPI (Python)[cite: 4].
* **Modelos de Inferencia:**
  * **Texto, RAG, Clasificación y Agentes:** `groq/openai/gpt-oss-120b`.
  * **Visión Multimodal:** `gemini-3.6-flash` (Google GenAI SDK)[cite: 2].
  * **Voz a Texto:** `groq/whisper-large-v3`[cite: 2, 4].
* **Pipeline RAG & Búsqueda Vectorial:**
  * **Motor Vectorial:** MongoDB Atlas Vector Search (índice `vector_index`, similitud coseno)[cite: 4].
  * **Generación de Embeddings:** `fastembed` con modelo `BAAI/bge-small-en-v1.5` (384 dimensiones)[cite: 4].
  * **Extracción Documental:** `pypdf`[cite: 2, 4].
* **Orquestación Multi-Agente:** CrewAI[cite: 2, 4].
* **Base de Datos:** MongoDB Atlas Cloud (PyMongo) con colecciones `conversations`, `chat_history`, `profiles` y `knowledge_base`[cite: 2, 4].
* **APIs Externas:** Open-Meteo API, NewsAPI, Tavily Search API y DuckDuckGo Search[cite: 2, 3, 4].

### **Infraestructura & Cloud**
* **Hosting Backend & Frontend:** Render Cloud Platform[cite: 4].
* **Base de Datos y Motor Vectorial:** MongoDB Atlas[cite: 4].
* **Control de Versiones:** Git & GitHub[cite: 4].

---

## 📁 Estructura del Proyecto

```text
ai-alexis/
├── ai-alexis-backend/               # Servidor FastAPI
│   ├── app/
│   │   ├── routers/
│   │   │   └── assistant.py         # Endpoints de chat, sesiones, voz, visión, RAG y CV
│   │   ├── schemas/
│   │   │   └── assistant.py         # Esquemas Pydantic con soporte de conversation_id
│   │   ├── services/
│   │   │   ├── rag_service.py       # Chunking, FastEmbed y Vector Search en Atlas
│   │   │   ├── weather_service.py   # Geocoding y pronósticos con Open-Meteo
│   │   │   └── news_service.py      # Búsqueda de noticias con NewsAPI
│   │   └── main.py                  # Punto de entrada de FastAPI
│   ├── .env                         # Variables de entorno del backend
│   └── requirements.txt             # Dependencias de Python
│
└── ai-alexis-frontend/              # Aplicación Next.js
    ├── src/
    │   ├── app/
    │   │   ├── layout.js            # Provider de Clerk y estilos globales
    │   │   └── page.js              # Sidebar de chats, visor multimedia, botón RAG y voz
    │   └── middleware.js            # Interceptor de seguridad Clerk
    ├── .env.local                   # Variables de entorno del frontend
    └── package.json
```

---

## ⚙️ Variables de Entorno

### **Backend (`.env` / Render)**
```ini
GROQ_API_KEY=tu_api_key_de_groq
GEMINI_API_KEY=tu_api_key_de_google_ai_studio
MONGODB_URI=mongodb+srv://<usuario>:<password>@cluster0.mongodb.net/?retryWrites=true&w=majority
TAVILY_API_KEY=tu_api_key_de_tavily
NEWS_API_KEY=tu_api_key_de_newsapi
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
| `POST` | `/api/v1/assistant/chat` | Procesa mensajes de texto/imagen asociados a un `conversation_id`[cite: 2]. |
| `POST` | `/api/v1/assistant/voice` | Recibe audio WebM, transcribe con Whisper, consulta RAG y responde[cite: 2]. |
| `GET`  | `/api/v1/assistant/conversations/{user_id}` | Lista todas las sesiones de chat del usuario ordenadas por fecha[cite: 2]. |
| `GET`  | `/api/v1/assistant/conversation/{conversation_id}` | Recupera el historial completo de mensajes de un chat específico[cite: 2]. |
| `DELETE` | `/api/v1/assistant/conversation/{conversation_id}` | Elimina una sesión de chat y sus mensajes asociados[cite: 2]. |
| `POST` | `/api/v1/assistant/upload-document` | Indexa PDFs o TXTs en MongoDB Atlas generando embeddings vectoriales[cite: 2]. |
| `POST` | `/api/v1/assistant/upload-cv` | Extrae y almacena el CV en PDF asociado al usuario en MongoDB Atlas[cite: 2]. |

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

1. **Backend (Web Service):** Conectar repositorio GitHub, configurar Python 3.11+, comando `uvicorn app.main:app --host 0.0.0.0 --port $PORT` y definir las variables `GROQ_API_KEY`, `GEMINI_API_KEY`, `MONGODB_URI`, `TAVILY_API_KEY` y `NEWS_API_KEY`.
2. **Frontend (Web Service / Static Site):** Conectar repositorio Next.js y configurar `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY`, `CLERK_SECRET_KEY` y `NEXT_PUBLIC_API_URL`.

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