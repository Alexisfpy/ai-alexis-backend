# 🤖 AI Alexis - Asistente Virtual Inteligente Multimodal

**AI Alexis** es un ecosistema de asistencia virtual e inteligencia artificial multimodal de nueva generación, inspirado en sistemas avanzados como J.A.R.V.I.S. Permite interacción conversacional por texto y voz, consulta de información meteorológica y web en tiempo real, análisis y optimización automatizada de currículums mediante un equipo de agentes autónomos, autenticación multiusuario y memoria conversacional persistente en la nube.

---

## 🚀 Características Principales

* 🔐 **Autenticación Multiusuario y Control de Acceso (Clerk):**
  * Inicio de sesión seguro mediante cuentas de Google u otros proveedores OAuth / Email.
  * Gestión de tokens y sesiones protegidas con `@clerk/nextjs` y Middleware en Next.js.
  * Aislamiento total de datos y perfiles por ID único de usuario (`userId`).

* 🧠 **Memoria Conversacional Persistente (MongoDB Atlas Cloud):**
  * Guardado automático de todas las interacciones (texto, voz e intenciones) en la nube.
  * Recuperación dinámica del historial al iniciar sesión o recargar la pantalla desde cualquier dispositivo.
  * Inyección del contexto histórico conversacional y del perfil del usuario en las respuestas de Llama 3.

* 🎙️ **Procesamiento de Voz en Tiempo Real (Whisper Large v3):**
  * Botón *Push-to-Talk* ("Mantener para hablar") integrado directamente en la interfaz.
  * Transcripción de audio de alta precisión mediante `groq/whisper-large-v3`.
  * Conversión automática de notas de voz a texto e integración con la memoria de la IA.

* 📄 **Indexación y Optimización de CV con Agentes Autónomos (CrewAI):**
  * Subida y extracción directa de documentos PDF.
  * Almacenamiento permanente del texto del CV en la colección `profiles` de MongoDB Atlas.
  * Orquestación de una **Crew de Agentes de IA** (*Reclutador Técnico Senior* + *Consultor y Redactor Técnico*) para adaptar el currículum del usuario según ofertas laborales específicas y filtros ATS.

* 🌤️ **Servicio Meteorológico en Tiempo Real (Open-Meteo):**
  * Detección automática de ubicaciones mediante LLM.
  * Consulta directa a la API de Open-Meteo para obtener temperaturas (máxima y mínima) y probabilidad de precipitación sin depender de búsquedas web genéricas.

* 🌐 **Búsqueda Web en Tiempo Real (Tavily Search API + Optimización LLM):**
  * Clasificador inteligente de intenciones (`WEATHER`, `CV_OPTIMIZATION`, `DOMOTICS_CONTROL`, `WEB_SEARCH`, `GENERAL_CHAT`).
  * Integración con **Tavily Search API** para la extracción directa y precisa de información en tiempo real sin bloqueos en la nube.
  * Optimizador de consultas para transformar lenguaje natural en palabras clave eficientes de búsqueda y síntesis de resultados actualizados.

---

## 🛠️ Tecnologías Utilizadas

### **Frontend**
* **Framework:** Next.js (React / App Router).
* **Autenticación:** Clerk (`@clerk/nextjs`).
* **Estilos:** Tailwind CSS con tema oscuro (*Dark Mode*).
* **Audio:** MediaRecorder API (grabación WebM).

### **Backend**
* **Framework:** FastAPI (Python).
* **Modelos LLM:** Groq / LiteLLM (`llama-3.1-8b-instant`, `groq/whisper-large-v3`).
* **Orquestación Multi-Agente:** CrewAI.
* **Base de Datos:** MongoDB Atlas Cloud (PyMongo).
* **Búsquedas & Clima:** Tavily Search API (`tavily-python`), Open-Meteo API, DuckDuckGo Search (`duckduckgo_search`).

### **Infraestructura & Cloud**
* **Hosting App & API:** Render Cloud Platform.
* **Base de Datos en la Nube:** MongoDB Atlas.
* **Control de Versiones:** Git & GitHub.

---

## 📁 Estructura del Proyecto

```text
ai-alexis/
├── ai-alexis-backend/               # Servidor FastAPI
│   ├── app/
│   │   ├── routers/
│   │   │   └── assistant.py         # Endpoints principal, chat, voz, CV e historial
│   │   ├── schemas/
│   │   │   └── assistant.py         # Esquemas Pydantic
│   │   ├── services/
│   │   │   └── weather_service.py   # Integración con Open-Meteo API
│   │   └── main.py                  # Punto de entrada de FastAPI
│   ├── .env                         # Variables de entorno del backend
│   └── requirements.txt             # Dependencias de Python
│
└── ai-alexis-frontend/              # Aplicación Next.js
    ├── src/
    │   ├── app/
    │   │   ├── layout.js            # Provider de Clerk y fuentes
    │   │   └── page.js              # Interfaz del chat, voz y subida de archivos
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
| `POST` | `/api/v1/assistant/voice` | Recibe audio WebM, transcribe con Whisper y responde. |
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
* 🧠 **Memoria Persistente:** MongoDB Atlas Cloud (historial e identidad por `user_id`).
* ⚡ **Inferencia de Alto Rendimiento:** Groq (`Llama 3.1 8B Instant`).
* 🌐 **Búsqueda Web en Tiempo Real:** Tavily Search API.
* 🎙️ **Transcripción de Voz:** Whisper Large v3.
* 🤖 **Agentes Autónomos:** CrewAI para optimización de CVs.
* 🌤️ **Servicio Meteorológico:** Open-Meteo API.