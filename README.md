# 🤖 AI Alexis Backend - Motor SaaS Multiusuario (J.A.R.V.I.S. Inspired)

Este repositorio contiene el núcleo del backend de **AI Alexis**, un asistente inteligente avanzado inspirado en la filosofía de J.A.R.V.I.S. Diseñado bajo una arquitectura ágil, desacoplada y multicliente (SaaS), el sistema es capaz de clasificar intenciones en tiempo real, ejecutar búsquedas dinámicas en internet, procesar notas de voz y optimizar currículums de forma personalizada utilizando un sistema multiagente.

---

## 🚀 Características Principales & Roadmap Completado

### 📍 Fase 1: Modularización Dinámica (Inyección de Perfil)
- Se ha eliminado la dependencia de perfiles cableados en código (`hardcoded`).
- El enrutador central se ha desacoplado por completo de la lógica de negocio mediante inversión de control, permitiendo inyectar el ADN profesional (`cv_texto`) como un argumento dinámico en cada petición.

### 📂 Fase 2: Pipeline de Ingestión (PDF a Texto)
- Integración de un sistema de extracción binaria de archivos PDF utilizando la librería de alta eficiencia `pypdf`.
- Endpoint de subida optimizado (`/upload-cv`) que convierte currículums en documentos de texto limpio estructurados en JSON en cuestión de milisegundos.

### 💾 Fase 3: Capa de Persistencia Multiusuario (MongoDB Atlas Cloud)
- Transición de un entorno local a un ecosistema multinivel en la nube utilizando **MongoDB Atlas**.
- Implementación de un aislamiento de datos basado en identificadores únicos (`user_id`).
- Sistema de guardado y actualización inteligente (`upsert`) en tiempo real. Cuando un usuario interactúa mediante chat o voz, la inteligencia artificial recupera automáticamente su CV específico de la base de datos cloud, aislando los datos entre diferentes clientes.

---

## 🛠️ Stack Tecnológico

- **Framework principal:** FastAPI (Python)
- **Orquestación de Agentes de IA:** CrewAI (Multi-Agent System: Reclutador Técnico + Redactor de Carreras)
- **Abstracción de LLMs & Audio:** LiteLLM (Integración con Groq Cloud: Llama 3.1 & Whisper-large-v3)
- **Base de Datos Cloud:** MongoDB Atlas (vía PyMongo)
- **Procesamiento de Documentos:** PyPDF
- **Gestor de Entorno:** Python-Dotenv & Pathlib para resolución absoluta de rutas.
- **Package Manager:** `uv` de Astral

---

## 🚦 Endpoints del Asistente (`/assistant`)

El backend expone una interfaz API documentada en Swagger bajo la ruta raíz `/docs`:

1. **`POST /assistant/upload-cv`**: Recibe un archivo binario `.pdf` y un `user_id` (vía Form-Data). Extrae el texto y lo almacena de forma persistente en MongoDB Atlas.
2. **`POST /assistant/chat`**: Recibe el mensaje del usuario y su `user_id`. Clasifica la intención y, si requiere optimización, consulta Atlas para inyectar dinámicamente el currículum del cliente en la CrewAI de agentes.
3. **`POST /assistant/voice`**: Recibe un archivo de audio, lo transcribe con Whisper en tiempo real y ejecuta el flujo inteligente inyectando la persistencia cloud del usuario.

---

## ⚙️ Configuración del Entorno (`.env`)

Para correr este proyecto, es necesario crear un archivo `.env` en la raíz del backend con la siguiente estructura:

```env
# Claves de las APIs de Inteligencia Artificial
GROQ_API_KEY="tu_api_key_de_groq_aqui"

# Cadena de conexión segura de MongoDB Atlas (Cloud)
MONGODB_URI="mongodb+srv://<usuario>:<password>@<cluster>.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
```