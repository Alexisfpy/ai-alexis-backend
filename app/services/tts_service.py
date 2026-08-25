import re
import edge_tts
import io

# Voz recomendada: es-ES-AlvaroNeural (estilo asistente sobrio/tecnológico)
# Alternativas: es-ES-ElviraNeural (femenina), es-ES-DarioNeural (masculina suave)
DEFAULT_VOICE = "es-ES-AlvaroNeural"

def limpiar_texto_para_tts(texto: str) -> str:
    """Elimina formato Markdown, enlaces y caracteres especiales antes de sintetizar."""
    # Eliminar bloques de código
    texto = re.sub(r'```[\s\S]*?```', '', texto)
    texto = re.sub(r'`.*?`', '', texto)
    # Eliminar enlaces markdown [Texto](url) -> Texto
    texto = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', texto)
    # Eliminar cabeceras, negritas, cursivas y viñetas
    texto = re.sub(r'[#*_~>-]', '', texto)
    # Reemplazar múltiples saltos de línea y espacios
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto

async def sintetizar_voz_neural(texto: str, voice: str = DEFAULT_VOICE) -> bytes:
    """Convierte texto en un buffer de audio MP3 usando voces neuronales de Edge TTS."""
    texto_limpio = limpiar_texto_para_tts(texto)
    if not texto_limpio:
        texto_limpio = "No hay contenido para reproducir."

    communicate = edge_tts.Communicate(text=texto_limpio, voice=voice)
    audio_buffer = io.BytesIO()

    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_buffer.write(chunk["data"])

    audio_buffer.seek(0)
    return audio_buffer.read()