import os
import base64
import httpx

SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID", "")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET", "")
SPOTIFY_REFRESH_TOKEN = os.getenv("SPOTIFY_REFRESH_TOKEN", "")

async def obtener_token_spotify() -> str:
    """Obtiene un access token válido usando el refresh token del usuario."""
    auth_str = f"{SPOTIFY_CLIENT_ID}:{SPOTIFY_CLIENT_SECRET}"
    b64_auth = base64.b64encode(auth_str.encode()).decode()

    url = "https://accounts.spotify.com/api/token"
    headers = {
        "Authorization": f"Basic {b64_auth}",
        "Content-Type": "application/x-www-form-urlencoded"
    }
    data = {
        "grant_type": "refresh_token",
        "refresh_token": SPOTIFY_REFRESH_TOKEN
    }

    async with httpx.AsyncClient() as client:
        res = await client.post(url, headers=headers, data=data)
        if res.status_code == 200:
            return res.json().get("access_token")
        return ""

async def controlar_spotify(accion: str, query_o_uri: str = "") -> str:
    """Controla la reproducción de Spotify (play, pause, next, search)."""
    token = await obtener_token_spotify()
    if not token:
        return "⚠️ Error de autenticación con Spotify."

    headers = {"Authorization": f"Bearer {token}"}
    async with httpx.AsyncClient() as client:
        if accion == "pause":
            res = await client.put("https://api.spotify.com/v1/me/player/pause", headers=headers)
            return "⏸️ Música pausada." if res.status_code in [200, 204] else "No hay dispositivos activos."
        
        elif accion == "play":
            # Si se pasa una URI o búsqueda
            payload = {}
            if query_o_uri:
                if query_o_uri.startswith("spotify:"):
                    payload["context_uri"] = query_o_uri
                else:
                    # Buscar track o playlist rápida
                    search_res = await client.get(
                        f"https://api.spotify.com/v1/search?q={query_o_uri}&type=track&limit=1",
                        headers=headers
                    )
                    if search_res.status_code == 200:
                        items = search_res.json().get("tracks", {}).get("items", [])
                        if items:
                            payload["uris"] = [items[0]["uri"]]
            
            res = await client.put("https://api.spotify.com/v1/me/player/play", headers=headers, json=payload if payload else None)
            return f"🎵 Reproduciendo {query_o_uri or ''}." if res.status_code in [200, 204] else "⚠️ No se encontró un dispositivo activo de Spotify."
            
    return "Comando de Spotify no reconocido."