import os
import httpx

NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")

async def buscar_noticias(query: str = "", pais: str = "es", max_articulos: int = 4) -> str:
    """Busca noticias de última hora o por término clave usando NewsAPI."""
    if not NEWS_API_KEY:
        return "⚠️ La variable NEWS_API_KEY no está configurada en el servidor."

    headers = {"X-Api-Key": NEWS_API_KEY}
    
    # Si hay una temática concreta usamos 'everything', si no, 'top-headlines'
    if query and query.strip().lower() not in ["actualidad", "noticias", "titulares", "ultimas noticias"]:
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": query,
            "language": "es",
            "sortBy": "publishedAt",
            "pageSize": max_articulos
        }
    else:
        url = "https://newsapi.org/v2/top-headlines"
        params = {
            "country": pais,
            "pageSize": max_articulos
        }

    try:
        async with httpx.AsyncClient() as client:
            res = await client.get(url, headers=headers, params=params, timeout=10.0)
            if res.status_code == 200:
                data = res.json()
                articulos = data.get("articles", [])
                if not articulos:
                    return f"No encontré noticias recientes sobre '{query}'."

                noticias_formateadas = []
                for art in articulos:
                    titulo = art.get("title", "Sin título")
                    fuente = art.get("source", {}).get("name", "Fuente desconocida")
                    desc = art.get("description") or "Sin descripción disponible."
                    url_art = art.get("url", "")
                    noticias_formateadas.append(f"* **{titulo}** ({fuente})\n  {desc}\n  [Leer más]({url_art})")

                return "\n\n".join(noticias_formateadas)
            elif res.status_code == 401:
                return "⚠️ Error de autenticación: API Key de NewsAPI inválida."
            else:
                return f"Error en NewsAPI: {res.status_code}."
    except Exception as e:
        return f"Error al conectar con el servicio de noticias: {str(e)}"