import httpx

# Mapeo de códigos WMO de Open-Meteo a descripciones en español
WMO_CODES = {
    0: "Despejado",
    1: "Mayormente despejado",
    2: "Parcialmente nublado",
    3: "Nublado",
    45: "Niebla",
    48: "Niebla con escarcha",
    51: "Llovizna ligera",
    53: "Llovizna moderada",
    55: "Llovizna densa",
    61: "Lluvia ligera",
    63: "Lluvia moderada",
    65: "Lluvia fuerte",
    71: "Nieve ligera",
    73: "Nieve moderada",
    75: "Nieve intensa",
    80: "Chubascos leves",
    81: "Chubascos moderados",
    82: "Chubascos violentos",
    95: "Tormenta eléctrica",
    96: "Tormenta con granizo leve",
    99: "Tormenta con granizo fuerte"
}

async def obtener_coordenadas(ciudad: str) -> tuple[float, float, str] | None:
    """Obtiene latitud, longitud y nombre formateado usando el geocoding gratuito de Open-Meteo."""
    url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {"name": ciudad, "count": 1, "language": "es", "format": "json"}
    
    try:
        async with httpx.AsyncClient() as client:
            res = await client.get(url, params=params, timeout=8.0)
            if res.status_code == 200:
                data = res.json()
                if "results" in data and len(data["results"]) > 0:
                    lugar = data["results"][0]
                    nombre_completo = f"{lugar.get('name')}, {lugar.get('country', '')}"
                    return lugar["latitude"], lugar["longitude"], nombre_completo
    except Exception as e:
        print(f"⚠️ Error en geocoding de Open-Meteo: {e}")
    return None

async def consultar_clima_open_meteo(ciudad: str, dias: int = 0) -> dict:
    """Obtiene el pronóstico para hoy o mañana (dias=0 hoy, dias=1 mañana)."""
    coords = await obtener_coordenadas(ciudad)
    if not coords:
        return {"error": f"No pude encontrar la ubicación '{ciudad}'."}

    lat, lon, ubicacion_nombre = coords
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_probability_mean"],
        "forecast_days": 2,  # para obtener hoy y mañana
        "timezone": "auto"
    }

    try:
        async with httpx.AsyncClient() as client:
            res = await client.get(url, params=params, timeout=8.0)
            if res.status_code == 200:
                data = res.json()
                daily = data.get("daily", {})
                times = daily.get("time", [])
                if not times or len(times) < 2:
                    return {"error": "No hay datos de pronóstico disponibles."}

                # Índice: 0 = hoy, 1 = mañana
                idx = dias if dias < len(times) else 0
                fecha = times[idx]
                max_temp = daily.get("temperature_2m_max", [None])[idx]
                min_temp = daily.get("temperature_2m_min", [None])[idx]
                prob_lluvia = daily.get("precipitation_probability_mean", [None])[idx]

                return {
                    "ubicacion": ubicacion_nombre,
                    "fecha": fecha,
                    "max": max_temp,
                    "min": min_temp,
                    "prob_lluvia": prob_lluvia
                }
            return {"error": f"Error en Open-Meteo (Código: {res.status_code})"}
    except Exception as e:
        return {"error": f"Error al consultar el clima: {str(e)}"}