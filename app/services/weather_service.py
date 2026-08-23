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

async def consultar_clima_open_meteo(ciudad: str) -> str:
    """Consulta temperatura, sensación térmica, viento y estado del cielo en tiempo real."""
    coords = await obtener_coordenadas(ciudad)
    if not coords:
        return f"No pude encontrar las coordenadas geográficas para '{ciudad}'."

    lat, lon, ubicacion_nombre = coords
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": ["temperature_2m", "relative_humidity_2m", "apparent_temperature", "weather_code", "wind_speed_10m"],
        "timezone": "auto"
    }

    try:
        async with httpx.AsyncClient() as client:
            res = await client.get(url, params=params, timeout=8.0)
            if res.status_code == 200:
                current = res.json().get("current", {})
                temp = current.get("temperature_2m")
                sensacion = current.get("apparent_temperature")
                humedad = current.get("relative_humidity_2m")
                viento = current.get("wind_speed_10m")
                wmo_code = current.get("weather_code", 0)
                estado = WMO_CODES.get(wmo_code, "Condición variable")

                return (
                    f"Condiciones actuales en {ubicacion_nombre}:\n"
                    f"- Estado: {estado}\n"
                    f"- Temperatura: {temp}°C (Sensación térmica: {sensacion}°C)\n"
                    f"- Humedad: {humedad}%\n"
                    f"- Viento: {viento} km/h"
                )
            return f"Error en la API de Open-Meteo (Código: {res.status_code})."
    except Exception as e:
        return f"Error al consultar el clima: {str(e)}"