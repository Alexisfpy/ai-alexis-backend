import httpx

async def obtener_clima_open_meteo(ubicacion: str):
    """
    1. Busca las coordenadas de la ubicación (Geocoding).
    2. Consulta el clima actual y la predicción para hoy/mañana en Open-Meteo.
    """
    async with httpx.AsyncClient() as client:
        try:
            # Paso A: Traducir nombre de ciudad/isla a Latitud y Longitud
            geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={ubicacion}&count=1&language=es&format=json"
            geo_response = await client.get(geo_url)
            geo_data = geo_response.json()

            if not geo_data.get("results"):
                return None

            lugar = geo_data["results"][0]
            lat = lugar["latitude"]
            lon = lugar["longitude"]
            nombre_oficial = lugar["name"]

            # Paso B: Petición a la API de Clima
            weather_url = (
                f"https://api.open-meteo.com/v1/forecast?"
                f"latitude={lat}&longitude={lon}"
                f"&daily=temperature_2m_max,temperature_2m_min,precipitation_probability_max"
                f"&timezone=auto"
            )
            weather_response = await client.get(weather_url)
            weather_data = weather_response.json()

            daily = weather_data.get("daily", {})

            # Extraemos los datos de HOY (índice 0) y MAÑANA (índice 1)
            return {
                "ubicacion": nombre_oficial,
                "hoy": {
                    "max": daily["temperature_2m_max"][0],
                    "min": daily["temperature_2m_min"][0],
                    "prob_lluvia": daily["precipitation_probability_max"][0]
                },
                "manana": {
                    "max": daily["temperature_2m_max"][1],
                    "min": daily["temperature_2m_min"][1],
                    "prob_lluvia": daily["precipitation_probability_max"][1]
                }
            }
        except Exception as e:
            print(f"Error al consultar Open-Meteo: {e}")
            return None