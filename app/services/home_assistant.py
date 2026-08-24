import os
import httpx

HA_URL = os.getenv("HOME_ASSISTANT_URL", "http://tu-home-assistant-ip:8123")
HA_TOKEN = os.getenv("HOME_ASSISTANT_TOKEN", "")

def get_ha_headers():
    return {
        "Authorization": f"Bearer {HA_TOKEN}",
        "Content-Type": "application/json",
    }

async def controlar_dispositivo_ha(domain: str, service: str, entity_id: str, extra_data: dict = None) -> str:
    """Ejecuta un servicio en Home Assistant (ej: light/turn_on, switch/turn_off)."""
    if not HA_TOKEN:
        return "⚠️ Token de Home Assistant no configurado en el servidor."
    
    url = f"{HA_URL}/api/services/{domain}/{service}"
    payload = {"entity_id": entity_id}
    if extra_data:
        payload.update(extra_data)

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=get_ha_headers(), json=payload, timeout=10.0)
            if response.status_code in [200, 201]:
                return f"✅ Comando ejecutado con éxito en {entity_id}."
            else:
                return f"❌ Error en Home Assistant: {response.status_code} - {response.text}"
    except Exception as e:
        return f"❌ Error de conexión con Home Assistant: {str(e)}"

async def obtener_estado_sensor(entity_id: str) -> str:
    """Consulta el estado actual de un sensor o dispositivo."""
    if not HA_TOKEN:
        return "⚠️ Token de Home Assistant no configurado."
    
    url = f"{HA_URL}/api/states/{entity_id}"
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=get_ha_headers(), timeout=10.0)
            if response.status_code == 200:
                data = response.json()
                state = data.get("state")
                unit = data.get("attributes", {}).get("unit_of_measurement", "")
                return f"El estado actual de {entity_id} es: {state} {unit}".strip()
            else:
                return f"No se pudo obtener el estado de {entity_id}."
    except Exception as e:
        return f"Error consultando sensor: {str(e)}"