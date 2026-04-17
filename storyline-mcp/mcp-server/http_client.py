# ============================================================
# http_client.py — Client HTTP condiviso per comunicare col bridge
# ============================================================

import httpx
from config import BRIDGE_URL, HTTP_TIMEOUT


async def bridge_post(endpoint: str, json_body: dict) -> dict:
    """Invia una richiesta POST al bridge e restituisce la risposta JSON."""
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        r = await client.post(f"{BRIDGE_URL}{endpoint}", json=json_body)
        r.raise_for_status()
        return r.json()


async def bridge_get(endpoint: str, params: dict | None = None) -> dict:
    """Invia una richiesta GET al bridge e restituisce la risposta JSON."""
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        r = await client.get(f"{BRIDGE_URL}{endpoint}", params=params or {})
        r.raise_for_status()
        return r.json()
