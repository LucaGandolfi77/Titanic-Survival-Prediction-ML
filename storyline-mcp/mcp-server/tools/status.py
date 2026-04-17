# ============================================================
# tools/status.py — Tool per verificare lo stato della connessione
# ============================================================

from __main__ import mcp
from http_client import bridge_get


@mcp.tool()
async def check_connection() -> str:
    """
    Verifica se Articulate Storyline 360 è connesso al bridge.

    Usa questa funzione PRIMA di qualsiasi altra operazione per assicurarti
    che il corso sia aperto nel browser e la connessione WebSocket sia attiva.

    Possibili cause di disconnessione:
    - Il bridge server (Node.js) non è in esecuzione
    - Il corso non è aperto nel browser
    - Il corso è aperto con protocollo file:// invece di http://localhost
    - Firewall blocca la porta 8765

    Returns:
        Messaggio con stato della connessione (✅ connesso o ❌ non connesso)
    """
    try:
        result = await bridge_get("/status")
        connected = result.get("connected", False)
        if connected:
            return "Storyline 360 connesso ✅"
        return (
            "Storyline 360 NON connesso ❌ — "
            "Apri il corso nel browser su http://localhost:9000/story.html"
        )
    except Exception as e:
        return (
            f"Bridge server non raggiungibile ❌ — "
            f"Avvia il bridge con: node bridge/server.js — Errore: {e}"
        )


@mcp.tool()
async def get_bridge_health() -> str:
    """
    Ottieni informazioni dettagliate sullo stato del bridge server.

    Returns:
        JSON con status, stato connessione Storyline e uptime del bridge
    """
    try:
        result = await bridge_get("/health")
        return (
            f"Bridge: {result.get('status', '?')} | "
            f"Storyline: {'connesso' if result.get('storylineConnected') else 'disconnesso'} | "
            f"Uptime: {result.get('uptime', 0):.0f}s"
        )
    except Exception as e:
        return f"Bridge non raggiungibile: {e}"
