# ============================================================
# config.py — Configurazione centralizzata del MCP server
# ============================================================

# URL base del bridge Node.js
BRIDGE_URL = "http://localhost:8765"

# Timeout HTTP per le richieste al bridge (secondi)
HTTP_TIMEOUT = 10

# Nome e descrizione del server MCP
MCP_NAME = "Storyline360 MCP"
MCP_DESCRIPTION = (
    "Controlla Articulate Storyline 360 da qualsiasi AI client MCP. "
    "Permette di leggere/scrivere variabili, navigare slide, "
    "animare oggetti e verificare lo stato della connessione."
)
