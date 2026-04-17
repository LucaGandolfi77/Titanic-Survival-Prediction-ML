# ============================================================
# storyline_mcp_server.py — Entry point del server MCP
# ============================================================
# Avvia con: python storyline_mcp_server.py
# Oppure configura il client MCP per avviarlo automaticamente.
# ============================================================

from mcp.server.fastmcp import FastMCP
from config import MCP_NAME, MCP_DESCRIPTION

# Crea l'istanza globale del server MCP
mcp = FastMCP(name=MCP_NAME, description=MCP_DESCRIPTION)

# Registra tutti i tool importando i moduli
# (i decoratori @mcp.tool() si agganciano all'istanza globale `mcp`)
import tools.variables   # noqa: F401 — registra set_variable, get_variable
import tools.navigation  # noqa: F401 — registra jump_to_slide, next/prev_slide
import tools.animation   # noqa: F401 — registra animate_object
import tools.status      # noqa: F401 — registra check_connection, get_bridge_health
import tools.execute     # noqa: F401 — registra execute_javascript


if __name__ == "__main__":
    print(f"[MCP] Avvio {MCP_NAME}...")
    print("[MCP] Tool registrati:")
    print("  - set_variable")
    print("  - get_variable")
    print("  - jump_to_slide")
    print("  - next_slide")
    print("  - previous_slide")
    print("  - animate_object")
    print("  - check_connection")
    print("  - get_bridge_health")
    print("  - execute_javascript")
    print("[MCP] Transport: stdio")
    mcp.run(transport="stdio")
