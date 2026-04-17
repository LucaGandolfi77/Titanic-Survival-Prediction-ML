# ============================================================
# tools/__init__.py — Registrazione di tutti i tool MCP
# ============================================================
# Importare questo modulo registra automaticamente tutti i tool
# sul server FastMCP passato.
# ============================================================


def register_all_tools(mcp):
    """Registra tutti i tool sul server MCP fornito."""
    # Importa i moduli che contengono le funzioni @mcp.tool()
    # L'importazione attiva i decoratori e registra i tool
    from tools import variables   # noqa: F401
    from tools import navigation  # noqa: F401
    from tools import animation   # noqa: F401
    from tools import status      # noqa: F401
    from tools import execute     # noqa: F401
