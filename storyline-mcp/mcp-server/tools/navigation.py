# ============================================================
# tools/navigation.py — Tool per la navigazione tra slide
# ============================================================

from __main__ import mcp
from http_client import bridge_post


@mcp.tool()
async def jump_to_slide(slide_index: int) -> str:
    """
    Porta il corso Storyline 360 a una slide specifica.

    L'indice è 1-based (la prima slide è 1, la seconda è 2, ecc.).
    Utile per:
    - Navigare il corso in modo programmatico
    - Saltare a sezioni specifiche
    - Ripristinare una slide dopo un errore
    - Creare percorsi di navigazione personalizzati

    Richiede Storyline 360 Build 3.98+ con JS API avanzata.

    Args:
        slide_index: Indice della slide (1-based)

    Returns:
        Stato dell'operazione ("ok" o messaggio di errore)
    """
    result = await bridge_post("/jump_slide", {"slideIndex": slide_index})
    return result.get("status", "ok")


@mcp.tool()
async def next_slide() -> str:
    """
    Avanza alla slide successiva nel corso Storyline 360.

    Returns:
        Stato dell'operazione
    """
    result = await bridge_post("/next_slide", {})
    return result.get("status", "ok")


@mcp.tool()
async def previous_slide() -> str:
    """
    Torna alla slide precedente nel corso Storyline 360.

    Returns:
        Stato dell'operazione
    """
    result = await bridge_post("/prev_slide", {})
    return result.get("status", "ok")
