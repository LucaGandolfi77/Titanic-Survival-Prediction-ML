# ============================================================
# tools/animation.py — Tool per animare oggetti Storyline
# ============================================================

import json
from __main__ import mcp
from http_client import bridge_post


@mcp.tool()
async def animate_object(object_name: str, props: str) -> str:
    """
    Anima un oggetto Storyline 360 usando proprietà GSAP/CSS.

    L'oggetto viene identificato tramite il suo attributo data-acc-text
    (il nome accessibile impostato in Storyline) oppure il suo ID DOM.

    Richiede Storyline 360 Build 3.98+ con JS API avanzata.
    GSAP è incluso internamente da Storyline.

    Args:
        object_name: Nome accessibile dell'oggetto (data-acc-text) o ID DOM
        props: Proprietà di animazione in formato JSON.
               Esempio: '{"x": 100, "opacity": 0.5, "duration": 1}'
               Proprietà supportate: x, y, rotation, scale, opacity,
               duration, delay, ease, ecc. (sintassi GSAP)

    Returns:
        Stato dell'operazione ("animated", "error", ecc.)
    """
    try:
        parsed_props = json.loads(props)
    except json.JSONDecodeError:
        return "Errore: 'props' non è un JSON valido"

    result = await bridge_post(
        "/animate",
        {"objectName": object_name, "props": parsed_props},
    )
    return result.get("status", "ok")
