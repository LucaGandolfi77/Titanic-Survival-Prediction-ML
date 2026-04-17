# ============================================================
# tools/execute.py — Tool per eseguire JavaScript in Storyline
# ============================================================

from __main__ import mcp
from http_client import bridge_post


@mcp.tool()
async def execute_javascript(code: str) -> str:
    """
    Esegue codice JavaScript arbitrario nel contesto del corso Storyline 360.

    ⚠️ ATTENZIONE: Questo tool è potente ma pericoloso.
    Usare SOLO in ambiente di sviluppo locale.
    Il codice viene eseguito nel browser dell'utente.

    Il codice ha accesso a:
    - GetPlayer() — oggetto player Storyline
    - document — DOM della pagina
    - window — oggetto globale
    - gsap — libreria animazione (se disponibile)

    Esempi di utilizzo:
    - Leggere proprietà complesse: "GetPlayer().GetVar('score')"
    - Manipolare il DOM: "document.querySelector('.slide-title').textContent"
    - Eseguire logica personalizzata non coperta dagli altri tool

    Args:
        code: Codice JavaScript da eseguire

    Returns:
        Risultato dell'esecuzione o messaggio di errore
    """
    result = await bridge_post("/execute_js", {"code": code})
    return str(result.get("result", result.get("status", "ok")))
