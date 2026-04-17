# ============================================================
# tools/variables.py — Tool per leggere/scrivere variabili Storyline
# ============================================================

from __main__ import mcp
from http_client import bridge_post, bridge_get


@mcp.tool()
async def set_variable(name: str, value: str) -> str:
    """
    Imposta il valore di una variabile in Articulate Storyline 360.

    Usa questa funzione per cambiare lo stato del corso:
    - Mostrare/nascondere oggetti tramite variabili condizionali
    - Aggiornare testi dinamici
    - Modificare punteggi o contatori
    - Triggerare logica condizionale nelle slide

    Args:
        name: Nome della variabile Storyline (es. "score", "userName")
        value: Valore da assegnare (stringa, numero, o booleano come stringa)

    Returns:
        Stato dell'operazione ("ok" o messaggio di errore)
    """
    result = await bridge_post("/set_var", {"name": name, "value": value})
    return result.get("status", "ok")


@mcp.tool()
async def get_variable(name: str) -> str:
    """
    Legge il valore attuale di una variabile Storyline 360.

    Usa questa funzione per capire lo stato corrente del corso:
    - Leggere punteggi o progressi dell'utente
    - Verificare risposte date dall'utente
    - Controllare lo stato di completamento di una slide/scena
    - Leggere input inseriti dall'utente in campi di testo

    Args:
        name: Nome della variabile Storyline da leggere

    Returns:
        Il valore attuale della variabile come stringa
    """
    result = await bridge_get("/get_var", {"name": name})
    return str(result.get("value", ""))
