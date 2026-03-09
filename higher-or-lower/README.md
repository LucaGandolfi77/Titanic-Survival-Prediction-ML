**Higher or Lower**

Gioco web semplice "Higher or Lower" (indovina se la prossima carta è più alta o più bassa).

**Caratteristiche**
- **Streaks**: aumenta la combo rispondendo correttamente.
- **Moltiplicatore**: cresce ogni 5 corrette per moltiplicare i punti.
- **Leaderboard**: salva e mostra i migliori punteggi (SQLite).

**Quick Start**

1. Apri una shell nella cartella del gioco:

```bash
cd higher-or-lower
```

2. (Opzionale) crea e attiva un ambiente virtuale:

```bash
python3 -m venv env
source env/bin/activate
```

3. Installa le dipendenze richieste:

```bash
pip install flask flask-cors
```

4. Avvia il server:

```bash
python server.py
```

5. Apri il client nel browser su `http://localhost:5000` oppure apri direttamente il file client con un server statico.

**API principali**
- `POST /api/new_game` — avvia una nuova partita, ritorna la carta corrente e lo stato iniziale.
- `POST /api/guess` — invia `{ "direction": "higher"|"lower" }`, riceve risultato, punteggio e stato del mazzo.
- `POST /api/save_score` — salva il punteggio corrente nel DB (body JSON con `name`).
- `GET  /api/leaderboard` — ritorna la top10 dei punteggi.

**File importanti**
- Client: [higher-or-lower/client/index.html](higher-or-lower/client/index.html)
- Server: [higher-or-lower/server.py](higher-or-lower/server.py)
- DB (SQLite): `scores.db` creato automaticamente nella cartella del progetto.

**Note di sviluppo**
- Il server serve anche i file statici dal sottofile `client`.
- Se vuoi resettare la leaderboard, elimina `scores.db` e riavvia il server.

**Contribuire**
- Apri un issue o invia una PR con miglioramenti (es. UI, test, persistente multi-utente).

---

Divertiti!
