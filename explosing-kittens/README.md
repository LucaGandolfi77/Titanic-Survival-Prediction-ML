**Exploding Kittens — Online (demo)**

Versione server-client in Flask di una variante di "Exploding Kittens" (partita multiplayer in-memory).

**Caratteristiche**
- Lobby per creare/joinare stanze (2–4 giocatori).
- Meccaniche principali: bombe, defuse, nope, favor, see-future, shuffle, ecc.
- Turni, log di gioco e gestione eventi (NOPE window, favor, defuse reinsertion).

**Quick Start**

1. Apri una shell nella cartella del gioco:

```bash
cd explosing-kittens
```

2. (Opzionale) crea e attiva un ambiente virtuale:

```bash
python3 -m venv env
source env/bin/activate
```

3. Installa le dipendenze:

```bash
pip install flask flask-cors
```

4. Avvia il server:

```bash
python server.py
```

5. Apri il client nel browser su `http://localhost:5000`.

**Endpoint API (principali)**
- `POST /api/create_room` — crea una stanza; body JSON: `{ "name": "YourName" }` → ritorna `{ room_id, player_id }`.
- `POST /api/join_room` — entra in una stanza; body JSON: `{ "room_id": "XXXXX", "name": "Name" }`.
- `POST /api/start_game` — (host) avvia la partita; body JSON: `{ "room_id": "RID", "player_id": "PID" }`.
- `GET  /api/state/<room_id>?pid=<player_id>` — ottieni stato pubblico per il giocatore.
- `POST /api/play_card` — gioca carta/e; body JSON: `{ "room_id":..., "player_id":..., "cards": [...], "target": "pid" }`.
- `POST /api/nope` — gioca un NOPE in risposta; body JSON: `{ "room_id":..., "player_id":... }`.
- `POST /api/draw_card` — pesca una carta; body JSON: `{ "room_id":..., "player_id":... }`.
- `POST /api/insert_bomb` — reinserisci bomba dopo defuse; body JSON: `{ "room_id":..., "player_id":..., "position": <int> }`.
- `POST /api/give_favor` — rispondi a una richiesta Favor; body JSON: `{ "room_id":..., "player_id":..., "card_id": "..." }`.

**File importanti**
- Client UI: [explosing-kittens/client/index.html](explosing-kittens/client/index.html)
- Server: [explosing-kittens/server.py](explosing-kittens/server.py)

**Note di sviluppo**
- Lo stato delle stanze è in memoria (`rooms` nel server). Riavviando il server le stanze vengono perse.
- Il mazzo e le carte sono rappresentate con ID tipo `CARDTYPE_<n>` gestiti dal server.
- Per debug, usare `curl` o Postman per chiamare gli endpoint e osservare `log` nella risposta di stato.

**Suggerimenti**
- Aggiungere persistence (Redis/Postgres) per partite long-lived.
- Aggiungere autenticazione e WebSocket per aggiornamenti realtime.

---

Divertiti!
