# 🃏 Briscola P2P

Implementazione multiplayer real-time del classico gioco di carte italiano **Briscola**, giocabile direttamente nel browser tramite connessione peer-to-peer WebRTC.

## 🎮 Regole del gioco

- **Mazzo**: 40 carte italiane (4 semi × 10 valori)
- **Semi**: Coppe ♥, Denari ♦, Bastoni ♣, Spade ♠
- **Valori** (dal più forte): Asso (11 pt), Tre (10 pt), Re (4 pt), Cavallo (3 pt), Fante (2 pt), 7-2 (0 pt)
- **Obiettivo**: accumulare più di 60 punti sui 120 totali

### Svolgimento

1. Si distribuiscono **3 carte** a ciascun giocatore
2. Si scopre la carta successiva che determina la **Briscola** (seme di trionfo)
3. Il non-mazziere gioca per primo
4. Ogni turno si gioca una carta ciascuno; vince la presa chi ha:
   - La Briscola più alta, oppure
   - La carta più alta dello stesso seme della prima giocata
5. Il vincitore della presa pesca per primo, poi l'avversario
6. Si prosegue per **20 prese** fino ad esaurire tutte le carte

## 🏗️ Architettura

```
 Giocatore A ←── WebRTC DataChannel ──→ Giocatore B
                    ↕ signaling ↕
                WebSocket Server
```

- **Host** è autoritativo: calcola lo stato di gioco, invia `sync_state`
- **Guest** invia richieste `play_card`, l'host valida e risponde
- Il server di segnalazione **non vede mai** i dati di gioco — serve solo per lo scambio SDP/ICE

## 🚀 Quick Start

### 1. Avvia il server di segnalazione

```bash
cd server
npm install
npm start       # → porta 8080
```

### 2. Apri il gioco

Servi i file statici con un qualsiasi HTTP server:

```bash
# Con Python
python3 -m http.server 3000

# Con npx
npx serve -p 3000
```

Apri `http://localhost:3000` nel browser.

### 3. Gioca

1. **Giocatore A**: clicca "Crea Stanza" → riceve un codice a 6 caratteri
2. **Giocatore B**: inserisce il codice → clicca "Entra"
3. La partita inizia automaticamente!

## 📁 Struttura del progetto

```
briscola-p2p/
├── index.html                Landing page (lobby)
├── game.html                 Tavolo da gioco
├── package.json              Jest config
├── css/
│   ├── main.css              Stili landing page
│   ├── game.css              Stili tavolo da gioco
│   └── cards.css             Stili carte e animazioni
├── js/
│   ├── utils.js              Utility condivise
│   ├── briscola.js           Logica di gioco pura (no DOM)
│   ├── signaling.js          Client WebSocket per segnalazione
│   ├── webrtc.js             Gestione connessione WebRTC
│   ├── ui.js                 Manipolazione DOM e animazioni
│   ├── app.js                Controller landing page
│   └── game.js               Controller principale di gioco
├── server/
│   ├── signaling-server.js   Server di segnalazione WebSocket
│   ├── package.json          Dipendenze server
│   └── Dockerfile            Deploy containerizzato
└── tests/
    └── briscola.test.js      Test unitari della logica di gioco
```

## 🛠️ Tech Stack

| Componente | Tecnologia |
|---|---|
| Frontend | Vanilla JS (ES2022+, ES Modules) |
| Networking | WebRTC DataChannel (P2P) |
| Segnalazione | WebSocket (`ws` library) |
| Styling | CSS3 Custom Properties, Animazioni, Glass Morphism |
| Font | Playfair Display + Inter (Google Fonts) |
| Test | Jest 29 con supporto ESM |
| QR Code | qrcode.js (CDN) |

## 🧪 Test

```bash
npm install
npm test
```

I test coprono:
- Creazione del mazzo (40 carte, 120 punti)
- Shuffle (Fisher-Yates, non-distruttivo)
- Confronto carte (stesso seme, briscola, semi diversi)
- Flusso completo di gioco (20 prese, punteggio totale = 120)
- Sanitizzazione dello stato per la rete
- Validazione delle mosse

## 🐳 Deploy con Docker

```bash
cd server
docker build -t briscola-signaling .
docker run -p 8080:8080 briscola-signaling
```

## 📝 Protocollo messaggi DataChannel

| Tipo | Direzione | Descrizione |
|---|---|---|
| `game_start` | Host → Guest | Stato iniziale della partita |
| `sync_state` | Host → Guest | Aggiornamento stato dopo ogni azione |
| `play_card` | Guest → Host | Richiesta di giocare una carta |
| `trick_result` | Host → Guest | Risultato della presa |
| `chat` | Bidirezionale | Messaggio di chat |
| `rematch_request` | Bidirezionale | Richiesta di rivincita |
| `rematch_accept` | Bidirezionale | Accettazione rivincita |
| `name_exchange` | Bidirezionale | Scambio nomi giocatori |

## 📄 Licenza

MIT
