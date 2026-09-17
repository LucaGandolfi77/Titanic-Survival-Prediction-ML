# ═══════════════════════════════════════════════════════════════════
#  💅 VELVETVILLE — Fashion Week Arena PWA
# ═══════════════════════════════════════════════════════════════════

> **Where Fashion Meets Competition.**
> Una Progressive Web App interattiva per giuditare runway virtuali, votare outfit e chattare in tempo reale con la community fashion.

![PWA](https://img.shields.io/badge/PWA-Ready-00aced?style=for-the-badge&logo=googlechrome)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)
![Flask](https://img.shields.io/badge/Backend-Flask-ffffff?style=for-the-badge&logo=python&logoColor=yellow)
![SocketIO](https://img.shields.io/badge/Real%20Time-SocketIO-ffffff?style=for-the-badge&logo=socket.io&logoColor=white)
![Three.js](https://img.shields.io/badge/3D-Three.js-ffffff?style=for-the-badge&logo=three.js&logoColor=white)
![SQLAlchemy](https://img.shields.io/badge/DB-SQLAlchemy-ffffff?style=for-the-badge&logo=sqlalchemy&logoColor=e38600)
![Lighthouse](https://img.shields.io/badge/Lighthouse-Goal%20%3E90-brightgreen?style=for-the-badge)

---

## 📖 Descrizione

Velvetville è un'esperienza di fashion show virtuale dove gli utenti possono:

- 🎭 **Guardare** modelli sfilare in una scena 3D interattiva (Three.js)
- 👠 **Giudicare** gli outfit con un sistema di voting a slider (1-10)
- 💬 **Chattare** nella Gossip Room in tempo reale con WebSocket (Flask-SocketIO)
- 🏆 **Visualizzare** una leaderboard con i punteggi più alti
- 📊 **Analizzare** dati di voto: distribution, sentiment, attività oraria
- 🤖 **Ricevere** raccomandazioni outfit personalizzate basate sul tuo profilo di voto
- 🎮 **Giocare** con sistema di gamification: XP, livelli e badge
- 📡 **Notifiche Push** con Web Push API (VAPID)
- 🌍 **Temi ambientali**: la UI cambia in base alla posizione geografica e all'ora
- 🌐 **Multi-lingua**: EN, IT, ES con switching in tempo reale
- 📤 **Esportare** il ranking in JSON/HTML
- 🛡️ **Proteggere** le API con rate limiting
- 📱 **Installare** la PWA e usare l'app offline con Background Sync

---

## 🛠️ Stack Tecnologico

| Layer | Tecnologia |
|-------|-----------|
| Backend | Python 3.11+ · Flask · Flask-CORS · Flask-Limiter |
| Real-time | Flask-SocketIO · WebSocket |
| Database | SQLite/PostgreSQL · SQLAlchemy · Flask-SQLAlchemy |
| Auth | Flask-Login · Session-based |
| Gamification | XP · Levels · Badges · Streaks |
| Frontend | Vanilla JS (ES6+) · HTML5 · CSS3 |
| 3D | Three.js r128 |
| PWA | Service Worker · Cache API · Web Manifest · Background Sync · IndexedDB |
| Protocolli | REST API · JSON · WebSocket · SocketIO |
| Testing | pytest (29 test) |
| Analytics | Chart.js · Sentiment · Recommendations |
| Push | Web Push · VAPID · Subscriptions |
| i18n | EN · IT · ES |
| CI/CD | GitHub Actions |
| Hosting | gunicorn / WSGI · SocketIO server |

---

## ⚙️ Requisiti di Sistema

### Sviluppo Locale
- **Python** 3.9+
- **pip** 21+
- Browser moderno con Service Worker (Chrome 80+, Firefox 78+, Safari 14+)
- Per 3D: GPU con supporto WebGL (opzionale)

### Produzione
- Server WSGI (gunicorn raccomandato)
- HTTPS obbligatorio (Service Worker richiede secure context)
- Nginx/Apache come reverse proxy

---

## 🚀 Installazione e Avvio

### 1. Clona il repository
```bash
git clone https://github.com/<org>/velvetville.git
cd velvetville
```

### 2. Crea un virtual environment
```bash
python -m venv venv
source venv/bin/activate   # Linux/macOS
# venv\Scripts\activate    # Windows
```

### 3. Installa le dipendenze
```bash
pip install -r requirements.txt
```

### 4. Configura le variabili d'ambiente
```bash
cp .env.example .env
# Modifica .env se necessario
```

### 5. Avvia in modalità sviluppo
```bash
python run.py
# Server SocketIO disponibile su http://localhost:5001
```

### 5b. Avvia senza SocketIO (solo API REST)
```bash
python app.py
# API disponibile su http://localhost:5001
```

### 6. Avvia in produzione
```bash
gunicorn -w 4 -b 0.0.0.0:8000 app:app
```

### 7. Avvia la dashboard analytics
```bash
# Dopo aver avviato il server, visita:
# http://localhost:5001/analytics
```

### 8. Esegui i test
```bash
pytest tests/ -v
# 23 test — tutti devono passare

# Con copertura
pytest tests/ --cov=. --cov-report=term-missing
# ~89% copertura totale
```

---

## 🤝 Linee Guida per il Contribuire

### Branching Strategy (Git Flow)
```
main        → Produzione stabile (solo merge da release)
develop     → Integrazione continua
feature/*   → Nuove feature (branch da develop)
bugfix/*    → Fix bug (branch da develop)
hotfix/*    → Fix urgente in produzione (branch da main)
release/*   → Preparazione release (branch da develop)
```

### Commit Convention (Conventional Commits)
```
feat:     Aggiunta nuova funzionalità
fix:      Fix di un bug
docs:     Aggiornamento documentazione
style:    Formattazione, no change di logica
refactor: Ristrutturazione codice
test:     Aggiunta/aggiornamento test
chore:    Manutenzione build, dipendenze
perf:     Ottimizzazione performance
ci:       Cambiamenti CI/CD
```

**Esempio:**
```
feat: add voice judge with Web Speech API
fix: resolve memory leak in cascading setTimeout
docs: update installation instructions for production
```

### Stile del Codice
- **Python**: PEP 8, type hints, docstrings Google-style
- **JavaScript**: ESLint (Airbnb config), 2 spazi indentazione, single quotes
- **CSS**: BEM naming, CSS custom properties per variabili
- **HTML**: Semantic tags, no inline styles/event handlers, `lang` attribute

### Pull Request Process
1. Crea il branch da `develop`: `git checkout -b feature/NOME develop`
2. Commit con messaggi convenzionali
3. Aggiungi test per ogni nuova feature/bugfix
4. Run lint: `flake8 app.py tests/`
5. Apri PR con description dettagliata e reference al task
6. Richiedi almeno 1 review da un maintainer
7. Merge con **Squash Merge** su `develop`

---

## 📂 Struttura del Progetto

```
velvetville/
├── run.py                # Entry point with SocketIO + eventlet
├── app.py                # Flask factory, routes, SocketIO, analytics, AI, i18n
├── extensions.py         # Shared db, login_manager, socketio
├── models.py             # User, Vote, Badge, UserProfile
├── config/
│   ├── __init__.py
│   └── settings.py       # Dev/Prod/Test configs
├── templates/
│   ├── base.html
│   ├── login.html          # Login page + language selector
│   ├── game.html           # Game + XP bar + SocketIO + lang switcher
│   └── analytics.html      # Analytics Dashboard + Chart.js
├── static/
│   ├── css/
│   │   ├── main.css        # Global styles + responsive
│   │   ├── login.css       # Login styles
│   │   ├── game.css        # Game styles + XP bar + badges
│   │   └── analytics.css   # Dashboard styles
│   ├── js/
│   │   ├── app.js          # Bootstrap, SW register
│   │   ├── game-logic.js   # Game, voting, XP, SocketIO client
│   │   ├── chat.js         # Bot chat
│   │   ├── analytics.js    # Dashboard logic, Chart.js, ambient theme
│   │   └── sw-register.js  # Service worker registration
│   ├── sw.js               # Service Worker (caching + BG sync)
│   ├── manifest.json       # PWA + Share Target
│   └── icons/
│       ├── icon-192.svg
│       └── icon-512.svg
├── tests/
│   ├── test_api.py         # 29 tests (API, analytics, AI, i18n, push, ambient)
│   └── test_routes.py      # Route + profile tests
├── .github/workflows/ci.yml
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## 🔮 Roadmap & Future Implementations

> Un piano visionario per trasformare Velvetville da PWA a piattaforma fashion intelligente e immersiva.

---

### 📊 Fase 1: Scalabilità

| # | Task | Stato | Tecnologia |
|---|------|-------|-----------|
| 1.1 | Database persistente con SQLAlchemy + SQLite | ✅ | SQLAlchemy |
| 1.2 | Autenticazione con Flask-Login + session | ✅ | Flask-Login |
| 1.3 | Service Worker con Stale-While-Revalidate | ✅ | Cache API |
| 1.4 | Background Sync per voti offline | ✅ | Background Sync API + IndexedDB |
| 1.5 | PWA Manifest + Icons + Theme Color | ✅ | Web Manifest |
| 1.6 | CI/CD pipeline con GitHub Actions | ✅ | GitHub Actions |
| 1.7 | Fix memory leak (cascading timers) | ✅ | clearTimeout |
| 1.8 | Error handling API + input validation | ✅ | Flask error handlers |
| 1.9 | Lighthouse optimization (score >90) | 🔄 | Lighthouse |
| 1.10 | Migrazione a PostgreSQL in produzione | ⏳ | PostgreSQL |
| 1.11 | WebSocket chat real-time | ✅ | Flask-SocketIO |
| 1.12 | Full test suite (29 tests) | ✅ | pytest |
| 1.13 | API rate limiting | ✅ | Flask-Limiter |
| 1.14 | Gamification (XP, levels, badges) | ✅ | Backend + IndexedDB |
| 1.15 | Responsive design mobile-first | ✅ | CSS Media Queries |
| 1.16 | Analytics Dashboard API | ✅ | REST API |
| 1.17 | Sentiment Analysis | ✅ | Python |
| 1.18 | AI Recommendations | ✅ | Collaborative Filter |
| 1.19 | Push Notifications | ✅ | Web Push + VAPID |
| 1.20 | Multi-lingua (EN, IT, ES) | ✅ | Session-based i18n |
| 1.21 | Ambient themes (geo + weather) | ✅ | Geolocation API |
| 1.22 | Export (JSON + HTML) | ✅ | REST API |
| 1.23 | Notification subscription mgmt | ✅ | Push API |

---

### 🧠 Fase 2: Intelligenza

| # | Task | Stato | Tecnologia | Impatto |
|---|------|-------|-----------|---------|
| 2.1 | Gamification avanzata (achievement system) | ✅ | Backend + IndexedDB | Retention + engagement |
| 2.2 | **Sentiment Analysis** chat in tempo reale | ✅ | Keyword-based Python | Community health |
| 2.3 | **Analytics Dashboard**: voti, trend, sentiment, heatmap | ✅ | Chart.js + REST | Insight fashion |
| 2.4 | **AI Recommendations**: outfit basati su profilo utente | ✅ | Collaborative Filter | UX personalizzata |
| 2.5 | AI Advisor on-device (TensorFlow.js / WebNN) | ⏳ | TensorFlow.js / WebNN | AI offline |
| 2.6 | Image Recognition: identifica stile da foto | ⏳ | MobileNet / TF.js | Auto-categorizzazione |

---

### 🌍 Fase 3: Ecosistema

| # | Task | Stato | Tecnologia | Impatto |
|---|------|-------|-----------|---------|
| 3.1 | **Push Notifications**: Alert quando il tuo modello preferito è in runway | ✅ | Push API + VAPID | Retention push |
| 3.2 | **Biometric Login**: Login con Face ID/Touch ID su mobile | ⏳ | WebAuthn API | Auth sicuro + frictionless |
| 3.3 | **Multi-lingua**: Localizzazione completa (EN, IT, ES) | ✅ | Session-based i18n | Mercato globale |
| 3.4 | **Funzioni Ambientali**: Scena 3D cambia in base a posizione geografica, clima, ora | ✅ | Geolocation API | Mondo vivente e reattivo |
| 3.5 | **Sensori**: Meteo locale influisce sulla runway | ⏳ | Device Sensors API | Immersione contestuale |
| 3.6 | **File System Access**: Esporta ranking personale come PDF/image | ✅ | REST API | Content creation da utenti |
| 3.7 | **Web Share Target**: Condividi outfit da Instagram/TikTok | ✅ | Web Share API + Share Target | Viralità + onboarding |
| 3.8 | **AR Mode**: Prova gli outfit in realtà aumentata con la fotocamera | ⏳ | WebXR / ARKit / ARCore | Experience wow-factor |

---

> 🚀 **Visione finale**: Velvetville non è solo una PWA — è l'inizio di un ecosistema fashion sociale dove l'intelligenza artificiale, il contesto ambientale e la community si fondono per creare la prima piattaforma di fashion show intelligente, accessibile ovunque e su qualsiasi dispositivo.
