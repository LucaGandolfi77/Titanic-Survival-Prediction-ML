# 🧬 Tamagotchi Social

> **Raise your digital pet, watch it evolve, befriend other trainers, and battle for glory!**

A full-stack social Tamagotchi game with a Flask/SQLite backend and a pure HTML/CSS/JS frontend featuring CSS-animated pets.

---

## ✨ Features

| Feature | Details |
|---|---|
| 🥚 Pet Creation | Choose from **Blob, Cat, Dragon, Bunny** species |
| 📊 Live Stats | Hunger, Happiness, Energy, Health — decay over real time |
| 🎮 Actions | Feed, Play, Sleep/Wake, Heal |
| 🌱 Evolution | 5 stages: Egg → Baby → Child → Teen → Adult (level-gated) |
| 🎨 CSS Animations | Mood-driven animations: bouncing, floating, sleeping, sick, shaking |
| 👥 Social | Send/accept friend requests, play together |
| ⚔️ Battles | Challenge friends — power formula uses Level + Health + Wins |
| 🏆 Leaderboard | Global rankings by level & wins |
| 🔐 Auth | Secure session-based login / register |

---

## 🗂 Project Structure

```
tamagotchi_social/
├── app.py               # Flask application + all API routes
├── wsgi.py              # PythonAnywhere WSGI entry point
├── requirements.txt
├── templates/
│   └── index.html       # SPA shell
└── static/
    ├── css/style.css    # Full design system + CSS pet art
    └── js/app.js        # Frontend logic (no frameworks)
```

---

## 🚀 Local Development

```bash
# 1. Create & activate virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run
python app.py
# → http://127.0.0.1:5000
```

The SQLite database (`tamagotchi.db`) is created automatically on first run.

---

## ☁️ Deploy to PythonAnywhere (Free Tier)

### Step 1 – Upload files
Upload the entire `tamagotchi_social/` folder via the **Files** tab  
(or `git clone` your repo in a Bash console).

### Step 2 – Install dependencies
Open a **Bash console** and run:
```bash
pip3.10 install --user Flask Flask-SQLAlchemy Werkzeug
```

### Step 3 – Create Web App
1. Go to **Web** tab → **Add a new web app**
2. Choose **Manual configuration** → **Python 3.10**
3. Set **Source code** directory to `/home/yourusername/tamagotchi_social`
4. Set **Working directory** to `/home/yourusername/tamagotchi_social`

### Step 4 – Configure WSGI
Click the WSGI file link and replace its contents with:
```python
import sys, os
path = '/home/yourusername/tamagotchi_social'
if path not in sys.path:
    sys.path.insert(0, path)
from app import app as application
with application.app_context():
    from app import db
    db.create_all()
```
Replace `yourusername` with your actual PythonAnywhere username.

### Step 5 – Static files
In the **Web** tab → **Static files** section, add:
| URL       | Directory                                           |
|-----------|-----------------------------------------------------|
| `/static/`| `/home/yourusername/tamagotchi_social/static/`      |

### Step 6 – Secret key (important!)
In the **Web** tab → **Environment variables**, add:
```
SECRET_KEY = some-long-random-string-here
```

### Step 7 – Reload & Visit
Click **Reload** → visit `https://yourusername.pythonanywhere.com`

---

## 🗃 Database Schema

| Table | Key Columns |
|---|---|
| `user` | id, username, password_hash |
| `pet` | hunger, happiness, energy, health, level, xp, evolution_stage, is_sleeping, is_sick |
| `friendship` | requester_id, receiver_id, status (pending/accepted/declined) |
| `challenge` | challenger_id, challenged_id, status, winner_id, battle_log |

---

## 🔌 REST API Reference

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/register` | `{username, password}` |
| POST | `/api/login` | `{username, password}` |
| POST | `/api/logout` | — |
| GET  | `/api/me` | Current user info |
| GET  | `/api/pet` | Your pet |
| POST | `/api/pet/create` | `{name, species}` |
| POST | `/api/pet/feed` | — |
| POST | `/api/pet/play` | — |
| POST | `/api/pet/sleep` | Toggle sleep/wake |
| POST | `/api/pet/heal` | Heal sick pet |
| GET  | `/api/social/users` | All trainers with alive pets |
| GET  | `/api/social/friends` | Your friend list |
| GET  | `/api/social/friend/requests` | Incoming requests |
| POST | `/api/social/friend/request` | `{user_id}` |
| POST | `/api/social/friend/respond` | `{friendship_id, action}` |
| POST | `/api/social/interact` | Play with friend `{user_id}` |
| GET  | `/api/social/challenges` | All challenges |
| POST | `/api/social/challenge` | Send challenge `{user_id}` |
| POST | `/api/social/challenge/respond` | `{challenge_id, action}` |
| GET  | `/api/leaderboard` | Top 20 pets |

---

## ⚙️ Pet Mechanics

### Stat Decay (per hour, while awake)
| Stat | Rate |
|---|---|
| Hunger | −8 / hour |
| Happiness | −5 / hour |
| Energy | −3 / hour |

### Evolution Stages
| Stage | Name | Level Required |
|---|---|---|
| 1 | 🥚 Egg | 1 |
| 2 | 🐣 Baby | 5 |
| 3 | 🌱 Child | 10 |
| 4 | ✨ Teen | 20 |
| 5 | 👑 Adult | 35 |

### Battle Formula
```
power = (level × 10 + health × 0.5 + happiness × 0.3 + wins × 2) × rand(0.85..1.15)
```

---

## 🛠 Tech Stack

- **Backend**: Python 3.10, Flask 2.3, Flask-SQLAlchemy, Werkzeug
- **Database**: SQLite (local) — migrate to MySQL for PA free tier if needed
- **Frontend**: Vanilla HTML5 / CSS3 / ES6+ JavaScript — zero dependencies
- **Hosting**: PythonAnywhere (free tier compatible)

---

## 📝 License

MIT — free to use and modify.
