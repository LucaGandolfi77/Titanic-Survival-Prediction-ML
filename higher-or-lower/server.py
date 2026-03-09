from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import random, sqlite3, os, time

app = Flask(__name__, static_folder='client', static_url_path='')
CORS(app)

DB = 'scores.db'

# ── Card definitions ──────────────────────────────────────────────────────────
SUITS  = ['H', 'D', 'C', 'S']   # Hearts, Diamonds, Clubs, Spades
RANKS  = ['2','3','4','5','6','7','8','9','10','J','Q','K','A']
VALUES = {r: i+2 for i, r in enumerate(RANKS)}   # 2=2 … A=14

# ── DB setup ──────────────────────────────────────────────────────────────────
def init_db():
    con = sqlite3.connect(DB)
    con.execute('''
        CREATE TABLE IF NOT EXISTS scores (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            name      TEXT    NOT NULL,
            streak    INTEGER NOT NULL,
            score     INTEGER NOT NULL,
            ts        INTEGER NOT NULL
        )
    ''')
    con.commit()
    con.close()

init_db()

# ── Game state (in-memory, single player) ─────────────────────────────────────
game = {}

def random_card():
    return {'rank': random.choice(RANKS), 'suit': random.choice(SUITS)}

def card_value(c):
    return VALUES[c['rank']]

def build_deck():
    return [{'rank': r, 'suit': s} for s in SUITS for r in RANKS]

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return send_from_directory('client', 'index.html')

@app.route('/api/new_game', methods=['POST'])
def new_game():
    global game
    deck = build_deck()
    random.shuffle(deck)
    game = {
        'deck':    deck,
        'current': deck.pop(),
        'streak':  0,
        'score':   0,
        'over':    False,
        'last':    None,
        'multiplier': 1,
    }
    return jsonify({
        'current': game['current'],
        'streak':  0,
        'score':   0,
        'over':    False,
        'multiplier': 1,
        'deck_count': len(game['deck']),
    })

@app.route('/api/guess', methods=['POST'])
def guess():
    global game
    if not game or game['over']:
        return jsonify({'error': 'No active game'}), 400

    direction = request.json.get('direction')   # 'higher' | 'lower'
    if direction not in ('higher', 'lower'):
        return jsonify({'error': 'Invalid direction'}), 400

    if not game['deck']:
        game['over'] = True
        return jsonify({
            'current':    game['current'],
            'next':       None,
            'correct':    True,
            'streak':     game['streak'],
            'score':      game['score'],
            'multiplier': game['multiplier'],
            'over':       True,
            'deck_empty': True,
        })

    prev  = game['current']
    nxt   = game['deck'].pop()
    pv, nv = card_value(prev), card_value(nxt)

    # Tie counts as correct (edge case)
    if nv == pv:
        correct = True
    elif direction == 'higher':
        correct = nv > pv
    else:
        correct = nv < pv

    if correct:
        game['streak']     += 1
        # Multiplier increases every 5 streak
        game['multiplier']  = min(1 + (game['streak'] // 5), 5)
        pts                 = 10 * game['multiplier']
        game['score']      += pts
        game['current']     = nxt
        game['last']        = {'correct': True, 'pts': pts}
    else:
        game['over'] = True
        game['last'] = {'correct': False, 'pts': 0}

    return jsonify({
        'prev':       prev,
        'current':    nxt,
        'correct':    correct,
        'streak':     game['streak'],
        'score':      game['score'],
        'multiplier': game['multiplier'],
        'over':       game['over'],
        'pts_gained': 10 * game['multiplier'] if correct else 0,
        'deck_count': len(game['deck']),
    })

@app.route('/api/save_score', methods=['POST'])
def save_score():
    data   = request.json
    name   = (data.get('name') or 'Anonymous')[:20].strip()
    streak = int(game.get('streak', 0))
    score  = int(game.get('score',  0))
    if score == 0 and streak == 0:
        return jsonify({'error': 'Nothing to save'}), 400
    con = sqlite3.connect(DB)
    con.execute('INSERT INTO scores (name,streak,score,ts) VALUES (?,?,?,?)',
                (name, streak, score, int(time.time())))
    con.commit()
    con.close()
    return jsonify({'ok': True})

@app.route('/api/leaderboard')
def leaderboard():
    con  = sqlite3.connect(DB)
    rows = con.execute(
        'SELECT name, streak, score FROM scores ORDER BY score DESC LIMIT 10'
    ).fetchall()
    con.close()
    return jsonify([{'name': r[0], 'streak': r[1], 'score': r[2]} for r in rows])

if __name__ == '__main__':
    app.run(debug=True, port=5000)
