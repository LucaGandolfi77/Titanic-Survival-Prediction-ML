from flask import Flask, jsonify, request, send_from_directory, Response
import os, json
from flask_cors import CORS
import random, time, uuid
import logging

app = Flask(__name__, static_folder='client', static_url_path='')
CORS(app)
# gunicorn expects a WSGI callable named `server` in some setups — alias it
server = app

# Basic logging to console
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# In-memory server log buffer (newest first)
server_logs = []

class InMemoryHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
        except Exception:
            msg = str(record)
        # prepend newest first
        server_logs.insert(0, msg)
        # keep buffer bounded
        if len(server_logs) > 500:
            server_logs[:] = server_logs[:500]

# Attach handler to root logger
handler = InMemoryHandler()
handler.setLevel(logging.INFO)
logging.getLogger().addHandler(handler)

# ── Card definitions ──────────────────────────────────────────────────────────
CARD_DEFS = {
    'BOMB':         {'type': 'bomb',   'emoji': '💣', 'color': '#1a1a2e', 'desc': 'EXPLODE unless you have a Defuse!'},
    'DEFUSE':       {'type': 'defuse', 'emoji': '🔧', 'color': '#80b918', 'desc': 'Counter a Bomb — reinsert it secretly'},
    'SKIP':         {'type': 'action', 'emoji': '⏭️', 'color': '#f4a261', 'desc': 'End your turn without drawing'},
    'ATTACK':       {'type': 'action', 'emoji': '⚔️', 'color': '#e63946', 'desc': 'Skip draw — next player takes 2 turns'},
    'NOPE':         {'type': 'react',  'emoji': '🚫', 'color': '#6d6875', 'desc': 'Cancel ANY action (even another Nope!)'},
    'SEE_FUTURE':   {'type': 'action', 'emoji': '🔮', 'color': '#4361ee', 'desc': 'Peek at the top 3 cards of the deck'},
    'SHUFFLE':      {'type': 'action', 'emoji': '🔀', 'color': '#2d6a4f', 'desc': 'Shuffle the deck randomly'},
    'STEAL':        {'type': 'target', 'emoji': '🎯', 'color': '#f72585', 'desc': 'Steal a random card from a player'},
    'FAVOR':        {'type': 'target', 'emoji': '🙏', 'color': '#7b2d8b', 'desc': 'Force a player to give you a card'},
    'REVERSE':      {'type': 'action', 'emoji': '🔄', 'color': '#457b9d', 'desc': 'Reverse the turn order'},
    'TACOCAT':      {'type': 'cat',    'emoji': '🌮', 'color': '#ffb703', 'desc': 'Cat card — pair/triple for combos'},
    'BEARD_CAT':    {'type': 'cat',    'emoji': '🧔', 'color': '#8338ec', 'desc': 'Cat card — pair/triple for combos'},
    'RAINBOW_CAT':  {'type': 'cat',    'emoji': '🌈', 'color': '#06d6a0', 'desc': 'Cat card — pair/triple for combos'},
    'CATTERMELON':  {'type': 'cat',    'emoji': '🍉', 'color': '#ef233c', 'desc': 'Cat card — pair/triple for combos'},
    'HAIRY_POTATO': {'type': 'cat',    'emoji': '🥔', 'color': '#a7c957', 'desc': 'Cat card — pair/triple for combos'},
}

BASE_DECK_COUNTS = {
    'SKIP': 4, 'ATTACK': 4, 'NOPE': 5, 'SEE_FUTURE': 4,
    'SHUFFLE': 4, 'STEAL': 4, 'FAVOR': 3, 'REVERSE': 3,
    'DEFUSE': 6,
    'TACOCAT': 4, 'BEARD_CAT': 4, 'RAINBOW_CAT': 4,
    'CATTERMELON': 4, 'HAIRY_POTATO': 4,
}

CAT_TYPES = {'TACOCAT', 'BEARD_CAT', 'RAINBOW_CAT', 'CATTERMELON', 'HAIRY_POTATO'}

rooms = {}
_uid  = 0

def uid():
    global _uid
    _uid += 1
    return _uid

def ctype(card_id): return card_id.rsplit('_', 1)[0]

def build_and_deal(n_players):
    deck = []
    for ct, count in BASE_DECK_COUNTS.items():
        for _ in range(count):
            deck.append(f"{ct}_{uid()}")
    random.shuffle(deck)

    hands = [[] for _ in range(n_players)]
    # 1 DEFUSE per player (extra, not from deck)
    for i in range(n_players):
        hands[i].append(f"DEFUSE_{uid()}")

    # 6 non-bomb cards per player
    remaining = []
    for card in deck:
        dealt = False
        for i in range(n_players):
            if len(hands[i]) < 7 and ctype(card) not in ('BOMB', 'DEFUSE'):
                hands[i].append(card)
                dealt = True
                break
        if not dealt:
            remaining.append(card)

    # Add n_players-1 bombs into remaining deck
    for _ in range(n_players - 1):
        remaining.append(f"BOMB_{uid()}")
    random.shuffle(remaining)
    return hands, remaining

def alive(room):
    return [p for p in room['order'] if room['players'][p]['alive']]

def cur(room):
    al = alive(room)
    return al[room['turn_idx'] % len(al)] if al else None

def advance_turn(room):
    al = alive(room)
    if not al: return
    cp = cur(room)
    room['players'][cp]['turns'] -= 1
    if room['players'][cp]['turns'] > 0:
        return
    room['players'][cp]['turns'] = 1
    room['turn_idx'] = (room['turn_idx'] + room['dir']) % len(al)

def add_log(room, msg):
    room['log'].insert(0, {'msg': msg, 'ts': time.time()})
    room['log'] = room['log'][:60]
    # Mirror room log to server console for visibility
    logging.info(f"[ROOM {room['id']}] {msg}")

def check_win(room):
    al = alive(room)
    if len(al) == 1:
        room['state'] = 'game_over'
        room['winner'] = al[0]
        add_log(room, f"🏆 {room['players'][al[0]]['name']} wins the game!")
        return True
    return False

def check_deadlines(room):
    if room['state'] == 'nope_window' and time.time() > room.get('nope_dl', 0):
        resolve_action(room)
    elif room['state'] == 'favor_pending' and time.time() > room.get('pending', {}).get('dl', 9e9):
        _auto_favor(room)

def resolve_action(room):
    p     = room.get('pending') or {}
    action = p.get('action')
    by     = p.get('by')
    nopes  = room.get('nope_count', 0)
    room['state']      = 'playing'
    room['pending']    = None
    room['nope_count'] = 0

    if nopes % 2 == 1:
        add_log(room, "🚫 Action was NOPE'd! Nothing happened.")
        return

    name = room['players'][by]['name']
    if action == 'SKIP':
        room['players'][by]['turns'] = 1
        advance_turn(room)
        add_log(room, f"⏭️ {name} skipped their turn.")

    elif action == 'ATTACK':
        al  = alive(room)
        room['players'][by]['turns'] = 1
        advance_turn(room)
        nxt = cur(room)
        room['players'][nxt]['turns'] += 2
        add_log(room, f"⚔️ {name} attacked! {room['players'][nxt]['name']} must take 2 turns.")

    elif action == 'SEE_FUTURE':
        top3 = list(reversed(room['deck'][-3:]))
        room['see_future'] = {'player': by, 'cards': top3}
        add_log(room, f"🔮 {name} peeked at the top 3 cards of the deck.")

    elif action == 'SHUFFLE':
        random.shuffle(room['deck'])
        room['see_future'] = None
        add_log(room, f"🔀 {name} shuffled the deck!")

    elif action == 'REVERSE':
        room['dir'] *= -1
        add_log(room, f"🔄 {name} reversed the turn order!")

    elif action in ('STEAL', 'CAT_PAIR'):
        tgt = p.get('target')
        if tgt and room['players'][tgt]['hand']:
            stolen = random.choice(room['players'][tgt]['hand'])
            room['players'][tgt]['hand'].remove(stolen)
            room['players'][by]['hand'].append(stolen)
            add_log(room, f"🎯 {name} stole a card from {room['players'][tgt]['name']}!")
        else:
            add_log(room, f"🎯 {room['players'].get(tgt,{}).get('name','?')} had no cards to steal!")

    elif action == 'FAVOR':
        tgt = p.get('target')
        room['state']   = 'favor_pending'
        room['pending'] = {'action': 'FAVOR', 'by': by, 'target': tgt, 'dl': time.time() + 20}
        add_log(room, f"🙏 {name} asks a favor from {room['players'][tgt]['name']}! (20s)")
        return

def _auto_favor(room):
    p   = room['pending']
    tgt = p['target']
    by  = p['by']
    if room['players'][tgt]['hand']:
        card = random.choice(room['players'][tgt]['hand'])
        room['players'][tgt]['hand'].remove(card)
        room['players'][by]['hand'].append(card)
        add_log(room, f"🙏 (time up!) {room['players'][tgt]['name']} gave a random card.")
    room['state']   = 'playing'
    room['pending'] = None

def public_state(room, pid):
    players = {}
    for p, d in room['players'].items():
        players[p] = {
            'name':   d['name'],
            'alive':  d['alive'],
            'turns':  d['turns'],
            'hand':   d['hand'] if p == pid else ['HIDDEN'] * len(d['hand']),
            'count':  len(d['hand']),
        }

    sf = None
    if room.get('see_future') and room['see_future']['player'] == pid:
        sf = room['see_future']['cards']

    favor_me = None
    if room['state'] == 'favor_pending' and room.get('pending', {}).get('target') == pid:
        favor_me = room['pending']

    defuse_me = None
    if room['state'] == 'defuse_pending' and room.get('pending', {}).get('player') == pid:
        defuse_me = {'deck_size': len(room['deck'])}

    pending_pub = None
    if room['state'] == 'nope_window':
        pending_pub = {k: v for k, v in (room.get('pending') or {}).items() if k != 'cards'}

    return {
        'room_id':      room['id'],
        'host':         room['host'],
        'state':        room['state'],
        'players':      players,
        'order':        room['order'],
        'current':      cur(room),
        'dir':          room['dir'],
        'deck_count':   len(room['deck']),
        'discard_top':  room['discard'][-1] if room['discard'] else None,
        'log':          room['log'][:20],
        'winner':       room['winner'],
        'pending':      pending_pub,
        'nope_dl':      room.get('nope_dl'),
        'nope_count':   room.get('nope_count', 0),
        'see_future':   sf,
        'favor_me':     favor_me,
        'defuse_me':    defuse_me,
        'can_nope':     (room['state'] == 'nope_window' and
                         pid != (room.get('pending') or {}).get('by') and
                         any(ctype(c) == 'NOPE' for c in room['players'].get(pid, {}).get('hand', []))),
        'my_id':        pid,
    }


@app.route('/api/server_logs')
def get_server_logs():
    n = int(request.args.get('n', 100))
    # return newest first
    return jsonify({'logs': server_logs[:n]})

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return send_from_directory('client', 'index.html')

@app.route('/api/create_room', methods=['POST'])
def create_room():
    name = (request.json.get('name') or 'Player')[:20].strip()
    logging.info(f"create_room request: name={name}")
    pid  = str(uuid.uuid4())[:8]
    rid  = str(uuid.uuid4())[:5].upper()
    rooms[rid] = {
        'id': rid, 'host': pid, 'state': 'lobby',
        'players': {pid: {'name': name, 'hand': [], 'alive': True, 'turns': 1}},
        'order': [pid], 'deck': [], 'discard': [],
        'turn_idx': 0, 'dir': 1, 'log': [], 'winner': None,
        'pending': None, 'nope_count': 0, 'nope_dl': 0, 'see_future': None,
    }
    add_log(rooms[rid], f"🏠 Room {rid} created by {name}.")
    return jsonify({'room_id': rid, 'player_id': pid})

@app.route('/api/join_room', methods=['POST'])
def join_room():
    data = request.json
    rid  = (data.get('room_id') or '').upper().strip()
    name = (data.get('name') or 'Player')[:20].strip()
    logging.info(f"join_room request: room={rid} name={name}")
    if rid not in rooms:
        logging.warning(f"join_room: room not found {rid}")
        return jsonify({'error': 'Room not found'}), 404
    room = rooms[rid]
    if room['state'] != 'lobby':
        return jsonify({'error': 'Game already in progress'}), 400
    if len(room['players']) >= 4:
        return jsonify({'error': 'Room is full (max 4)'}), 400
    pid = str(uuid.uuid4())[:8]
    room['players'][pid] = {'name': name, 'hand': [], 'alive': True, 'turns': 1}
    room['order'].append(pid)
    add_log(room, f"👋 {name} joined!")
    return jsonify({'room_id': rid, 'player_id': pid})

@app.route('/api/start_game', methods=['POST'])
def start_game():
    data = request.json
    rid  = data.get('room_id')
    pid  = data.get('player_id')
    logging.info(f"start_game request: room={rid} by={pid}")
    if rid not in rooms: return jsonify({'error': 'Room not found'}), 404
    room = rooms[rid]
    if room['host'] != pid: return jsonify({'error': 'Only host can start'}), 403
    if len(room['players']) < 2: return jsonify({'error': 'Need at least 2 players'}), 400

    n     = len(room['players'])
    pids  = list(room['players'].keys())
    random.shuffle(pids)
    room['order'] = pids

    hands, deck = build_and_deal(n)
    for i, p in enumerate(pids):
        room['players'][p]['hand']  = hands[i]
        room['players'][p]['alive'] = True
        room['players'][p]['turns'] = 1

    room['deck']  = deck
    room['state'] = 'playing'
    add_log(room, f"🚀 Game started! {n} players. Good luck!")
    return jsonify(public_state(room, pid))

@app.route('/api/state/<rid>')
def get_state(rid):
    if rid not in rooms: return jsonify({'error': 'Not found'}), 404
    room = rooms[rid]
    pid  = request.args.get('pid')
    logging.info(f"get_state: room={rid} pid={pid}")
    check_deadlines(room)
    return jsonify(public_state(room, pid))

@app.route('/api/play_card', methods=['POST'])
def play_card():
    data   = request.json
    rid    = data.get('room_id')
    pid    = data.get('player_id')
    cards  = data.get('cards', [])
    target = data.get('target')
    logging.info(f"play_card: room={rid} pid={pid} cards={cards} target={target}")

    if rid not in rooms: return jsonify({'error': 'Room not found'}), 404
    room = rooms[rid]
    check_deadlines(room)

    if room['state'] != 'playing':
        return jsonify({'error': f'Cannot play in state: {room["state"]}'}), 400
    if cur(room) != pid:
        return jsonify({'error': 'Not your turn'}), 400

    hand   = room['players'][pid]['hand']
    ctypes = [ctype(c) for c in cards]

    for c in cards:
        if c not in hand:
            logging.warning(f"play_card: {c} not in hand for pid={pid}")
            return jsonify({'error': f'{c} not in hand'}), 400
    if 'DEFUSE' in ctypes:
        logging.warning(f"play_card: attempted to play DEFUSE manually by {pid}")
        return jsonify({'error': 'Defuse is used automatically against Bombs'}), 400
    if 'NOPE' in ctypes:
        logging.warning(f"play_card: attempted to play NOPE as action by {pid}")
        return jsonify({'error': 'NOPE is played reactively, not as your action'}), 400

    def restore():
        for c in cards:
            if c in room['discard']:
                room['discard'].remove(c)
            if c not in hand:
                hand.append(c)

    for c in cards: hand.remove(c)
    room['discard'].extend(cards)

    action = None
    if len(cards) == 1:
        ct = ctypes[0]
        if ct in ('SKIP','ATTACK','SEE_FUTURE','SHUFFLE','REVERSE'):
            action = ct
        elif ct in ('STEAL','FAVOR'):
            if not target or target not in room['players'] \
               or not room['players'][target]['alive'] or target == pid:
                restore()
                return jsonify({'error': 'Choose a valid target player'}), 400
            action = ct
        elif ct in CAT_TYPES:
            add_log(room, f"😺 {room['players'][pid]['name']} played a lone cat card... (needs a pair!)")
            return jsonify(public_state(room, pid))
        else:
            restore(); return jsonify({'error': 'Unknown card'}), 400

    elif len(cards) == 2:
        if len(set(ctypes)) == 1 and ctypes[0] in CAT_TYPES:
            if not target or target not in room['players'] \
               or not room['players'][target]['alive'] or target == pid:
                restore(); return jsonify({'error': 'Choose a valid target'}), 400
            action = 'CAT_PAIR'
        else:
            restore(); return jsonify({'error': 'Play a matching pair of cat cards'}), 400

    elif len(cards) >= 3:
        if len(set(ctypes)) == 1 and ctypes[0] in CAT_TYPES:
            if not target or target == pid:
                restore(); return jsonify({'error': 'Choose a valid target'}), 400
            action = 'CAT_PAIR'
        else:
            restore(); return jsonify({'error': 'Invalid combination'}), 400

    if action:
        room['state']      = 'nope_window'
        room['pending']    = {'action': action, 'by': pid, 'target': target, 'cards': cards}
        room['nope_dl']    = time.time() + 4.5
        room['nope_count'] = 0
        tgt_name = f" → {room['players'][target]['name']}" if target else ''
        add_log(room, f"🃏 {room['players'][pid]['name']} plays {action}{tgt_name}! (4s to NOPE)")
        logging.info(f"action pending: room={rid} by={pid} action={action} target={target}")

    return jsonify(public_state(room, pid))

@app.route('/api/nope', methods=['POST'])
def play_nope():
    data = request.json
    rid  = data.get('room_id')
    pid  = data.get('player_id')
    logging.info(f"play_nope: room={rid} pid={pid}")
    if rid not in rooms: return jsonify({'error': 'Room not found'}), 404
    room = rooms[rid]
    if room['state'] != 'nope_window':
        logging.warning(f"play_nope: nothing to nope in room={rid}")
        return jsonify({'error': 'Nothing to NOPE right now'}), 400
    hand = room['players'][pid]['hand']
    nope = next((c for c in hand if ctype(c) == 'NOPE'), None)
    if not nope: return jsonify({'error': 'No NOPE card in hand'}), 400

    hand.remove(nope)
    room['discard'].append(nope)
    room['nope_count'] = room.get('nope_count', 0) + 1
    room['nope_dl']    = time.time() + 3
    n = room['nope_count']
    if n % 2 == 1:
        add_log(room, f"🚫 {room['players'][pid]['name']} NOPEd the action!")
    else:
        add_log(room, f"🚫🚫 {room['players'][pid]['name']} NOPEd the NOPE! Action goes through!")
    return jsonify(public_state(room, pid))

@app.route('/api/draw_card', methods=['POST'])
def draw_card():
    data = request.json
    rid  = data.get('room_id')
    pid  = data.get('player_id')
    logging.info(f"draw_card: room={rid} pid={pid}")
    if rid not in rooms: return jsonify({'error': 'Room not found'}), 404
    room = rooms[rid]
    check_deadlines(room)

    if room['state'] != 'playing':
        return jsonify({'error': f'Cannot draw in state: {room["state"]}'}), 400
    if cur(room) != pid:
        return jsonify({'error': 'Not your turn'}), 400
    if not room['deck']:
        return jsonify({'error': 'Deck is empty'}), 400

    card = room['deck'].pop()
    name = room['players'][pid]['name']

    if ctype(card) == 'BOMB':
        defuse = next((c for c in room['players'][pid]['hand'] if ctype(c) == 'DEFUSE'), None)
        if defuse:
            room['players'][pid]['hand'].remove(defuse)
            room['discard'].append(defuse)
            room['state']   = 'defuse_pending'
            room['pending'] = {'player': pid, 'bomb': card, 'dl': time.time() + 15}
            add_log(room, f"💣 {name} drew a BOMB! 🔧 Defuse activated — choose where to reinsert it! (15s)")
            logging.info(f"draw_card: bomb drawn by {pid}, defuse present")
        else:
            room['discard'].append(card)
            room['players'][pid]['alive'] = False
            add_log(room, f"💥 {name} drew a BOMB and had no Defuse! 💀 ELIMINATED!")
            logging.info(f"draw_card: bomb drawn by {pid}, eliminated")
            if not check_win(room):
                advance_turn(room)
    else:
        room['players'][pid]['hand'].append(card)
        add_log(room, f"🃏 {name} drew a card.")
        if room.get('see_future') and room['see_future']['player'] == pid:
            room['see_future'] = None
        advance_turn(room)

    return jsonify(public_state(room, pid))

@app.route('/api/insert_bomb', methods=['POST'])
def insert_bomb():
    data = request.json
    rid  = data.get('room_id')
    pid  = data.get('player_id')
    pos  = int(data.get('position', 0))
    logging.info(f"insert_bomb: room={rid} pid={pid} pos={pos}")
    if rid not in rooms: return jsonify({'error': 'Room not found'}), 404
    room = rooms[rid]
    if room['state'] != 'defuse_pending': return jsonify({'error': 'No bomb pending'}), 400
    if room['pending']['player'] != pid:  return jsonify({'error': 'Not your bomb'}), 400

    bomb = room['pending']['bomb']
    pos  = max(0, min(pos, len(room['deck'])))
    room['deck'].insert(pos, bomb)
    add_log(room, f"🔧 {room['players'][pid]['name']} reinserted the Bomb at position {pos}/{len(room['deck'])}.")
    room['state']   = 'playing'
    room['pending'] = None
    advance_turn(room)
    return jsonify(public_state(room, pid))

@app.route('/api/give_favor', methods=['POST'])
def give_favor():
    data    = request.json
    rid     = data.get('room_id')
    pid     = data.get('player_id')
    card_id = data.get('card_id')
    logging.info(f"give_favor: room={rid} pid={pid} card={card_id}")
    if rid not in rooms: return jsonify({'error': 'Room not found'}), 404
    room = rooms[rid]
    if room['state'] != 'favor_pending': return jsonify({'error': 'No favor pending'}), 400
    if room['pending']['target'] != pid:  return jsonify({'error': 'Not your favor'}), 400
    if card_id not in room['players'][pid]['hand']:
        return jsonify({'error': 'Card not in hand'}), 400

    by = room['pending']['by']
    room['players'][pid]['hand'].remove(card_id)
    room['players'][by]['hand'].append(card_id)
    add_log(room, f"🙏 {room['players'][pid]['name']} gave a card to {room['players'][by]['name']}.")
    room['state']   = 'playing'
    room['pending'] = None
    return jsonify(public_state(room, pid))

if __name__ == '__main__':
    app.run(debug=True, port=5000)


@app.route('/config.js')
def config_js():
    """Return a small JS snippet setting window.EK_API_BASE from env var.

    This allows the frontend to pick up a server-configured API base URL
    without templating the HTML file.
    """
    base = os.environ.get('EK_API_BASE', '')
    # JSON-encode to safely quote the string
    js = f"window.EK_API_BASE = {json.dumps(base)};"
    return Response(js, mimetype='application/javascript')
