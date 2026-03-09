from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import random
import os

app = Flask(__name__, static_folder='client', static_url_path='')
CORS(app)
app.secret_key = 'briscola_2024'

# ─── Card definitions ────────────────────────────────────────────────────────

SUITS = ['C', 'D', 'B', 'S']   # Cups, Coins, Clubs, Swords
RANKS = ['A', '3', 'K', 'Q', 'J', '7', '6', '5', '4', '2']

CARD_VALUES = {'A': 11, '3': 10, 'K': 4, 'Q': 3, 'J': 2,
               '7': 0,  '6': 0,  '5': 0, '4': 0, '2': 0}

# Higher = stronger in a trick
CARD_STRENGTH = {'A': 10, '3': 9, 'K': 8, 'Q': 7, 'J': 6,
                 '7': 5,  '6': 4, '5': 3, '4': 2, '2': 1}

SUIT_NAMES = {'C': 'Cups', 'D': 'Coins', 'B': 'Clubs', 'S': 'Swords'}
RANK_NAMES = {
    'A': 'Ace', '3': 'Three', 'K': 'King', 'Q': 'Horse', 'J': 'Jack',
    '7': 'Seven', '6': 'Six', '5': 'Five', '4': 'Four', '2': 'Two'
}

game = None   # single global game state (None when no active game)


def card_rank(c):   return c.split('_')[0]
def card_suit(c):   return c.split('_')[1]
def card_value(c):  return CARD_VALUES[card_rank(c)]
def card_strength(c): return CARD_STRENGTH[card_rank(c)]

def card_label(c):
    r, s = card_rank(c), card_suit(c)
    return f"{RANK_NAMES[r]} of {SUIT_NAMES[s]}"


def create_deck():
    deck = [f"{r}_{s}" for s in SUITS for r in RANKS]
    random.shuffle(deck)
    return deck


# ─── Trick logic ─────────────────────────────────────────────────────────────

def trick_winner(lead, follow, briscola_suit):
    """Return 'lead' or 'follow'."""
    ls, fs = card_suit(lead), card_suit(follow)
    lb, fb = ls == briscola_suit, fs == briscola_suit

    if lb and fb:
        return 'lead' if card_strength(lead) >= card_strength(follow) else 'follow'
    if fb:  return 'follow'
    if lb:  return 'lead'
    # Neither briscola
    if fs == ls:
        return 'lead' if card_strength(lead) >= card_strength(follow) else 'follow'
    return 'lead'   # follow played off-suit → lead wins


# ─── AI strategy ─────────────────────────────────────────────────────────────

def ai_play(hand, table_card, briscola_suit):
    """
    If table_card is None: AI leads.
    Otherwise: AI follows table_card.
    """
    bris = briscola_suit

    if table_card is None:
        # Lead with lowest-value non-briscola; otherwise lowest overall
        non_bris = [c for c in hand if card_suit(c) != bris]
        pool = non_bris if non_bris else hand
        return min(pool, key=lambda c: (card_value(c), card_strength(c)))

    # Following: categorise our cards
    tval = card_value(table_card)

    def beats(c):
        return trick_winner(table_card, c, bris) == 'follow'

    winning     = [c for c in hand if beats(c)]
    non_bris_w  = [c for c in winning if card_suit(c) != bris]
    bris_w      = [c for c in winning if card_suit(c) == bris]
    losers      = [c for c in hand if not beats(c)]
    non_bris_l  = [c for c in losers if card_suit(c) != bris]

    if tval >= 10:
        # High value card on table – try hard to win
        if non_bris_w:
            return min(non_bris_w, key=card_strength)
        if bris_w:
            return min(bris_w, key=card_strength)
    elif tval >= 4:
        # Medium value – win cheaply without briscola
        if non_bris_w:
            return min(non_bris_w, key=card_strength)

    # Can't or shouldn't win – throw cheapest non-briscola
    if non_bris_l:
        return min(non_bris_l, key=lambda c: (card_value(c), card_strength(c)))
    # Forced to throw briscola or a winner
    return min(hand, key=lambda c: (card_value(c), card_strength(c)))


# ─── Draw after trick ────────────────────────────────────────────────────────

def draw_cards(winner_key):
    """Winner draws first, then loser. Briscola card is drawn last."""
    loser_key = 'ai' if winner_key == 'player' else 'player'
    for who in [winner_key, loser_key]:
        if game['deck']:
            game['hands'][who].append(game['deck'].pop(0))
        elif game['briscola_card']:
            game['hands'][who].append(game['briscola_card'])
            game['briscola_card'] = None


# ─── Resolve trick ───────────────────────────────────────────────────────────

def resolve_trick():
    pc  = game['table']['player']
    ac  = game['table']['ai']
    lead = game['lead']

    if lead == 'player':
        result = trick_winner(pc, ac, game['briscola_suit'])
        winner = 'player' if result == 'lead' else 'ai'
    else:
        result = trick_winner(ac, pc, game['briscola_suit'])
        winner = 'ai' if result == 'lead' else 'player'

    pts = card_value(pc) + card_value(ac)
    game['scores'][winner]    += pts
    game['captured'][winner] += [pc, ac]
    game['last_trick'] = {'player': pc, 'ai': ac,
                          'winner': winner, 'value': pts}

    game['table']['player'] = None
    game['table']['ai']     = None

    draw_cards(winner)

    # Check game over
    if not game['hands']['player'] and not game['hands']['ai']:
        finalise_game()
        return

    game['lead'] = winner
    game['turn'] = winner

    if winner == 'player':
        game['message'] = f"You won the trick! (+{pts} pts) — Your lead."
    else:
        game['message'] = f"AI won the trick (+{pts} pts) — AI leads."
        do_ai_lead()


def do_ai_lead():
    """AI leads the next trick."""
    c = ai_play(game['hands']['ai'], None, game['briscola_suit'])
    game['hands']['ai'].remove(c)
    game['table']['ai'] = c
    game['lead']  = 'ai'
    game['turn']  = 'player'
    game['message'] += f"  AI played {card_label(c)}. Your turn!"


def finalise_game():
    ps, as_ = game['scores']['player'], game['scores']['ai']
    game['game_over'] = True
    if ps > as_:
        game['winner']  = 'player'
        game['message'] = f"🎉 You win!  {ps} – {as_}"
    elif as_ > ps:
        game['winner']  = 'ai'
        game['message'] = f"🤖 AI wins!  {as_} – {ps}"
    else:
        game['winner']  = 'draw'
        game['message'] = f"🤝 Draw!  {ps} – {as_}"


# ─── State helper ────────────────────────────────────────────────────────────

def client_state():
    return {
        'player_hand':    game['hands']['player'],
        'ai_card_count':  len(game['hands']['ai']),
        'table':          game['table'],
        'scores':         game['scores'],
        'briscola_card':  game['briscola_card'],
        'briscola_suit':  game['briscola_suit'],
        'deck_count':     len(game['deck']) + (1 if game['briscola_card'] else 0),
        'turn':           game['turn'],
        'lead':           game['lead'],
        'last_trick':     game['last_trick'],
        'message':        game['message'],
        'game_over':      game['game_over'],
        'winner':         game['winner'],
        'captured_counts': {
            'player': len(game['captured']['player']),
            'ai':     len(game['captured']['ai']),
        }
    }


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return send_from_directory('client', 'index.html')


@app.route('/api/new_game', methods=['POST'])
def new_game():
    global game
    deck = create_deck()

    player_hand = deck[:3]
    ai_hand     = deck[3:6]
    remaining   = deck[6:]
    briscola    = remaining[-1]

    game = {
        'deck':         remaining[:-1],
        'briscola_card': briscola,
        'briscola_suit': card_suit(briscola),
        'hands':        {'player': player_hand, 'ai': ai_hand},
        'table':        {'player': None, 'ai': None},
        'scores':       {'player': 0, 'ai': 0},
        'captured':     {'player': [], 'ai': []},
        'turn':         'player',
        'lead':         'player',
        'last_trick':   None,
        'message':      "Game started! You lead — play a card.",
        'game_over':    False,
        'winner':       None,
    }
    return jsonify(client_state())


@app.route('/api/game_state')
def get_state():
    if game is None:
        return jsonify({'error': 'No active game'}), 404
    return jsonify(client_state())


@app.route('/api/reset', methods=['POST'])
def reset_game():
    """Reset/clear the in-memory game state (useful during development)."""
    global game
    game = None
    return jsonify({'ok': True})


@app.route('/api/play_card', methods=['POST'])
def play_card():
    if not game:
        return jsonify({'error': 'No active game'}), 404
    if game['game_over']:
        return jsonify({'error': 'Game over'}), 400
    if game['turn'] != 'player':
        return jsonify({'error': 'Not your turn'}), 400

    card = request.json.get('card')
    if card not in game['hands']['player']:
        return jsonify({'error': 'Card not in hand'}), 400

    game['hands']['player'].remove(card)
    game['table']['player'] = card

    if game['lead'] == 'player':
        # Player led → AI follows
        ac = ai_play(game['hands']['ai'], card, game['briscola_suit'])
        game['hands']['ai'].remove(ac)
        game['table']['ai'] = ac
        game['turn'] = 'resolve'
        resolve_trick()
    else:
        # AI led → player followed → resolve
        game['turn'] = 'resolve'
        resolve_trick()

    return jsonify(client_state())


if __name__ == '__main__':
    app.run(debug=True, port=6000)
