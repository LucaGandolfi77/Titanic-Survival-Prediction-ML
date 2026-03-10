from flask import Flask, session, render_template, redirect, url_for
import random

app = Flask(__name__)
app.secret_key = 'briscola-secret-2026'

SUITS = ['Coppe', 'Denari', 'Bastoni', 'Spade']
SUIT_SYMBOLS = {'Coppe': '♥', 'Denari': '♦', 'Bastoni': '♣', 'Spade': '♠'}
RANKS = ['A', '2', '3', '4', '5', '6', '7', 'J', 'Q', 'K']
RANK_NAMES = {
    'A': 'Asso', '2': '2', '3': '3', '4': '4', '5': '5',
    '6': '6', '7': '7', 'J': 'Fante', 'Q': 'Cavallo', 'K': 'Re'
}
POINTS   = {'A': 11, '3': 10, 'K': 4, 'Q': 3, 'J': 2, '2': 0, '4': 0, '5': 0, '6': 0, '7': 0}
STRENGTH = {'A': 10, '3': 9,  'K': 8, 'Q': 7, 'J': 6, '7': 5, '6': 4, '5': 3, '4': 2, '2': 1}

def make_deck():
    deck = [{'suit': s, 'rank': r} for s in SUITS for r in RANKS]
    random.shuffle(deck)
    return deck

def card_wins(attacker, defender, briscola, lead_suit):
    a_b = attacker['suit'] == briscola
    d_b = defender['suit'] == briscola
    if a_b and not d_b: return True
    if d_b and not a_b: return False
    if attacker['suit'] == lead_suit and defender['suit'] != lead_suit: return True
    if defender['suit'] == lead_suit and attacker['suit'] != lead_suit: return False
    return STRENGTH[attacker['rank']] > STRENGTH[defender['rank']]

def ai_choose(hand, lead_card, briscola):
    if lead_card is None:
        pool = [c for c in hand if c['suit'] != briscola] or hand
        return min(pool, key=lambda c: (POINTS[c['rank']], STRENGTH[c['rank']]))
    lead_suit = lead_card['suit']
    winners = [c for c in hand if card_wins(c, lead_card, briscola, lead_suit)]
    if winners and POINTS[lead_card['rank']] > 0:
        return min(winners, key=lambda c: STRENGTH[c['rank']])
    losers = [c for c in hand if not card_wins(c, lead_card, briscola, lead_suit)]
    pool = losers if losers else hand
    return min(pool, key=lambda c: (POINTS[c['rank']], STRENGTH[c['rank']]))

def cstr(c):
    return f"{RANK_NAMES[c['rank']]} {SUIT_SYMBOLS[c['suit']]}"

@app.route('/')
def index():
    g = session.get('game')
    # Se tocca all'AI aprire, calcola la carta subito
    if g and not g['player_first'] and g['ai_lead'] is None and not g['game_over']:
        ai_card = ai_choose(g['ai_hand'], None, g['briscola']['suit'])
        g['ai_hand'].remove(ai_card)
        g['ai_lead'] = ai_card
        g['message'] = f"L'AI ha giocato {cstr(ai_card)}. Scegli come rispondere!"
        session['game'] = g
        session.modified = True
    return render_template('index.html', game=g,
                           suit_symbols=SUIT_SYMBOLS, rank_names=RANK_NAMES)

@app.route('/new_game', methods=['POST'])
def new_game():
    deck = make_deck()
    session['game'] = {
        'deck':         deck,
        'player_hand':  [deck.pop() for _ in range(3)],
        'ai_hand':      [deck.pop() for _ in range(3)],
        'briscola':     deck[0],
        'player_score': 0,
        'ai_score':     0,
        'player_first': True,
        'ai_lead':      None,
        'last_trick':   None,
        'message':      'Sei il primo! Gioca una carta.',
        'game_over':    False,
    }
    session.modified = True
    return redirect(url_for('index'))

@app.route('/play/<int:idx>', methods=['POST'])
def play(idx):
    g = session.get('game')
    if not g or g['game_over'] or idx >= len(g['player_hand']):
        return redirect(url_for('index'))

    p_card = g['player_hand'].pop(idx)

    if g['player_first']:
        # Giocatore apre, AI risponde
        ai_card   = ai_choose(g['ai_hand'], p_card, g['briscola']['suit'])
        g['ai_hand'].remove(ai_card)
        lead_suit = p_card['suit']
        p_wins    = card_wins(p_card, ai_card, g['briscola']['suit'], lead_suit)
    else:
        # AI ha già aperto (ai_lead), giocatore risponde
        ai_card   = g['ai_lead']
        g['ai_lead'] = None
        lead_suit = ai_card['suit']
        p_wins    = card_wins(p_card, ai_card, g['briscola']['suit'], lead_suit)

    pts = POINTS[p_card['rank']] + POINTS[ai_card['rank']]
    if p_wins:
        g['player_score'] += pts
        g['player_first']  = True
        g['message']       = f"Hai vinto la mano! +{pts} punti 🎉"
    else:
        g['ai_score']     += pts
        g['player_first']  = False
        g['message']       = f"L'AI ha vinto la mano. +{pts} punti all'AI 🤖"

    g['last_trick'] = {
        'player': p_card, 'ai': ai_card,
        'winner': 'player' if p_wins else 'ai'
    }

    # Pesca carte (chi ha vinto pesca per primo)
    w_hand = g['player_hand'] if p_wins else g['ai_hand']
    l_hand = g['ai_hand']     if p_wins else g['player_hand']
    if g['deck']: w_hand.append(g['deck'].pop(0))
    if g['deck']: l_hand.append(g['deck'].pop(0))

    # Fine partita
    if not g['player_hand'] and not g['ai_hand'] and not g['deck']:
        g['game_over'] = True
        ps, ai = g['player_score'], g['ai_score']
        if ps > ai:   g['message'] = f"🏆 Hai vinto! {ps} - {ai}"
        elif ai > ps: g['message'] = f"😞 L'AI ha vinto! {ai} - {ps}"
        else:         g['message'] = f"🤝 Pareggio! {ps} - {ai}"

    session['game'] = g
    session.modified = True
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
