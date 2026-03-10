from flask import Flask, render_template, request, redirect, url_for
from flask_socketio import SocketIO, emit, join_room
from datetime import date
import random
import string
import copy

app = Flask(__name__)
app.config["SECRET_KEY"] = "chaos-deckbuilder-secret"
socketio = SocketIO(app, cors_allowed_origins="*")

MAX_HP = 25
STARTING_COINS = 4
HAND_SIZE = 5
ROOMS = {}

CARD_POOL = [
    {"id": "strike", "name": "Strike", "cost": 1, "color": "red", "power": 2, "shield": 0, "heal": 0, "draw": 0, "gold": 0, "poison": 0, "text": "+2 power"},
    {"id": "guard", "name": "Guard", "cost": 1, "color": "blue", "power": 0, "shield": 2, "heal": 0, "draw": 0, "gold": 0, "poison": 0, "text": "+2 shield"},
    {"id": "spark", "name": "Spark", "cost": 2, "color": "yellow", "power": 1, "shield": 0, "heal": 0, "draw": 1, "gold": 0, "poison": 0, "text": "+1 power, +1 draw next round"},
    {"id": "medic", "name": "Medic", "cost": 2, "color": "green", "power": 0, "shield": 0, "heal": 2, "draw": 0, "gold": 0, "poison": 0, "text": "Heal 2"},
    {"id": "venom", "name": "Venom Pin", "cost": 2, "color": "purple", "power": 0, "shield": 0, "heal": 0, "draw": 0, "gold": 0, "poison": 2, "text": "Deal 2 poison"},
    {"id": "crush", "name": "Crush", "cost": 3, "color": "red", "power": 4, "shield": 0, "heal": 0, "draw": 0, "gold": 0, "poison": 0, "text": "+4 power"},
    {"id": "wall", "name": "Wall", "cost": 3, "color": "blue", "power": 0, "shield": 5, "heal": 0, "draw": 0, "gold": 0, "poison": 0, "text": "+5 shield"},
    {"id": "tide", "name": "Tidal Jab", "cost": 3, "color": "blue", "power": 2, "shield": 2, "heal": 0, "draw": 0, "gold": 0, "poison": 0, "text": "+2 power, +2 shield"},
    {"id": "greed", "name": "Greed Coin", "cost": 2, "color": "gold", "power": 0, "shield": 0, "heal": 0, "draw": 0, "gold": 2, "poison": 0, "text": "+2 coins next round"},
    {"id": "wild", "name": "Wild Echo", "cost": 4, "color": "yellow", "power": 3, "shield": 0, "heal": 0, "draw": 1, "gold": 1, "poison": 0, "text": "+3 power, +1 draw, +1 coin"},
    {"id": "feast", "name": "Royal Feast", "cost": 4, "color": "green", "power": 0, "shield": 2, "heal": 4, "draw": 0, "gold": 0, "poison": 0, "text": "Heal 4, +2 shield"},
    {"id": "chaos", "name": "Chaos Crown", "cost": 0, "color": "legendary", "power": 5, "shield": 5, "heal": 0, "draw": 1, "gold": 1, "poison": 1, "text": "Mission reward. Everything at once."},
]

MISSIONS = [
    {"id": "blue_buyer", "name": "Underwater Collector", "desc": "Buy 3 blue cards.", "goal": 3},
    {"id": "big_hit", "name": "Dramatic Entrance", "desc": "Deal 6 or more damage in one round.", "goal": 1},
    {"id": "healer", "name": "Suspiciously Kind", "desc": "Heal a total of 6 HP.", "goal": 6},
    {"id": "shield", "name": "Turtle Mode", "desc": "Reach 6 shield in one round.", "goal": 1},
]

ACHIEVEMENTS = {
    "dignity": {
        "name": "Lost 7 Times With Dignity",
        "desc": "Lose 7 games in a row and somehow remain classy."
    },
    "fashion": {
        "name": "Fashion Victim",
        "desc": "Change cosmetics 10 times in one game."
    },
    "exactly_one": {
        "name": "One HP Main Character",
        "desc": "Win the game with exactly 1 HP left."
    },
    "underwater_addict": {
        "name": "Certified Mermaid",
        "desc": "Buy 4 blue cards on the underwater daily."
    },
}

DAILY_MODIFIERS = [
    {
        "id": "underwater",
        "name": "Everything Is Underwater",
        "desc": "Blue cards cost 1 less. All attack cards lose 1 power."
    },
    {
        "id": "heatwave",
        "name": "Ridiculous Heatwave",
        "desc": "Red cards gain +1 power. Healing is reduced by 1."
    },
    {
        "id": "echoes",
        "name": "Hall of Echoes",
        "desc": "Cards with draw gain +1 extra draw."
    },
    {
        "id": "gravity",
        "name": "Heavy Gravity Day",
        "desc": "Shield cards gain +1 shield. Draw effects lose 1."
    },
]

COSMETICS = {
    "tables": ["neon-lagoon", "velvet-casino", "moon-parlor", "gold-doom"],
    "backs": ["stars", "skulls", "koi", "glitch"],
    "particles": ["sparkles", "bubbles", "embers", "confetti"]
}

def stable_daily_modifier():
    seed = int(date.today().strftime("%Y%m%d"))
    rng = random.Random(seed)
    return rng.choice(DAILY_MODIFIERS)

DAILY = stable_daily_modifier()

def card_by_id(card_id):
    for c in CARD_POOL:
        if c["id"] == card_id:
            return copy.deepcopy(c)
    return None

def random_shop():
    base = [c for c in CARD_POOL if c["id"] != "chaos"]
    return [copy.deepcopy(random.choice(base)) for _ in range(5)]

def make_starter_deck():
    return ["strike", "strike", "guard", "guard", "spark", "medic", "greed", "venom"]

def generate_room_code():
    while True:
        code = "".join(random.choices(string.ascii_uppercase + string.digits, k=5))
        if code not in ROOMS:
            return code

def shuffle_in_place(lst):
    random.shuffle(lst)
    return lst

def new_player(username):
    deck = make_starter_deck()
    shuffle_in_place(deck)
    return {
        "username": username[:18],
        "hp": MAX_HP,
        "deck": deck,
        "discard": [],
        "hand": [],
        "coins": STARTING_COINS,
        "bonus_draw": 0,
        "bonus_gold": 0,
        "shield_now": 0,
        "mission": copy.deepcopy(random.choice(MISSIONS)),
        "mission_progress": 0,
        "mission_done": False,
        "achievements": [],
        "stats": {
            "loss_streak": 0,
            "cosmetic_changes": 0,
            "blue_buys_today": 0,
            "heal_total": 0,
        },
        "cosmetics": {
            "table": random.choice(COSMETICS["tables"]),
            "back": random.choice(COSMETICS["backs"]),
            "particles": random.choice(COSMETICS["particles"]),
        },
        "submitted": [],
        "bought": False,
    }

def ensure_draw(player, n):
    while len(player["hand"]) < n:
        if not player["deck"]:
            if not player["discard"]:
                break
            player["deck"] = player["discard"][:]
            player["discard"] = []
            shuffle_in_place(player["deck"])
        if player["deck"]:
            player["hand"].append(player["deck"].pop())

def start_match(room):
    room["phase"] = "shop"
    room["round"] = 1
    room["log"] = [f"Daily rule: {room['daily']['name']} — {room['daily']['desc']}"]
    for player in room["players"].values():
        ensure_draw(player, HAND_SIZE)

def effective_cost(card, room):
    cost = card["cost"]
    if room["daily"]["id"] == "underwater" and card["color"] == "blue":
        cost -= 1
    return max(1, cost) if card["cost"] > 0 else 0

def compute_bundle(cards, room):
    bundle = {"power": 0, "shield": 0, "heal": 0, "draw": 0, "gold": 0, "poison": 0}
    for card in cards:
        bundle["power"] += card["power"]
        bundle["shield"] += card["shield"]
        bundle["heal"] += card["heal"]
        bundle["draw"] += card["draw"]
        bundle["gold"] += card["gold"]
        bundle["poison"] += card["poison"]

        if room["daily"]["id"] == "underwater" and card["power"] > 0:
            bundle["power"] -= 1
        if room["daily"]["id"] == "heatwave" and card["color"] == "red":
            bundle["power"] += 1
        if room["daily"]["id"] == "heatwave" and card["heal"] > 0:
            bundle["heal"] -= 1
        if room["daily"]["id"] == "echoes" and card["draw"] > 0:
            bundle["draw"] += 1
        if room["daily"]["id"] == "gravity" and card["shield"] > 0:
            bundle["shield"] += 1
        if room["daily"]["id"] == "gravity" and card["draw"] > 0:
            bundle["draw"] -= 1

    for k in bundle:
        bundle[k] = max(0, bundle[k])
    return bundle

def maybe_complete_mission(player):
    mission = player["mission"]
    if player["mission_done"]:
        return None
    if player["mission_progress"] >= mission["goal"]:
        player["mission_done"] = True
        player["discard"].append("chaos")
        return f"{player['username']} completed the secret mission '{mission['name']}' and gained a Chaos Crown."
    return None

def give_achievement(player, key):
    if key not in player["achievements"]:
        player["achievements"].append(key)

def resolve_round(room):
    sids = list(room["players"].keys())
    p1 = room["players"][sids[0]]
    p2 = room["players"][sids[1]]

    cards1 = [card_by_id(cid) for cid in p1["submitted"]]
    cards2 = [card_by_id(cid) for cid in p2["submitted"]]

    b1 = compute_bundle(cards1, room)
    b2 = compute_bundle(cards2, room)

    p1["shield_now"] = b1["shield"]
    p2["shield_now"] = b2["shield"]

    damage_to_p2 = max(0, b1["power"] - b2["shield"]) + b1["poison"]
    damage_to_p1 = max(0, b2["power"] - b1["shield"]) + b2["poison"]

    p1["hp"] = min(MAX_HP, p1["hp"] - damage_to_p1 + b1["heal"])
    p2["hp"] = min(MAX_HP, p2["hp"] - damage_to_p2 + b2["heal"])

    p1["bonus_draw"] = b1["draw"]
    p2["bonus_draw"] = b2["draw"]
    p1["bonus_gold"] = b1["gold"]
    p2["bonus_gold"] = b2["gold"]

    p1["stats"]["heal_total"] += b1["heal"]
    p2["stats"]["heal_total"] += b2["heal"]

    if damage_to_p2 >= 6:
        p1["mission_progress"] = max(p1["mission_progress"], 1) if p1["mission"]["id"] == "big_hit" else p1["mission_progress"]
    if damage_to_p1 >= 6:
        p2["mission_progress"] = max(p2["mission_progress"], 1) if p2["mission"]["id"] == "big_hit" else p2["mission_progress"]

    if b1["shield"] >= 6 and p1["mission"]["id"] == "shield":
        p1["mission_progress"] = 1
    if b2["shield"] >= 6 and p2["mission"]["id"] == "shield":
        p2["mission_progress"] = 1

    if p1["mission"]["id"] == "healer":
        p1["mission_progress"] = p1["stats"]["heal_total"]
    if p2["mission"]["id"] == "healer":
        p2["mission_progress"] = p2["stats"]["heal_total"]

    log_line = (
        f"Round {room['round']}: {p1['username']} dealt {damage_to_p2} / blocked {b1['shield']} / healed {b1['heal']} "
        f"— {p2['username']} dealt {damage_to_p1} / blocked {b2['shield']} / healed {b2['heal']}."
    )
    room["log"].append(log_line)

    m1 = maybe_complete_mission(p1)
    m2 = maybe_complete_mission(p2)
    if m1:
        room["log"].append(m1)
    if m2:
        room["log"].append(m2)

    p1["submitted"] = []
    p2["submitted"] = []
    p1["bought"] = False
    p2["bought"] = False

    p1["coins"] = STARTING_COINS + p1["bonus_gold"]
    p2["coins"] = STARTING_COINS + p2["bonus_gold"]

    ensure_draw(p1, HAND_SIZE + p1["bonus_draw"])
    ensure_draw(p2, HAND_SIZE + p2["bonus_draw"])

    p1["bonus_draw"] = 0
    p2["bonus_draw"] = 0
    p1["bonus_gold"] = 0
    p2["bonus_gold"] = 0

    room["round"] += 1
    room["phase"] = "shop"

    loser = None
    winner = None
    if p1["hp"] <= 0 or p2["hp"] <= 0:
        room["phase"] = "game_over"
        if p1["hp"] > p2["hp"]:
            winner, loser = p1, p2
        elif p2["hp"] > p1["hp"]:
            winner, loser = p2, p1

        if winner and loser:
            loser["stats"]["loss_streak"] += 1
            winner["stats"]["loss_streak"] = 0
            if loser["stats"]["loss_streak"] >= 7:
                give_achievement(loser, "dignity")
            if winner["hp"] == 1:
                give_achievement(winner, "exactly_one")
            room["log"].append(f"{winner['username']} wins the game.")
        else:
            room["log"].append("The game ends in a dramatic tie.")

def serialize_card_for_client(card, room):
    item = copy.deepcopy(card)
    item["effective_cost"] = effective_cost(card, room)
    return item

def serialize_state(room, sid):
    you = room["players"][sid]
    opponents = []
    for other_sid, other in room["players"].items():
        if other_sid == sid:
            continue
        opponents.append({
            "username": other["username"],
            "hp": other["hp"],
            "hand_count": len(other["hand"]),
            "coins": other["coins"],
            "shield_now": other["shield_now"],
            "cosmetics": other["cosmetics"],
            "achievement_count": len(other["achievements"]),
        })

    return {
        "room": room["code"],
        "phase": room["phase"],
        "round": room["round"],
        "daily": room["daily"],
        "market": [serialize_card_for_client(c, room) for c in room["market"]],
        "you": {
            "username": you["username"],
            "hp": you["hp"],
            "coins": you["coins"],
            "deck_count": len(you["deck"]),
            "discard_count": len(you["discard"]),
            "hand": [card_by_id(cid) for cid in you["hand"]],
            "mission": {
                "name": you["mission"]["name"],
                "desc": you["mission"]["desc"],
                "goal": you["mission"]["goal"],
                "progress": you["mission_progress"],
                "done": you["mission_done"]
            },
            "cosmetics": you["cosmetics"],
            "achievements": [ACHIEVEMENTS[a] for a in you["achievements"]],
            "submitted": you["submitted"],
            "bought": you["bought"],
        },
        "opponents": opponents,
        "players_in_room": len(room["players"]),
        "log": room["log"][-8:],
        "cosmetics_catalog": COSMETICS,
    }

def broadcast_state(room_code):
    room = ROOMS.get(room_code)
    if not room:
        return
    for sid in list(room["players"].keys()):
        socketio.emit("state", serialize_state(room, sid), to=sid)

@app.route("/")
def home():
    return render_template("index.html", daily=DAILY)

@app.route("/create", methods=["POST"])
def create_room():
    username = request.form.get("username", "Player").strip() or "Player"
    code = generate_room_code()
    ROOMS[code] = {
        "code": code,
        "daily": copy.deepcopy(DAILY),
        "market": random_shop(),
        "players": {},
        "phase": "lobby",
        "round": 1,
        "log": []
    }
    return redirect(url_for("room_view", room_code=code, username=username))

@app.route("/join", methods=["POST"])
def join_existing():
    username = request.form.get("username", "Player").strip() or "Player"
    code = request.form.get("room_code", "").strip().upper()
    if code not in ROOMS:
        return redirect(url_for("home"))
    return redirect(url_for("room_view", room_code=code, username=username))

@app.route("/room/<room_code>")
def room_view(room_code):
    username = request.args.get("username", "Player")
    if room_code not in ROOMS:
        return redirect(url_for("home"))
    return render_template("room.html", room_code=room_code, username=username)

@socketio.on("join_game")
def on_join_game(data):
    room_code = data["room"]
    username = data["username"].strip()[:18] or "Player"

    if room_code not in ROOMS:
        emit("error_message", {"message": "Room not found."})
        return

    room = ROOMS[room_code]

    if len(room["players"]) >= 2 and request.sid not in room["players"]:
        emit("error_message", {"message": "Room is full."})
        return

    if request.sid not in room["players"]:
        room["players"][request.sid] = new_player(username)

    join_room(room_code)

    if len(room["players"]) == 2 and room["phase"] == "lobby":
        start_match(room)

    broadcast_state(room_code)

@socketio.on("buy_card")
def on_buy_card(data):
    room = ROOMS.get(data["room"])
    if not room or request.sid not in room["players"] or room["phase"] != "shop":
        return

    player = room["players"][request.sid]
    if player["bought"]:
        return

    idx = int(data["index"])
    if idx < 0 or idx >= len(room["market"]):
        return

    card = room["market"][idx]
    cost = effective_cost(card, room)
    if player["coins"] < cost:
        return

    player["coins"] -= cost
    player["discard"].append(card["id"])
    player["bought"] = True

    if player["mission"]["id"] == "blue_buyer" and card["color"] == "blue":
        player["mission_progress"] += 1

    if room["daily"]["id"] == "underwater" and card["color"] == "blue":
        player["stats"]["blue_buys_today"] += 1
        if player["stats"]["blue_buys_today"] >= 4:
            give_achievement(player, "underwater_addict")

    maybe_complete_mission(player)
    room["market"][idx] = copy.deepcopy(random.choice([c for c in CARD_POOL if c["id"] != "chaos"]))
    room["log"].append(f"{player['username']} bought {card['name']}.")
    broadcast_state(room["code"])

@socketio.on("submit_cards")
def on_submit_cards(data):
    room = ROOMS.get(data["room"])
    if not room or request.sid not in room["players"] or room["phase"] not in ("shop", "battle"):
        return

    player = room["players"][request.sid]
    selected = data.get("cards", [])[:3]

    clean = []
    hand_copy = player["hand"][:]
    for cid in selected:
        if cid in hand_copy:
            clean.append(cid)
            hand_copy.remove(cid)

    player["submitted"] = clean
    room["phase"] = "battle"
    room["log"].append(f"{player['username']} locked in {len(clean)} card(s).")

    everyone_ready = len(room["players"]) == 2 and all(p["submitted"] is not None and len(p["submitted"]) >= 0 for p in room["players"].values())
    both_nonempty = len(room["players"]) == 2 and all(len(p["submitted"]) > 0 for p in room["players"].values())

    if everyone_ready and both_nonempty:
        for p in room["players"].values():
            for cid in p["submitted"]:
                if cid in p["hand"]:
                    p["hand"].remove(cid)
                    p["discard"].append(cid)
            leftovers = p["hand"][:]
            p["discard"].extend(leftovers)
            p["hand"] = []
        resolve_round(room)

    broadcast_state(room["code"])

@socketio.on("set_cosmetic")
def on_set_cosmetic(data):
    room = ROOMS.get(data["room"])
    if not room or request.sid not in room["players"]:
        return

    player = room["players"][request.sid]
    category = data.get("category")
    value = data.get("value")

    if category not in COSMETICS or value not in COSMETICS[category]:
        return

    key = "table" if category == "tables" else "back" if category == "backs" else "particles"
    player["cosmetics"][key] = value
    player["stats"]["cosmetic_changes"] += 1

    if player["stats"]["cosmetic_changes"] >= 10:
        give_achievement(player, "fashion")

    broadcast_state(room["code"])

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)
