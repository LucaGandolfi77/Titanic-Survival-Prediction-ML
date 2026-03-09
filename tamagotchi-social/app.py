import os
import random
from datetime import datetime
from functools import wraps

from flask import Flask, jsonify, render_template, request, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import check_password_hash, generate_password_hash

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "change-me-in-production-abc123xyz")
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///" + os.path.join(BASE_DIR, "tamagotchi.db")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_HTTPONLY"] = True

db = SQLAlchemy(app)

# ── Models ──────────────────────────────────────────────────────────────────

class User(db.Model):
    id           = db.Column(db.Integer, primary_key=True)
    username     = db.Column(db.String(50), unique=True, nullable=False)
    password_hash= db.Column(db.String(256), nullable=False)
    created_at   = db.Column(db.DateTime, default=datetime.utcnow)
    pet          = db.relationship("Pet", backref="owner", uselist=False,
                                   cascade="all, delete-orphan")

SPECIES = ["blob", "cat", "dragon", "bunny"]

class Pet(db.Model):
    id              = db.Column(db.Integer, primary_key=True)
    user_id         = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    name            = db.Column(db.String(50), nullable=False)
    species         = db.Column(db.String(20), default="blob")
    hunger          = db.Column(db.Float, default=80.0)
    happiness       = db.Column(db.Float, default=80.0)
    energy          = db.Column(db.Float, default=80.0)
    health          = db.Column(db.Float, default=100.0)
    level           = db.Column(db.Integer, default=1)
    experience      = db.Column(db.Float, default=0.0)
    age_days        = db.Column(db.Float, default=0.0)
    evolution_stage = db.Column(db.Integer, default=1)
    is_sleeping     = db.Column(db.Boolean, default=False)
    is_sick         = db.Column(db.Boolean, default=False)
    is_alive        = db.Column(db.Boolean, default=True)
    wins            = db.Column(db.Integer, default=0)
    losses          = db.Column(db.Integer, default=0)
    created_at      = db.Column(db.DateTime, default=datetime.utcnow)
    last_updated    = db.Column(db.DateTime, default=datetime.utcnow)

    EVO_NAMES = {1:"Egg", 2:"Baby", 3:"Child", 4:"Teen", 5:"Adult"}
    EVO_THRESH = {2:5, 3:10, 4:20, 5:35}

    def update_stats(self):
        if not self.is_alive:
            return
        now = datetime.utcnow()
        elapsed = (now - self.last_updated).total_seconds()
        if elapsed < 1:
            return
        hours = elapsed / 3600
        self.age_days += elapsed / 86400

        if self.is_sleeping:
            self.energy = min(100, self.energy + hours * 20)
            self.hunger = max(0,   self.hunger  - hours * 2)
            if self.energy >= 100:
                self.is_sleeping = False
        else:
            self.hunger    = max(0, self.hunger    - hours * 8)
            self.happiness = max(0, self.happiness - hours * 5)
            self.energy    = max(0, self.energy    - hours * 3)

        avg = (self.hunger + self.happiness + self.energy) / 3
        if avg < 20:
            self.health = max(0, self.health - hours * 5)
            self.is_sick = True
        elif avg > 65:
            self.health = min(100, self.health + hours * 1.5)
            if self.health > 75:
                self.is_sick = False

        if self.health <= 0:
            self.is_alive = False

        for stage, thresh in self.EVO_THRESH.items():
            if self.level >= thresh and self.evolution_stage < stage:
                self.evolution_stage = stage

        self.last_updated = now
        db.session.commit()

    def gain_xp(self, amount):
        self.experience += amount
        while self.experience >= 100:
            self.experience -= 100
            self.level += 1
        db.session.commit()

    def mood(self):
        if not self.is_alive:  return "dead"
        if self.is_sick:       return "sick"
        if self.is_sleeping:   return "sleeping"
        avg = (self.hunger + self.happiness + self.energy) / 3
        if avg >= 75: return "happy"
        if avg >= 50: return "content"
        if avg >= 30: return "sad"
        return "critical"

    def to_dict(self):
        self.update_stats()
        return {
            "id": self.id, "name": self.name, "species": self.species,
            "hunger": round(self.hunger, 1), "happiness": round(self.happiness, 1),
            "energy": round(self.energy, 1),  "health": round(self.health, 1),
            "level": self.level, "experience": round(self.experience, 1),
            "age_days": round(self.age_days, 1),
            "evolution_stage": self.evolution_stage,
            "evolution_name": self.EVO_NAMES.get(self.evolution_stage, "?"),
            "is_sleeping": self.is_sleeping, "is_sick": self.is_sick,
            "is_alive": self.is_alive, "mood": self.mood(),
            "wins": self.wins, "losses": self.losses,
            "owner": self.owner.username,
        }


class Friendship(db.Model):
    id           = db.Column(db.Integer, primary_key=True)
    requester_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    receiver_id  = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    status       = db.Column(db.String(20), default="pending")
    created_at   = db.Column(db.DateTime, default=datetime.utcnow)
    requester    = db.relationship("User", foreign_keys=[requester_id])
    receiver     = db.relationship("User", foreign_keys=[receiver_id])


class Challenge(db.Model):
    id             = db.Column(db.Integer, primary_key=True)
    challenger_id  = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    challenged_id  = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    status         = db.Column(db.String(20), default="pending")
    winner_id      = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=True)
    battle_log     = db.Column(db.Text, nullable=True)
    created_at     = db.Column(db.DateTime, default=datetime.utcnow)
    challenger     = db.relationship("User", foreign_keys=[challenger_id])
    challenged     = db.relationship("User", foreign_keys=[challenged_id])
    winner         = db.relationship("User", foreign_keys=[winner_id])

# ── Helpers ─────────────────────────────────────────────────────────────────

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("user_id"):
            return jsonify({"error": "Not authenticated"}), 401
        return f(*args, **kwargs)
    return decorated

def current_user():
    return User.query.get(session["user_id"])

# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")

# Auth
@app.route("/api/register", methods=["POST"])
def register():
    d = request.get_json()
    username = (d.get("username") or "").strip()
    password = d.get("password") or ""
    if not username or not password:
        return jsonify({"error": "Username and password required"}), 400
    if len(username) < 3:
        return jsonify({"error": "Username must be ≥ 3 characters"}), 400
    if len(password) < 6:
        return jsonify({"error": "Password must be ≥ 6 characters"}), 400
    if User.query.filter_by(username=username).first():
        return jsonify({"error": "Username already taken"}), 400
    user = User(username=username, password_hash=generate_password_hash(password))
    db.session.add(user)
    db.session.commit()
    session["user_id"] = user.id
    return jsonify({"message": "Account created!", "user": {"id": user.id, "username": user.username}}), 201

@app.route("/api/login", methods=["POST"])
def login():
    d = request.get_json()
    user = User.query.filter_by(username=d.get("username")).first()
    if not user or not check_password_hash(user.password_hash, d.get("password", "")):
        return jsonify({"error": "Invalid credentials"}), 401
    session["user_id"] = user.id
    return jsonify({"message": "Logged in!", "user": {"id": user.id, "username": user.username}})

@app.route("/api/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify({"message": "Logged out!"})

@app.route("/api/me")
@login_required
def me():
    u = current_user()
    return jsonify({"id": u.id, "username": u.username})

# Pet
@app.route("/api/pet")
@login_required
def get_pet():
    pet = current_user().pet
    return jsonify({"pet": pet.to_dict() if pet else None})

@app.route("/api/pet/create", methods=["POST"])
@login_required
def create_pet():
    u = current_user()
    if u.pet:
        return jsonify({"error": "You already have a pet!"}), 400
    d = request.get_json()
    name    = (d.get("name") or "Buddy").strip()
    species = d.get("species", "blob")
    if species not in SPECIES:
        species = "blob"
    if not name:
        return jsonify({"error": "Pet name is required"}), 400
    pet = Pet(user_id=u.id, name=name, species=species)
    db.session.add(pet)
    db.session.commit()
    return jsonify({"message": f"{name} has been born! 🥚", "pet": pet.to_dict()}), 201

@app.route("/api/pet/feed", methods=["POST"])
@login_required
def feed_pet():
    pet = current_user().pet
    if not pet or not pet.is_alive:
        return jsonify({"error": "No alive pet"}), 404
    pet.update_stats()
    if pet.is_sleeping:
        return jsonify({"error": f"{pet.name} is sleeping!"}), 400
    if pet.hunger >= 95:
        return jsonify({"error": f"{pet.name} is not hungry!"}), 400
    pet.hunger = min(100, pet.hunger + random.uniform(22, 30))
    pet.gain_xp(5)
    return jsonify({"message": f"{pet.name} enjoyed the meal! 🍎", "pet": pet.to_dict()})

@app.route("/api/pet/play", methods=["POST"])
@login_required
def play_pet():
    pet = current_user().pet
    if not pet or not pet.is_alive:
        return jsonify({"error": "No alive pet"}), 404
    pet.update_stats()
    if pet.is_sleeping:
        return jsonify({"error": f"{pet.name} is sleeping!"}), 400
    if pet.energy < 15:
        return jsonify({"error": f"{pet.name} is too tired to play!"}), 400
    if pet.happiness >= 95:
        return jsonify({"error": f"{pet.name} is already very happy!"}), 400
    pet.happiness = min(100, pet.happiness + random.uniform(22, 30))
    pet.energy    = max(0,   pet.energy    - random.uniform(10, 14))
    pet.gain_xp(8)
    return jsonify({"message": f"{pet.name} had a great time! 🎮", "pet": pet.to_dict()})

@app.route("/api/pet/sleep", methods=["POST"])
@login_required
def sleep_pet():
    pet = current_user().pet
    if not pet or not pet.is_alive:
        return jsonify({"error": "No alive pet"}), 404
    pet.update_stats()
    if pet.is_sleeping:
        pet.is_sleeping = False
        db.session.commit()
        return jsonify({"message": f"{pet.name} woke up! 🌅", "pet": pet.to_dict()})
    pet.is_sleeping = True
    db.session.commit()
    return jsonify({"message": f"{pet.name} is sleeping... 💤", "pet": pet.to_dict()})

@app.route("/api/pet/heal", methods=["POST"])
@login_required
def heal_pet():
    pet = current_user().pet
    if not pet or not pet.is_alive:
        return jsonify({"error": "No alive pet"}), 404
    pet.update_stats()
    if not pet.is_sick:
        return jsonify({"error": f"{pet.name} is healthy!"}), 400
    pet.health = min(100, pet.health + 35)
    pet.is_sick = False
    pet.gain_xp(3)
    return jsonify({"message": f"{pet.name} feels better! 💊", "pet": pet.to_dict()})

@app.route("/api/pet/revive", methods=["POST"])
@login_required
def revive_pet():
    u = current_user()
    if u.pet:
        db.session.delete(u.pet)
        db.session.commit()
    return jsonify({"message": "Your pet has passed. Create a new one to continue."})

# Social – users
@app.route("/api/social/users")
@login_required
def social_users():
    u = current_user()
    users = User.query.filter(User.id != u.id).all()
    result = []
    for usr in users:
        if usr.pet and usr.pet.is_alive:
            result.append({
                "id": usr.id, "username": usr.username,
                "pet": {
                    "name": usr.pet.name, "species": usr.pet.species,
                    "level": usr.pet.level,
                    "evolution_name": usr.pet.EVO_NAMES.get(usr.pet.evolution_stage, "?"),
                    "mood": usr.pet.mood(),
                }
            })
    return jsonify({"users": result})

# Friends
@app.route("/api/social/friends")
@login_required
def get_friends():
    uid = session["user_id"]
    fships = Friendship.query.filter(
        ((Friendship.requester_id == uid) | (Friendship.receiver_id == uid)) &
        (Friendship.status == "accepted")
    ).all()
    out = []
    for f in fships:
        friend = f.receiver if f.requester_id == uid else f.requester
        out.append({
            "friendship_id": f.id,
            "user": {"id": friend.id, "username": friend.username},
            "pet": friend.pet.to_dict() if friend.pet else None,
        })
    return jsonify({"friends": out})

@app.route("/api/social/friend/requests")
@login_required
def friend_requests():
    uid = session["user_id"]
    reqs = Friendship.query.filter_by(receiver_id=uid, status="pending").all()
    return jsonify({"requests": [
        {"id": r.id,
         "requester": {"id": r.requester.id, "username": r.requester.username},
         "pet": r.requester.pet.to_dict() if r.requester.pet else None}
        for r in reqs
    ]})

@app.route("/api/social/friend/request", methods=["POST"])
@login_required
def send_friend_request():
    u = current_user()
    target_id = request.get_json().get("user_id")
    if target_id == u.id:
        return jsonify({"error": "Cannot befriend yourself!"}), 400
    target = User.query.get(target_id)
    if not target:
        return jsonify({"error": "User not found"}), 404
    existing = Friendship.query.filter(
        ((Friendship.requester_id == u.id) & (Friendship.receiver_id == target_id)) |
        ((Friendship.requester_id == target_id) & (Friendship.receiver_id == u.id))
    ).first()
    if existing:
        return jsonify({"error": "Request already exists or already friends"}), 400
    db.session.add(Friendship(requester_id=u.id, receiver_id=target_id))
    db.session.commit()
    return jsonify({"message": f"Friend request sent to {target.username}!"})

@app.route("/api/social/friend/respond", methods=["POST"])
@login_required
def respond_friend():
    uid = session["user_id"]
    d = request.get_json()
    f = Friendship.query.get(d.get("friendship_id"))
    if not f or f.receiver_id != uid:
        return jsonify({"error": "Not found"}), 404
    f.status = "accepted" if d.get("action") == "accept" else "declined"
    db.session.commit()
    msg = f"You are now friends with {f.requester.username}! 🤝" if f.status == "accepted" else "Request declined."
    return jsonify({"message": msg})

# Interact (friends only)
@app.route("/api/social/interact", methods=["POST"])
@login_required
def interact():
    u = current_user()
    target_id = request.get_json().get("user_id")
    if not u.pet or not u.pet.is_alive:
        return jsonify({"error": "You need a living pet!"}), 400
    target = User.query.get(target_id)
    if not target or not target.pet or not target.pet.is_alive:
        return jsonify({"error": "Target has no alive pet"}), 404
    uid = u.id
    f = Friendship.query.filter(
        ((Friendship.requester_id == uid) & (Friendship.receiver_id == target_id)) |
        ((Friendship.requester_id == target_id) & (Friendship.receiver_id == uid)),
        Friendship.status == "accepted"
    ).first()
    if not f:
        return jsonify({"error": "You can only interact with friends!"}), 400
    u.pet.happiness = min(100, u.pet.happiness + 15)
    u.pet.gain_xp(10)
    return jsonify({"message": f"{u.pet.name} played with {target.pet.name}! 🎉", "pet": u.pet.to_dict()})

# Challenges
@app.route("/api/social/challenges")
@login_required
def get_challenges():
    uid = session["user_id"]
    def c2d(c):
        return {
            "id": c.id,
            "challenger": {"id": c.challenger.id, "username": c.challenger.username},
            "challenged": {"id": c.challenged.id, "username": c.challenged.username},
            "status": c.status,
            "winner": {"id": c.winner.id, "username": c.winner.username} if c.winner else None,
            "battle_log": c.battle_log,
        }
    incoming  = Challenge.query.filter_by(challenged_id=uid,  status="pending").all()
    outgoing  = Challenge.query.filter_by(challenger_id=uid,  status="pending").all()
    completed = Challenge.query.filter(
        (Challenge.challenger_id == uid) | (Challenge.challenged_id == uid),
        Challenge.status == "completed"
    ).order_by(Challenge.created_at.desc()).limit(10).all()
    return jsonify({
        "incoming":  [c2d(c) for c in incoming],
        "outgoing":  [c2d(c) for c in outgoing],
        "completed": [c2d(c) for c in completed],
    })

@app.route("/api/social/challenge", methods=["POST"])
@login_required
def send_challenge():
    u = current_user()
    target_id = request.get_json().get("user_id")
    if not u.pet or not u.pet.is_alive:
        return jsonify({"error": "You need a living pet to challenge!"}), 400
    target = User.query.get(target_id)
    if not target or not target.pet or not target.pet.is_alive:
        return jsonify({"error": "Target has no alive pet"}), 404
    existing = Challenge.query.filter(
        ((Challenge.challenger_id == u.id) & (Challenge.challenged_id == target_id)) |
        ((Challenge.challenger_id == target_id) & (Challenge.challenged_id == u.id)),
        Challenge.status == "pending"
    ).first()
    if existing:
        return jsonify({"error": "A pending challenge already exists!"}), 400
    db.session.add(Challenge(challenger_id=u.id, challenged_id=target_id))
    db.session.commit()
    return jsonify({"message": f"Challenge sent to {target.username}! ⚔️"})

@app.route("/api/social/challenge/respond", methods=["POST"])
@login_required
def respond_challenge():
    uid = session["user_id"]
    d = request.get_json()
    c = Challenge.query.get(d.get("challenge_id"))
    if not c or c.challenged_id != uid:
        return jsonify({"error": "Challenge not found"}), 404
    if d.get("action") == "decline":
        c.status = "declined"
        db.session.commit()
        return jsonify({"message": "Challenge declined."})
    if d.get("action") != "accept":
        return jsonify({"error": "Invalid action"}), 400

    cp = c.challenger.pet
    dp = c.challenged.pet
    if not cp or not dp:
        return jsonify({"error": "Both pets must exist!"}), 400

    def power(p):
        return (p.level * 10 + p.health * 0.5 + p.happiness * 0.3 +
                p.wins * 2) * random.uniform(0.85, 1.15)

    cp_pow = power(cp); dp_pow = power(dp)
    log = [
        f"⚔️  {cp.name} (Lv.{cp.level}) vs {dp.name} (Lv.{dp.level})",
        f"💪 {cp.name}: {cp_pow:.0f} pts  |  {dp.name}: {dp_pow:.0f} pts",
    ]
    if cp_pow > dp_pow:
        winner, loser = c.challenger, c.challenged
        log.append(f"🏆 {cp.name} wins the battle!")
    else:
        winner, loser = c.challenged, c.challenger
        log.append(f"🏆 {dp.name} wins the battle!")

    winner.pet.wins   += 1;  winner.pet.gain_xp(15)
    loser.pet.losses  += 1;  loser.pet.gain_xp(5)
    c.status    = "completed"
    c.winner_id = winner.id
    c.battle_log = "\n".join(log)
    db.session.commit()

    return jsonify({
        "message": f"Battle complete! {winner.pet.name} wins! 🏆",
        "battle_log": log, "winner": winner.username,
        "pet": current_user().pet.to_dict(),
    })

@app.route("/api/leaderboard")
@login_required
def leaderboard():
    pets = Pet.query.filter_by(is_alive=True).order_by(Pet.level.desc(), Pet.wins.desc()).limit(20).all()
    return jsonify({"leaderboard": [
        {"rank": i+1, "owner": p.owner.username, "name": p.name,
         "species": p.species, "level": p.level, "wins": p.wins,
         "evolution_name": p.EVO_NAMES.get(p.evolution_stage, "?")}
        for i, p in enumerate(pets)
    ]})

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)
