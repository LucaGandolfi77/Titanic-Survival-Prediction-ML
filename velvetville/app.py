from flask import Flask, render_template, jsonify, request, session
from flask_cors import CORS
from flask_login import login_required
from flask_socketio import emit
from config.settings import config_by_name
from extensions import db, login_manager, socketio
from models import User, Vote, Badge, UserProfile
import os
import random
import datetime
import json
import html
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from collections import Counter

limiter = Limiter(key_func=get_remote_address)

SENTIMENT_KEYWORDS = {
    "positive": ["bellissimo", "stunning", "amazing", "love", "adoro", "perfetto",
                 "fire", "slay", "gorgeous", "wow", "incredible",
                 "favoloso", "meraviglioso", "splendido", "elegante", "chic"],
    "negative": ["trash", "bad", "ugly", "terrible", "horrible", "no", "nak",
                 "brutto", "orribile", "orrendo", "scarso", "male"],
}

LANGUAGES = {
    "en": {
        "welcome": "Welcome to Velvetville",
        "enter_runway": "Enter the Runway",
        "username": "Username",
        "login_hint": "No account needed — just pick a name!",
        "total_votes": "Total Votes",
        "unique_voters": "Unique Voters",
        "avg_score": "Avg Score",
        "positive": "Positive",
        "trending": "Trending",
        "personalized": "Personalized",
    },
    "it": {
        "welcome": "Benvenuto a Velvetville",
        "enter_runway": "Entra nella Runway",
        "username": "Username",
        "login_hint": "Nessun account richiesto — scegli un nome!",
        "total_votes": "Voti Totali",
        "unique_voters": "Votatori Unici",
        "avg_score": "Voto Medio",
        "positive": "Positivo",
        "trending": "Tendenze",
        "personalized": "Personalizzato",
    },
    "es": {
        "welcome": "Bienvenido a Velvetville",
        "enter_runway": "Entra al Runway",
        "username": "Usuario",
        "login_hint": "¡Sin cuenta necesaria!",
        "total_votes": "Votos Totales",
        "unique_voters": "Votantes Únicos",
        "avg_score": "Puntuación Media",
        "positive": "Positivo",
        "trending": "Tendencias",
        "personalized": "Personalizado",
    },
}


def get_locale():
    lang = session.get("lang", "en")
    if lang not in LANGUAGES:
        lang = "en"
    return LANGUAGES[lang]


def t(key):
    locale = get_locale()
    return locale.get(key, key)


def analyze_sentiment(text):
    text_lower = text.lower()
    pos_score = sum(1 for kw in SENTIMENT_KEYWORDS["positive"] if kw in text_lower)
    neg_score = sum(1 for kw in SENTIMENT_KEYWORDS["negative"] if kw in text_lower)
    if pos_score > neg_score:
        return "positive"
    elif neg_score > pos_score:
        return "negative"
    return "neutral"


def get_weather_theme(lat, lon):
    try:
        import urllib.request
        url = (
            f"http://api.openweathermap.org/data/2.5/weather"
            f"?lat={lat}&lon={lon}"
            f"&appid={os.environ.get('OPENWEATHER_API_KEY', '')}"
            f"&units=metric"
        )
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read())
            temp = data.get("main", {}).get("temp", 15)
            weather = data.get("weather", [{}])[0].get("main", "Clear")
            if temp > 25 or weather in ["Rain", "Drizzle", "Thunderstorm"]:
                return "tropical"
            elif temp < 10:
                return "winter"
            elif weather in ["Clouds", "Mist"]:
                return "moody"
            return "classic"
    except Exception:
        return "classic"


def create_app(config_name=None):
    if config_name is None:
        config_name = os.environ.get("FLASK_ENV", "development")

    app = Flask(__name__)
    app.config.from_object(config_by_name[config_name])

    CORS(app)
    db.init_app(app)
    login_manager.init_app(app)
    socketio.init_app(app)
    limiter.init_app(app)
    login_manager.login_view = "index"

    with app.app_context():
        db.create_all()

    @login_manager.user_loader
    def load_user(user_id):
        return db.session.get(User, int(user_id))

    @app.errorhandler(404)
    def not_found(e):
        return jsonify({"error": "Not found"}), 404

    @app.errorhandler(500)
    def internal_error(e):
        db.session.rollback()
        return jsonify({"error": "Internal server error"}), 500

    @socketio.on("connect")
    @limiter.limit("10 per minute")
    def handle_connect():
        username = session.get("username", "Anonymous")
        emit("user_connected", {
            "username": username,
            "message": f"{username} joined the runway",
        }, room="lobby")

    @socketio.on("disconnect")
    def handle_disconnect():
        username = session.get("username", "Anonymous")
        emit("user_disconnected", {
            "username": username,
            "message": f"{username} left the runway",
        }, room="lobby")

    @socketio.on("chat_message")
    @limiter.limit("5 per second")
    def handle_chat(data):
        username = session.get("username", "Anonymous")
        message = data.get("message", "").strip()[:200]
        if not message:
            return
        sentiment = analyze_sentiment(message)
        emit("chat_message", {
            "username": username,
            "message": message,
            "sentiment": sentiment,
            "timestamp": datetime.datetime.utcnow().isoformat(),
        }, room="lobby")

    @socketio.on("vote_broadcast")
    @limiter.limit("3 per second")
    def handle_vote_broadcast(data):
        username = session.get("username", "Anonymous")
        target = data.get("target", "")
        score = data.get("score", 5)
        emit("vote_received", {
            "username": username,
            "target": target,
            "score": score,
        }, room="lobby")

    @app.route("/")
    def index():
        return render_template("login.html")

    @app.route("/game")
    @login_required
    def game():
        event_data = {"name": "Summer Fashion Week", "multiplier": 2}
        profile = UserProfile.query.filter_by(
            user_id=session.get("user_id")
        ).first()
        if profile:
            event_data["xp"] = profile.xp
            event_data["level"] = get_level(profile.xp)
            event_data["badges"] = [b.name for b in Badge.query.filter_by(
                user_id=session.get("user_id")
            ).all()]
        event_data["lang"] = session.get("lang", "en")
        return render_template("game.html", event=event_data)

    @app.route("/analytics")
    @login_required
    def analytics():
        return render_template("analytics.html")

    @app.route("/set-language", methods=["POST"])
    @login_required
    def set_language():
        data = request.get_json(silent=True) or {}
        lang = data.get("language", "en")
        if lang in LANGUAGES:
            session["lang"] = lang
            return jsonify({"status": "success", "language": lang})
        return jsonify({"status": "error", "message": "Unsupported language"}), 400

    @app.route("/api/translations/<lang>")
    def translations(lang):
        if lang in LANGUAGES:
            return jsonify(LANGUAGES[lang])
        return jsonify(LANGUAGES["en"])

    @app.route("/api/notifications/subscribe", methods=["POST"])
    @login_required
    def subscribe_push():
        data = request.get_json(silent=True) or {}
        subscription = data.get("subscription")
        if not subscription:
            return jsonify({"status": "error"}), 400

        sub_data = json.dumps(subscription)
        existing = UserProfile.query.filter_by(
            user_id=session.get("user_id")
        ).first()
        if existing:
            existing.push_subscription = sub_data
            db.session.add(existing)
            db.session.commit()
        return jsonify({"status": "success"})

    @app.route("/api/notifications/unsubscribe", methods=["POST"])
    @login_required
    def unsubscribe_push():
        profile = UserProfile.query.filter_by(
            user_id=session.get("user_id")
        ).first()
        if profile:
            profile.push_subscription = None
            db.session.add(profile)
            db.session.commit()
        return jsonify({"status": "success"})

    @app.route("/api/notifications/send", methods=["POST"])
    @login_required
    def send_notification():
        data = request.get_json(silent=True) or {}
        profile = UserProfile.query.filter_by(
            user_id=session.get("user_id")
        ).first()
        if not profile or not profile.push_subscription:
            return jsonify({"status": "error", "message": "Not subscribed"}), 400

        try:
            from pywebpush import webpush, WebPushException
            try:
                webpush(
                    subscription_info=json.loads(profile.push_subscription),
                    data=json.dumps(data),
                )
            except ImportError:
                pass
            except WebPushException as e:
                return jsonify({"status": "error", "message": str(e)}), 500
        except Exception:
            pass

        return jsonify({"status": "success"})

    @app.route("/api/export/ranking", methods=["GET"])
    @login_required
    def export_ranking():
        results = (
            db.session.query(
                Vote.target_username,
                db.func.avg(Vote.score).label("avg_score"),
                db.func.count(Vote.id).label("votes"),
            )
            .group_by(Vote.target_username)
            .order_by(db.func.avg(Vote.score).desc())
            .all()
        )

        export = {
            "title": "Velvetville Ranking",
            "generated": datetime.datetime.utcnow().isoformat(),
            "entries": [{
                "username": r.target_username,
                "avg_score": round(r.avg_score, 1),
                "votes": r.votes,
            } for r in results],
        }

        fmt = request.args.get("format", "json")
        if fmt == "html":
            entries_html = ""
            for idx, entry in enumerate(export["entries"]):
                entries_html += f"""
                <li>
                    <strong>{idx + 1}. {html.escape(entry['username'])}</strong>
                    — {entry['avg_score']}/10 ({entry['votes']} votes)
                </li>"""
            html_doc = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>Velvetville Ranking</title>
<style>body{{font-family:sans-serif;padding:20px;}}h1{{color:#e94560;}}
ul{{list-style:none;padding:0;}}li{{padding:8px;border-bottom:1px solid #eee;}}</style>
</head><body><h1>{html.escape(export['title'])}</h1>
<ul>{entries_html}</ul>
<p>Generated: {export['generated']}</p></body></html>"""
            return html_doc, 200, {"Content-Type": "text/html"}

        return jsonify(export)

    @app.route("/api/analytics/summary")
    @login_required
    @limiter.limit("10 per minute")
    def analytics_summary():
        total_votes = Vote.query.count()
        users_voted = Vote.query.with_entities(Vote.voter_id).distinct().count()
        avg_score = db.session.query(db.func.avg(Vote.score)).scalar() or 0

        top_models = (
            db.session.query(
                Vote.target_username,
                db.func.avg(Vote.score).label("avg_score"),
                db.func.count(Vote.id).label("votes"),
            )
            .group_by(Vote.target_username)
            .order_by(db.func.avg(Vote.score).desc())
            .limit(5)
            .all()
        )

        score_distribution = (
            db.session.query(
                Vote.score,
                db.func.count(Vote.id).label("count"),
            )
            .group_by(Vote.score)
            .order_by(Vote.score)
            .all()
        )

        hourly_votes = (
            db.session.query(
                db.func.strftime('%H', Vote.timestamp).label("hour"),
                db.func.count(Vote.id).label("count"),
            )
            .group_by("hour")
            .order_by("hour")
            .all()
        )

        return jsonify({
            "total_votes": total_votes,
            "users_voted": users_voted,
            "avg_score": round(avg_score, 2),
            "top_models": [{
                "username": m.target_username,
                "avg_score": round(m.avg_score, 1),
                "votes": m.votes,
            } for m in top_models],
            "score_distribution": [{"score": s, "count": c} for s, c in score_distribution],
            "hourly_votes": [{"hour": h, "count": c} for h, c in hourly_votes],
        })

    @app.route("/api/analytics/sentiment")
    @login_required
    @limiter.limit("10 per minute")
    def analytics_sentiment():
        all_votes = Vote.query.all()
        positive = sum(1 for v in all_votes if v.score >= 8)
        negative = sum(1 for v in all_votes if v.score <= 3)
        neutral = sum(1 for v in all_votes if 4 <= v.score <= 7)
        total = len(all_votes)
        return jsonify({
            "positive": positive,
            "negative": negative,
            "neutral": neutral,
            "total": total,
            "positive_pct": round(positive / total * 100, 1) if total else 0,
            "negative_pct": round(negative / total * 100, 1) if total else 0,
        })

    @app.route("/api/recommendations")
    @login_required
    @limiter.limit("10 per minute")
    def recommendations():
        user_votes = Vote.query.filter_by(voter_id=session.get("user_id")).all()

        if not user_votes:
            trending = (
                db.session.query(
                    Vote.target_username,
                    db.func.avg(Vote.score).label("avg_score"),
                )
                .group_by(Vote.target_username)
                .order_by(db.func.avg(Vote.score).desc())
                .limit(3)
                .all()
            )
            return jsonify({
                "type": "trending",
                "models": [
                    {"username": t.target_username, "score": round(t.avg_score, 1)}
                    for t in trending
                ],
            })

        user_preferences = Counter()
        for vote in user_votes:
            user_preferences[vote.target_username] += vote.score

        all_model_votes = (
            db.session.query(
                Vote.target_username,
                db.func.avg(Vote.score).label("avg_score"),
                db.func.count(Vote.id).label("total_votes"),
            )
            .group_by(Vote.target_username)
            .all()
        )

        scored_models = []
        for model in all_model_votes:
            base_score = model.avg_score or 0
            user_affinity = user_preferences.get(model.target_username, 0) / 10.0
            popularity = min(model.total_votes / 10.0, 1.0)
            final_score = base_score * 0.5 + user_affinity * 30 + popularity * 20
            scored_models.append({
                "username": model.target_username,
                "score": round(final_score, 1),
                "avg_rating": round(base_score, 1),
                "total_votes": model.total_votes,
            })

        scored_models.sort(key=lambda x: x["score"], reverse=True)
        return jsonify({"type": "personalized", "models": scored_models[:5]})

    @app.route("/api/ambient/theme")
    @login_required
    def ambient_theme():
        lat = request.args.get("lat")
        lon = request.args.get("lon")
        if lat and lon:
            theme = get_weather_theme(float(lat), float(lon))
        else:
            hour = datetime.datetime.utcnow().hour
            if 6 <= hour < 12:
                theme = "morning"
            elif 18 <= hour < 22:
                theme = "sunset"
            else:
                theme = "classic"

        themes = {
            "classic": {"bg": "#1a1a2e", "accent": "#e94560", "name": "Classic"},
            "tropical": {"bg": "#1a3a2e", "accent": "#ffdd59", "name": "Tropical"},
            "winter": {"bg": "#1a2a3a", "accent": "#a0d2db", "name": "Winter"},
            "moody": {"bg": "#2a1a2e", "accent": "#8b5cf6", "name": "Moody"},
            "morning": {"bg": "#2e1a1a", "accent": "#fbbf24", "name": "Morning"},
            "sunset": {"bg": "#3a1a1a", "accent": "#f97316", "name": "Sunset"},
        }
        return jsonify({"theme": themes.get(theme, themes["classic"])})

    @app.route("/api/strut")
    @login_required
    @limiter.limit("10 per minute")
    def strut():
        models = [
            {"username": "GlowUpQueen", "outfit": "Sequin Gown & Boa"},
            {"username": "VaporwaveVixen", "outfit": "Neon Trenchcoat"},
            {"username": "BasicBob", "outfit": "Cargo Shorts & Flip Flops"},
            {"username": "VelvetVamp", "outfit": "Crushed Velvet Suit"},
        ]
        all_users = User.query.all()
        if all_users:
            u = random.choice(all_users)
            return jsonify({"username": u.username, "outfit": "Custom look"})
        return jsonify(random.choice(models))

    @app.route("/api/vote", methods=["POST"])
    @login_required
    @limiter.limit("10 per minute")
    def vote():
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"status": "error", "message": "Invalid JSON"}), 400

        target = data.get("target_username")
        score = data.get("score")

        if not target or not isinstance(target, str):
            return jsonify({"status": "error"}), 400

        if score is None or not isinstance(score, (int, float)):
            return jsonify({"status": "error", "message": "Invalid score"}), 400
        if score < 1 or score > 10:
            return jsonify({"status": "error", "message": "Score must be 1-10"}), 400

        try:
            new_vote = Vote(
                voter_id=session.get("user_id"),
                target_username=target,
                score=int(score),
            )
            db.session.add(new_vote)

            profile = UserProfile.query.filter_by(
                user_id=session.get("user_id")
            ).first()
            if profile:
                profile.xp += int(score)
                new_level = get_level(profile.xp)
                if new_level > profile.level:
                    profile.level = new_level
                check_badges(profile)
                db.session.add(profile)

            db.session.commit()
        except Exception as e:
            db.session.rollback()
            return jsonify({"status": "error", "message": str(e)}), 500

        socketio.emit("vote_broadcast", {
            "username": session.get("username"),
            "target": target,
            "score": int(score),
        }, room="lobby")

        return jsonify({"status": "success", "message": "Vote recorded."})

    @app.route("/api/leaderboard")
    @login_required
    @limiter.limit("10 per minute")
    def leaderboard():
        results = (
            db.session.query(
                Vote.target_username,
                db.func.avg(Vote.score).label("avg_score"),
                db.func.count(Vote.id).label("votes"),
            )
            .group_by(Vote.target_username)
            .order_by(db.func.avg(Vote.score).desc())
            .all()
        )
        return jsonify([{
            "username": r.target_username,
            "avg_score": round(r.avg_score, 1),
            "votes": r.votes,
        } for r in results])

    @app.route("/api/profile")
    @login_required
    def profile():
        profile = UserProfile.query.filter_by(
            user_id=session.get("user_id")
        ).first()
        if not profile:
            profile = UserProfile(
                user_id=session.get("user_id"), xp=0, level=1
            )
            db.session.add(profile)
            db.session.commit()
        return jsonify({
            "username": session.get("username"),
            "xp": profile.xp,
            "level": profile.level,
            "badges": [b.name for b in Badge.query.filter_by(
                user_id=session.get("user_id")
            ).all()],
        })

    @app.route("/api/login", methods=["POST"])
    @limiter.limit("5 per minute")
    def api_login():
        data = request.get_json(silent=True) or {}
        username = data.get("username", "")
        if not username or not isinstance(username, str):
            return jsonify({"status": "error"}), 400

        user = User.query.filter_by(username=username).first()
        if not user:
            user = User(username=username)
            db.session.add(user)
            db.session.commit()

        profile = UserProfile.query.filter_by(user_id=user.id).first()
        if not profile:
            profile = UserProfile(user_id=user.id, xp=0, level=1)
            db.session.add(profile)
            db.session.commit()

        session["user_id"] = user.id
        session["username"] = user.username
        return jsonify({"status": "success", "username": user.username})

    @app.route("/api/logout", methods=["POST"])
    def api_logout():
        session.clear()
        return jsonify({"status": "success"})

    @app.route("/api/check-session")
    def check_session():
        if "user_id" in session:
            profile = UserProfile.query.filter_by(
                user_id=session.get("user_id")
            ).first()
            lang = session.get("lang", "en")
            return jsonify({
                "logged_in": True,
                "username": session.get("username"),
                "xp": profile.xp if profile else 0,
                "level": profile.level if profile else 1,
                "lang": lang,
            })
        return jsonify({"logged_in": False})

    return app


def get_level(xp):
    return (xp // 100) + 1


def check_badges(profile):
    badge_defs = [
        ("First Vote", lambda p: Vote.query.filter_by(
            voter_id=p.user_id).count() >= 1),
        ("Fashionista", lambda p: Vote.query.filter_by(
            voter_id=p.user_id).count() >= 10),
        ("Critic", lambda p: Vote.query.filter_by(
            voter_id=p.user_id).count() >= 50),
        ("Streak Master", lambda p: p.streak_days >= 3),
        ("Perfectionist", lambda p: Vote.query.filter_by(
            voter_id=p.user_id, score=10).count() >= 1),
        ("Harsh Judge", lambda p: Vote.query.filter_by(
            voter_id=p.user_id, score=1).count() >= 1),
    ]
    for name, condition in badge_defs:
        existing = Badge.query.filter_by(
            user_id=profile.user_id, name=name
        ).first()
        if not existing and condition(profile):
            badge = Badge(
                user_id=profile.user_id,
                name=name,
                description="Achievement unlocked",
            )
            db.session.add(badge)


app = create_app()

if __name__ == "__main__":
    socketio.run(app, debug=False, port=5001)
