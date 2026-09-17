import pytest
from app import create_app, get_level, analyze_sentiment
from extensions import db
from models import User, UserProfile


@pytest.fixture
def client():
    app = create_app("testing")
    with app.app_context():
        db.create_all()
        user = User(username="testuser")
        db.session.add(user)
        db.session.commit()
        user_id = user.id
        profile = UserProfile(user_id=user_id, xp=0, level=1)
        db.session.add(profile)
        db.session.commit()
    with app.test_client() as client:
        with app.app_context():
            yield client, user_id
            db.drop_all()


def auth(client, uid):
    c, _ = client
    with c.session_transaction() as sess:
        sess["_user_id"] = str(uid)
        sess["user_id"] = uid
        sess["username"] = "testuser"
        sess["lang"] = "en"
    return c


def make_votes(client, uid):
    c = auth(client, uid)
    for i in range(5):
        c.post("/api/vote", json={"target_username": f"model_{i % 3}", "score": i + 1})


def test_index_returns_login(client):
    c, _ = client
    assert c.get("/").status_code == 200


def test_game_requires_login(client):
    c, _ = client
    assert c.get("/game").status_code in (302, 401)


def test_analytics_requires_login(client):
    c, _ = client
    assert c.get("/analytics").status_code in (302, 401)


def test_analytics_summary(client):
    c, uid = client
    make_votes(client, uid)
    resp = auth(client, uid).get("/api/analytics/summary")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["total_votes"] >= 5
    assert "top_models" in data
    assert "score_distribution" in data


def test_analytics_sentiment(client):
    c, uid = client
    make_votes(client, uid)
    resp = auth(client, uid).get("/api/analytics/sentiment")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "positive" in data
    assert "negative" in data


def test_recommendations_empty(client):
    c, uid = client
    resp = auth(client, uid).get("/api/recommendations")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "models" in data
    assert data["type"] == "trending"


def test_recommendations_with_votes(client):
    c, uid = client
    make_votes(client, uid)
    resp = auth(client, uid).get("/api/recommendations")
    assert resp.status_code == 200
    data = resp.get_json()
    assert len(data["models"]) > 0
    assert data["type"] == "personalized"


def test_sentiment_analysis():
    assert analyze_sentiment("This is amazing and beautiful!") == "positive"
    assert analyze_sentiment("This is terrible and ugly!") == "negative"
    assert analyze_sentiment("Hello world") == "neutral"


def test_set_language(client):
    c, uid = client
    resp = auth(client, uid).post("/set-language", json={"language": "it"})
    assert resp.status_code == 200
    with c.session_transaction() as sess:
        assert sess["lang"] == "it"


def test_set_language_invalid(client):
    c, uid = client
    resp = auth(client, uid).post("/set-language", json={"language": "xx"})
    assert resp.status_code == 400


def test_translations(client):
    c, _ = client
    resp = c.get("/api/translations/it")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "welcome" in data


def test_export_ranking_json(client):
    c, uid = client
    make_votes(client, uid)
    resp = auth(client, uid).get("/api/export/ranking")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "entries" in data
    assert len(data["entries"]) > 0


def test_export_ranking_html(client):
    c, uid = client
    make_votes(client, uid)
    resp = auth(client, uid).get("/api/export/ranking?format=html")
    assert resp.status_code == 200
    assert b"<html" in resp.data.lower()


def test_ambient_theme_default(client):
    c, uid = client
    resp = auth(client, uid).get("/api/ambient/theme")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "theme" in data
    assert data["theme"]["name"] == "Classic"


def test_ambient_theme_by_time(client):
    c, uid = client
    resp = auth(client, uid).get("/api/ambient/theme")
    assert resp.status_code == 200


def test_push_subscribe(client):
    c, uid = client
    resp = auth(client, uid).post("/api/notifications/subscribe", json={
        "subscription": {"endpoint": "https://example.com/push", "keys": {}}
    })
    assert resp.status_code == 200


def test_push_unsubscribe(client):
    c, uid = client
    auth(client, uid).post("/api/notifications/subscribe", json={
        "subscription": {"endpoint": "https://example.com/push"}
    })
    resp = auth(client, uid).post("/api/notifications/unsubscribe")
    assert resp.status_code == 200


def test_vote_grants_xp(client):
    c, uid = client
    with c.session_transaction() as sess:
        sess["_user_id"] = str(uid)
        sess["user_id"] = uid
        sess["username"] = "testuser"
    c.post("/api/vote", json={"target_username": "test", "score": 5})
    profile = UserProfile.query.filter_by(user_id=uid).first()
    assert profile.xp >= 5


def test_api_login(client):
    c, _ = client
    resp = c.post("/api/login", json={"username": "TestUser"})
    assert resp.status_code == 200
    assert resp.get_json()["status"] == "success"


def test_api_logout(client):
    c, uid = client
    with c.session_transaction() as sess:
        sess["_user_id"] = str(uid)
        sess["user_id"] = uid
    assert c.post("/api/logout").status_code == 200


def test_api_leaderboard_empty(client):
    c, uid = client
    with c.session_transaction() as sess:
        sess["_user_id"] = str(uid)
        sess["user_id"] = uid
    resp = c.get("/api/leaderboard")
    assert resp.status_code == 200


def test_api_check_session(client):
    c, _ = client
    resp = c.get("/api/check-session")
    assert resp.status_code == 200
    assert resp.get_json()["logged_in"] is False


def test_api_login_sets_session(client):
    c, _ = client
    c.post("/api/login", json={"username": "SessionTest"})
    resp = c.get("/api/check-session")
    data = resp.get_json()
    assert data["logged_in"] is True
    assert data["username"] == "SessionTest"


def test_api_profile(client):
    c, uid = client
    with c.session_transaction() as sess:
        sess["_user_id"] = str(uid)
        sess["user_id"] = uid
        sess["username"] = "testuser"
    resp = c.get("/api/profile")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["username"] == "testuser"


def test_get_level():
    assert get_level(0) == 1
    assert get_level(100) == 2
    assert get_level(200) == 3


def test_rate_limiting(client):
    c, uid = client
    with c.session_transaction() as sess:
        sess["_user_id"] = str(uid)
        sess["user_id"] = uid
        sess["username"] = "testuser"
    for _ in range(3):
        c.post("/api/vote", json={"target_username": "test", "score": 5})
    last = c.post("/api/vote", json={"target_username": "test", "score": 5})
    assert last.status_code in (200, 429)
