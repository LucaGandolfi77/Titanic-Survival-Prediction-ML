import pytest
from app import create_app
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


def test_db_created(client):
    c, _ = client
    with c.session_transaction():
        user = User.query.first()
        assert user is not None
        assert user.username == "testuser"


def test_user_not_logged_in(client):
    c, _ = client
    resp = c.get("/api/check-session")
    data = resp.get_json()
    assert data["logged_in"] is False


def test_profile_created_on_login(client):
    c, uid = client
    with c.session_transaction() as sess:
        sess["_user_id"] = str(uid)
        sess["user_id"] = uid
    resp = c.get("/api/profile")
    data = resp.get_json()
    assert data is not None
    assert data["level"] == 1
