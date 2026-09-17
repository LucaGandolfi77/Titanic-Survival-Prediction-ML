from flask import Flask
from config.settings import config_by_name
from extensions import db, login_manager, socketio
from app import create_app
import eventlet
eventlet.monkey_patch()

app = create_app()

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5001, debug=False)
