import sys
import os

# ── PythonAnywhere: replace "yourusername" with your PA username ──
path = "/home/yourusername/tamagotchi_social"
if path not in sys.path:
    sys.path.insert(0, path)

from app import app as application

with application.app_context():
    from app import db
    db.create_all()
