from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import random

app = Flask(__name__)
CORS(app)

models = [
    {"username": "GlowUpQueen", "outfit": "Sequin Gown & Boa"},
    {"username": "VaporwaveVixen", "outfit": "Neon Trenchcoat"},
    {"username": "BasicBob", "outfit": "Cargo Shorts & Flip Flops"},
    {"username": "VelvetVamp", "outfit": "Crushed Velvet Suit"}
]

@app.route('/')
def index():
    return render_template('login.html')

@app.route('/game')
def game():
    event_data = {
        "name": "Summer Fashion Week",
        "multiplier": 2
    }
    return render_template('game.html', event=event_data)

@app.route('/api/strut')
def strut():
    model = random.choice(models)
    return jsonify(model)

@app.route('/api/vote', methods=['POST'])
def vote():
    data = request.json
    target = data.get('target_username')
    score = data.get('score')
    print(f"Voted {score}/10 for {target}")
    return jsonify({"status": "success", "message": "Vote recorded."})

if __name__ == '__main__':
    app.run(debug=True, port=5001)
