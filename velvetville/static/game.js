let currentTarget = null;

async function fetchApi(path, options = {}) {
    const baseUrl = window.location.port === '5500' ? 'http://127.0.0.1:5001' : '';
    const res = await fetch(baseUrl + path, options);
    return res.json();
}

async function nextModel() {
    const data = await fetchApi('/api/strut');
    
    if (data.username) {
        currentTarget = data.username;
        document.getElementById('target-model').innerText = data.username;
        document.getElementById('target-outfit').innerText = data.outfit;
        document.getElementById('vote-feedback').innerText = "";
        
        // Visual flair: reset player rotation and change background slightly
        playerMesh.rotation.y = 0; 
        scene.background = new THREE.Color(0xffb6c1);
    } else {
        document.getElementById('target-model').innerText = "Ghost Town";
    }
}

async function submitVote() {
    if (!currentTarget) return;
    const score = parseInt(document.getElementById('score-slider').value);

    const data = await fetchApi('/api/vote', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ target_username: currentTarget, score: score })
    });

    // Visual and text feedback based on how harsh the judge is
    const feedback = document.getElementById('vote-feedback');
    if (score >= 8) {
        feedback.innerText = "🔥 ABSOLUTE FIRE!";
        feedback.style.color = "#ffdd59";
        scene.background = new THREE.Color(0xffd700); // Gold flash in 3D scene!
        playerMesh.scale.set(1.2, 1.2, 1.2); // Player swells with pride
    } else if (score <= 3) {
        feedback.innerText = "🗑️ TRASH!";
        feedback.style.color = "#1a1a2e";
        scene.background = new THREE.Color(0x333333); // Dark gloomy flash
        playerMesh.scale.set(0.8, 0.8, 0.8); // Player shrinks in shame
    } else {
        feedback.innerText = "Meh. Play it safe.";
        feedback.style.color = "#fff";
        playerMesh.scale.set(1, 1, 1);
    }

    // Automatically load the next victim after 2 seconds
    setTimeout(nextModel, 2000);
}

// Mock Three.js init just in case it's missing from your setup
if (typeof window.scene === 'undefined') {
    window.scene = { background: { set: function(){} } };
}
if (typeof window.playerMesh === 'undefined') {
    window.playerMesh = { rotation: { y: 0 }, scale: { set: function(){} } };
}

// Call this once at the bottom of your script to load the first player
setTimeout(() => {
    nextModel();
}, 100);

function sendMessage() {
    const input = document.getElementById('chat-input');
    const box = document.getElementById('chat-box');
    const msg = input.value.trim();
    if (msg) {
        const msgDiv = document.createElement('div');
        msgDiv.className = 'msg';
        msgDiv.innerText = "You: " + msg;
        box.appendChild(msgDiv);
        input.value = '';
        box.scrollTop = box.scrollHeight;
    }
}
window.sendMessage = sendMessage;
