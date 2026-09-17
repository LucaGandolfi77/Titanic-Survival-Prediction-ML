let currentTarget = null;
let nextTimerId = null;
let socket = null;

function getApiBase() {
    if (window.location.hostname === 'localhost' && window.location.port !== '5001') {
        return `http://${window.location.hostname}:5001`;
    }
    return '';
}

async function fetchApi(path, options = {}) {
    const baseUrl = getApiBase();
    const res = await fetch(baseUrl + path, options);
    if (!res.ok) {
        throw new Error(`API ${res.status}: ${res.statusText}`);
    }
    return res.json();
}

function initSocket() {
    socket = io({
        path: '/socket.io',
        transports: ['websocket', 'polling'],
    });

    socket.on('connect', () => {
        console.log('Socket connected');
        const online = document.getElementById('online-count');
        if (online) online.textContent = '🟢 Online';
    });

    socket.on('disconnect', () => {
        const online = document.getElementById('online-count');
        if (online) online.textContent = '🔴 Offline';
    });

    socket.on('user_connected', (data) => {
        addSystemMessage(data.message);
        const online = document.getElementById('online-count');
        if (online) online.textContent = '🟢 Connected';
    });

    socket.on('user_disconnected', (data) => {
        addSystemMessage(data.message);
    });

    socket.on('chat_message', (data) => {
        addChatMessage(data.username, data.message);
    });

    socket.on('vote_broadcast', (data) => {
        showVoteFeedback(`${data.username} voted ${data.score}/10 for ${data.target}`, '#a0a0b5');
    });

    socket.on('connect_error', (err) => {
        console.warn('Socket connection error:', err.message);
    });
}

function addSystemMessage(text) {
    const box = document.getElementById('chat-box');
    if (!box) return;
    const msgDiv = document.createElement('div');
    msgDiv.className = 'msg';
    msgDiv.style.fontStyle = 'italic';
    msgDiv.style.color = '#a0a0b5';
    msgDiv.textContent = text;
    box.appendChild(msgDiv);
    box.scrollTop = box.scrollHeight;
}

function addChatMessage(username, message) {
    const box = document.getElementById('chat-box');
    if (!box) return;
    const msgDiv = document.createElement('div');
    msgDiv.className = 'msg';
    const userSpan = document.createElement('b');
    userSpan.textContent = username + ': ';
    msgDiv.appendChild(userSpan);
    msgDiv.appendChild(document.createTextNode(message));
    box.appendChild(msgDiv);
    box.scrollTop = box.scrollHeight;
}

function showVoteFeedback(text, color) {
    const feedback = document.getElementById('vote-feedback');
    if (feedback) {
        feedback.innerText = text;
        feedback.style.color = color;
    }
}

async function nextModel() {
    if (nextTimerId) {
        clearTimeout(nextTimerId);
        nextTimerId = null;
    }

    try {
        const data = await fetchApi('/api/strut');
        if (data.username) {
            currentTarget = data.username;
            document.getElementById('target-model').innerText = data.username;
            document.getElementById('target-outfit').innerText = data.outfit;
            showVoteFeedback('', '');

            if (window.playerMesh) window.playerMesh.rotation.y = 0;
            if (window.scene && window.scene.background && window.scene.background.set) {
                window.scene.background.set(0xffb6c1);
            }
        } else {
            document.getElementById('target-model').innerText = 'Ghost Town';
        }
    } catch (err) {
        console.error('Failed to load model:', err);
        document.getElementById('target-model').innerText = 'Connection lost';
        document.getElementById('target-outfit').innerText = 'Retrying...';
    }
}

async function submitVote() {
    if (!currentTarget) return;
    const slider = document.getElementById('score-slider');
    const score = parseInt(slider.value, 10);

    if (isNaN(score) || score < 1 || score > 10) return;

    try {
        await fetchApi('/api/vote', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({target_username: currentTarget, score}),
        });
    } catch (err) {
        console.warn('Vote failed, queuing for sync:', err);
        if (window.queueVote) {
            await window.queueVote(currentTarget, score);
        }
        showVoteFeedback('Queued — will sync when online', '#ffdd59');
        return;
    }

    if (socket) {
        socket.emit('vote_broadcast', {target: currentTarget, score});
    }

    const feedback = document.getElementById('vote-feedback');
    if (score >= 8) {
        showVoteFeedback('🔥 ABSOLUTE FIRE!', '#ffdd59');
        if (window.scene && window.scene.background && window.scene.background.set) {
            window.scene.background.set(0xffd700);
        }
        if (window.playerMesh && window.playerMesh.scale && window.playerMesh.scale.set) {
            window.playerMesh.scale.set(1.2, 1.2, 1.2);
        }
        if ('vibrate' in navigator) navigator.vibrate(50);
    } else if (score <= 3) {
        showVoteFeedback('🗑️ TRASH!', '#1a1a2e');
        if (window.scene && window.scene.background && window.scene.background.set) {
            window.scene.background.set(0x333333);
        }
        if (window.playerMesh && window.playerMesh.scale && window.playerMesh.scale.set) {
            window.playerMesh.scale.set(0.8, 0.8, 0.8);
        }
        if ('vibrate' in navigator) navigator.vibrate(200);
    } else {
        showVoteFeedback('Meh. Play it safe.', '#fff');
        if (window.playerMesh && window.playerMesh.scale && window.playerMesh.scale.set) {
            window.playerMesh.scale.set(1, 1, 1);
        }
    }

    nextTimerId = setTimeout(nextModel, 2000);
}

function sendMessage() {
    const input = document.getElementById('chat-input');
    const msg = input.value.trim();
    if (msg && msg.length <= 200) {
        if (socket) {
            socket.emit('chat_message', {message: msg});
        }
        addChatMessage('You', msg);
        input.value = '';
        const box = document.getElementById('chat-box');
        if (box) box.scrollTop = box.scrollHeight;
        if ('vibrate' in navigator) navigator.vibrate(10);
    }
}

function updateProfile(data) {
    const xpEl = document.getElementById('user-xp');
    const levelEl = document.getElementById('user-level');
    const xpFill = document.getElementById('xp-fill');
    const badgesEl = document.getElementById('badges-container');

    if (xpEl) xpEl.textContent = `XP: ${data.xp || 0}`;
    if (levelEl) levelEl.textContent = `Lv.${data.level || 1}`;

    const nextLevelXP = data.level * 100;
    const prevLevelXP = (data.level - 1) * 100;
    const progress = data.xp ? Math.min(100, ((data.xp - prevLevelXP) / (nextLevelXP - prevLevelXP)) * 100) : 0;
    if (xpFill) xpFill.style.width = `${progress}%`;

    if (badgesEl && data.badges) {
        badgesEl.innerHTML = '';
        data.badges.forEach(badge => {
            const span = document.createElement('span');
            span.className = 'badge';
            span.textContent = badge;
            badgesEl.appendChild(span);
        });
    }
}

async function loadProfile() {
    try {
        const data = await fetchApi('/api/profile');
        updateProfile(data);
    } catch (e) {
        console.warn('Failed to load profile:', e);
    }
}

if (typeof window.scene === 'undefined') {
    window.scene = {background: {set: function() {}}};
}
if (typeof window.playerMesh === 'undefined') {
    window.playerMesh = {rotation: {y: 0}, scale: {set: function() {}}};
}

document.addEventListener('DOMContentLoaded', () => {
    const slayBtn = document.getElementById('btn-slay');
    const skipBtn = document.getElementById('btn-skip');
    const sendBtn = document.getElementById('btn-send');
    const chatInput = document.getElementById('chat-input');
    const slider = document.getElementById('score-slider');
    const display = document.getElementById('score-display');
    const lb = document.getElementById('leaderboard-list');

    initSocket();
    loadProfile();

    if (slayBtn) slayBtn.addEventListener('click', submitVote);
    if (skipBtn) skipBtn.addEventListener('click', () => { nextTimerId = setTimeout(nextModel, 2000); });
    if (sendBtn) sendBtn.addEventListener('click', sendMessage);
    if (chatInput) {
        chatInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') sendMessage();
        });
    }
    if (slider && display) {
        slider.addEventListener('input', () => { display.innerText = slider.value; });
    }

    if (lb) {
        fetchApi('/api/leaderboard').then(data => {
            if (Array.isArray(data) && data.length) {
                data.forEach(entry => {
                    const li = document.createElement('li');
                    li.textContent = `${entry.username} — ${entry.avg_score}/10 (${entry.votes} votes)`;
                    lb.appendChild(li);
                });
            }
        }).catch(() => {});
    }

    nextTimerId = setTimeout(nextModel, 100);

    setInterval(loadProfile, 30000);
});

window.sendMessage = sendMessage;
window.submitVote = submitVote;
window.nextModel = nextModel;
