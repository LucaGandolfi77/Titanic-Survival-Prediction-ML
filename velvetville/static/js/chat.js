document.addEventListener('DOMContentLoaded', () => {
    const chatBox = document.getElementById('chat-box');
    if (!chatBox) return;

    const BOT_MESSAGES = [
        "Sooo stunning! 😍",
        "Basic. Next!",
        "That's FIRE! 🔥",
        "I can't with this energy",
        "VelvetVamp slayed that one 💅",
        "Cargo shorts in 2024? Bold choice.",
        "Where can I get that outfit?",
        "The runway is BEGGING for more sparkle ✨",
        "This is giving ✨ chic ✨",
        "Slay! 🙌",
    ];

    let botInterval = null;

    function addBotMessage() {
        const msg = BOT_MESSAGES[Math.floor(Math.random() * BOT_MESSAGES.length)];
        addChatMessage('Bot', msg);
    }

    function addChatMessage(username, message) {
        const msgDiv = document.createElement('div');
        msgDiv.className = 'msg';
        const userSpan = document.createElement('b');
        userSpan.textContent = username + ': ';
        msgDiv.appendChild(userSpan);
        msgDiv.appendChild(document.createTextNode(message));
        chatBox.appendChild(msgDiv);
        chatBox.scrollTop = chatBox.scrollHeight;
    }

    function startBot() {
        if (botInterval) clearInterval(botInterval);
        botInterval = setInterval(addBotMessage, 8000 + Math.random() * 7000);
        addBotMessage();
    }

    setTimeout(startBot, 3000);

    window.addEventListener('beforeunload', () => {
        if (botInterval) clearInterval(botInterval);
    });
});
