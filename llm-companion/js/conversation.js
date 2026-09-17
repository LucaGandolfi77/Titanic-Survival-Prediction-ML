const Conversation = (() => {
  function init() {
    const sendBtn = document.getElementById('convo-send');
    const input = document.getElementById('convo-input');
    if (sendBtn) sendBtn.addEventListener('click', () => Lumina.sendMessage());
    if (input) input.addEventListener('keydown', (e) => { if (e.key === 'Enter') Lumina.sendMessage(); });
  }

  function loadConversation(messages) {
    const container = document.getElementById('convo-messages');
    if (!container) return;
    container.innerHTML = '';
    messages.forEach(m => {
      const bubble = document.createElement('div');
      bubble.className = `convo-bubble ${m.speaker}`; bubble.textContent = m.text;
      container.appendChild(bubble);
    });
    container.scrollTop = container.scrollHeight;
  }

  return { init, loadConversation };
})();
