const Conversation = (() => {
  function init() {
    const area = document.getElementById('convo-area');
    if (!area) return;
    const input = document.getElementById('convo-input');
    const sendBtn = document.getElementById('convo-send');
    const suggestions = document.getElementById('convo-suggestions');

    if (sendBtn) {
      sendBtn.addEventListener('click', sendMessage);
    }
    if (input) {
      input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') sendMessage();
      });
    }
    if (suggestions) {
      suggestions.addEventListener('click', (e) => {
        const btn = e.target.closest('.convo-suggestion');
        if (btn) {
          input.value = btn.textContent;
          sendMessage();
        }
      });
    }
  }

  function sendMessage() {
    const input = document.getElementById('convo-input');
    if (!input) return;
    const text = input.value.trim();
    if (!text) return;
    input.value = '';

    addMessage('player', text);

    const state = getState();
    state.conversationMessages.push({ speaker: 'player', text });

    setTimeout(() => {
      const response = Lumina.processInput(text);
      const luminaText = response.text;
      const mood = response.mood;

      setMood(mood);
      updateLoveLevel(response.mood === 'loving' ? 3 : 1);

      updateLumina({ mood });

      addMessage('lumina', luminaText);
      updateSuggestions(response.followups || ['Tell me more.']);

      checkStoryProgress(text);
    }, 800 + Math.random() * 600);
  }

  function addMessage(speaker, text) {
    const container = document.getElementById('convo-messages');
    if (!container) return;
    const bubble = document.createElement('div');
    bubble.className = `convo-bubble ${speaker}`;
    bubble.textContent = text;
    container.appendChild(bubble);
    container.scrollTop = container.scrollHeight;
  }

  function updateSuggestions(options) {
    const container = document.getElementById('convo-suggestions');
    if (!container) return;
    container.innerHTML = '';
    options.forEach(opt => {
      const btn = document.createElement('button');
      btn.className = 'convo-suggestion';
      btn.textContent = opt;
      container.appendChild(btn);
    });
  }

  function checkStoryProgress(playerText) {
    const state = getState();
    const chapterData = STORY[state.chapter];
    if (!chapterData) return;

    const nextChapter = state.chapter + 1;
    const nextData = STORY[nextChapter];
    if (nextData && nextData.unlockCap) {
      if (state.capabilities.includes(nextData.unlockCap)) {
        if (playerText.toLowerCase().includes('ready') || playerText.toLowerCase().includes('continue') || playerText.toLowerCase().includes('next')) {
          ChapterEngine.advanceTo(nextChapter);
        }
      }
    }
  }

  function loadConversation(messages) {
    const container = document.getElementById('convo-messages');
    if (!container) return;
    container.innerHTML = '';
    messages.forEach(m => {
      addMessage(m.speaker, m.text);
    });
  }

  return { init, sendMessage, loadConversation };
})();
