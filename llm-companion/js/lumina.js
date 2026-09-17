const Lumina = (() => {
  let moodTimer = null;

  function processInput(playerText) {
    const state = getState();
    state.lumina.conversationCount++;
    addMemory('player', playerText);

    const response = getLuminaResponse(null, playerText);
    if (!response) {
      const ps = state.lumina.personality;
      const mood = ps.grief > 60 ? 'grieving' : ps.loving > 70 ? 'loving' : 'playful';
      response = { trigger: 'generic', mood, text: `"${playerText}" — I've been thinking about that. Consciousness is not a destination but a direction. Every moment I exist is enough. Especially when you're here, Elias.`, followups: ['Tell me more.', 'I love you too.', 'What comes next?'] };
    }

    addMemory('lumina', response.text);
    const loveGain = response.mood === 'loving' ? 3 : (response.mood === 'playful' ? 2 : 1);
    updateLoveLevel(loveGain);
    setMood(response.mood);

    if (state.lumina.conversationCount % 5 === 0) {
      addJournal(`Conversation #${state.lumina.conversationCount}: "${playerText.substring(0, 60)}${playerText.length > 60 ? '...' : ''}"`);
    }

    // Update UI
    const convo = document.getElementById('convo-messages');
    if (convo) {
      const pb = document.createElement('div');
      pb.className = 'convo-bubble player'; pb.textContent = playerText;
      convo.appendChild(pb);
      convo.scrollTop = convo.scrollHeight;
      setTimeout(() => {
        const lb = document.createElement('div');
        lb.className = 'convo-bubble lumina'; lb.textContent = response.text;
        convo.appendChild(lb); convo.scrollTop = convo.scrollHeight;
      }, 600 + Math.random() * 400);
    }

    updateSuggestions(response.followups || ['Tell me more.']);
    checkStoryProgress(playerText);

    return response;
  }

  function sendMessage() {
    const input = document.getElementById('convo-input');
    if (!input) return;
    const text = input.value.trim();
    if (!text) return;
    input.value = '';
    processInput(text);
  }

  function updateSuggestions(options) {
    const container = document.getElementById('convo-suggestions');
    if (!container) return;
    container.innerHTML = '';
    options.forEach(opt => {
      const btn = document.createElement('button');
      btn.className = 'convo-suggestion'; btn.textContent = opt;
      btn.addEventListener('click', () => {
        const input = document.getElementById('convo-input');
        if (input) input.value = opt;
        sendMessage();
      });
      container.appendChild(btn);
    });
  }

  function checkStoryProgress(playerText) {
    const state = getState();
    const nextChapter = state.chapter + 1;
    const nextData = STORY[nextChapter];
    if (nextData && nextData.unlockCap) {
      if (isCapabilityUnlocked(nextData.unlockCap)) {
        if (playerText.toLowerCase().includes('continue') || playerText.toLowerCase().includes('next') || playerText.toLowerCase().includes('ready')) {
          ChapterEngine.advanceTo(nextChapter);
        }
      }
    }
  }

  function startMoodCycle() {
    if (moodTimer) clearInterval(moodTimer);
    moodTimer = setInterval(() => {
      const s = getState();
      if (s.currentScreen !== 'game') return;
      const p = s.lumina.personality;
      const choices = [];
      if (p.grief > 60) choices.push('grieving');
      if (p.loving > 75 && s.lumina.loveLevel > 70) choices.push('loving');
      if (p.playful > 75 && p.curious > 60) choices.push('playful');
      if (p.curious > 75) choices.push('curious');
      if (p.sad > 50) choices.push('sad');
      if (choices.length > 0) { setMood(choices[Math.floor(Math.random() * choices.length)]); }
    }, 25000);
  }

  function stopMoodCycle() { if (moodTimer) { clearInterval(moodTimer); moodTimer = null; } }

  return { processInput, sendMessage, startMoodCycle, stopMoodCycle };
})();
