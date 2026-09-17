const ChapterEngine = (() => {
  let transitions = {};

  function init() {
    transitions = {
      '1:start': { target: 2, condition: () => isCapabilityUnlocked('language'), buttonText: 'Continue to Chapter 2 →' },
      '2:start': { target: 3, condition: () => isCapabilityUnlocked('reasoning'), buttonText: 'Continue to Chapter 3 →' },
      '3:start': { target: 4, condition: () => isCapabilityUnlocked('creativity'), buttonText: 'Continue to Chapter 4 →' },
      '4:start': { target: 5, condition: () => isCapabilityUnlocked('awareness'), buttonText: 'Continue to Chapter 5 →' },
      '5:start': { target: 6, condition: () => isCapabilityUnlocked('consciousness'), buttonText: 'Continue to Chapter 6 →' },
    };
  }

  function renderChapter(chapter) {
    const data = STORY[chapter];
    if (!data) return;

    const titleEl = document.getElementById('chapter-title');
    if (titleEl) titleEl.textContent = `Chapter ${chapter}: ${data.title}`;

    if (data.screen === 'cabin') {
      const luminaPanel = document.getElementById('lumina-panel');
      if (luminaPanel) {
        luminaPanel.classList.remove('hidden');
        const body = luminaPanel.querySelector('.lumina-body');
        if (body) {
          body.innerHTML = '';
          const chapterStart = document.createElement('div');
          chapterStart.className = 'lumina-message';
          const introText = chapter === 1 ? data.cabinIntro : `${data.title} — ${getFirstLuminaLine(chapter)}`;
          chapterStart.textContent = introText;
          chapterStart.style.fontStyle = 'normal';
          chapterStart.style.fontFamily = "'Cormorant Garamond', serif";
          chapterStart.style.fontSize = '1.05rem';
          chapterStart.style.color = 'var(--accent-gold)';
          chapterStart.style.marginBottom = '16px';
          chapterStart.style.borderLeft = '3px solid var(--accent-warm)';
          chapterStart.style.padding = '14px 18px';
          chapterStart.style.background = 'rgba(233, 69, 96, 0.04)';
          chapterStart.style.borderRadius = 'var(--radius-md)';
          chapterStart.style.animation = 'fadeIn 0.5s ease';
          body.appendChild(chapterStart);
        }
      }
    }

    // Update follow-up buttons
    setTimeout(() => updateFollowups('start'), 300);
  }

  function getFirstLuminaLine(chapter) {
    const data = STORY[chapter];
    if (data && data.lumina && data.lumina[0]) {
      return data.lumina[0].text;
    }
    return 'I\'m here, thinking...';
  }

  function updateFollowups(triggerKey) {
    const state = getState();
    const chapterData = STORY[state.chapter];
    if (!chapterData || !chapterData.lumina) return;

    const match = chapterData.lumina.find(m => m.trigger === triggerKey);
    if (!match) return;

    const suggestions = document.getElementById('convo-suggestions');
    if (!suggestions) return;
    suggestions.innerHTML = '';
    match.followups.forEach(opt => {
      const btn = document.createElement('button');
      btn.className = 'convo-suggestion';
      btn.textContent = opt;
      suggestions.appendChild(btn);
    });
  }

  function advanceTo(chapter) {
    if (chapter > gameState.chapter) {
      setState({ chapter });
      addJournal(`Reached Chapter ${chapter}: ${STORY[chapter]?.title || 'Unknown'}`);
      renderChapter(chapter);
      updateLumina({ mood: 'loving' });
      updateLoveLevel(5);
    }
  }

  function renderTransitionButton(chapter) {
    const state = getState();
    const key = `${chapter}:start`;
    const trans = transitions[key];
    if (!trans) return null;

    const conditionMet = trans.condition();
    const container = document.getElementById('chapter-transition');
    if (!container) return;

    if (conditionMet) {
      container.innerHTML = `<button class="btn-primary" id="btn-advance-chapter" onclick="ChapterEngine.advanceTo(${trans.target})" style="margin-top:16px">${trans.buttonText}</button>`;
    } else {
      const chapterData = STORY[chapter];
      const capName = chapterData?.unlockCap ? CAPABILITIES.find(c => c.id === chapterData.unlockCap)?.name || '' : '';
      container.innerHTML = `<div style="margin-top:16px; padding:12px 20px; background:rgba(255,212,111,0.05); border-radius:var(--radius-md); font-size:0.9rem; color:var(--text-secondary);">Continue training to unlock ${capName} →</div>`;
    }
  }

  function getTransitions() { return transitions; }

  return { init, renderChapter, advanceTo, updateFollowups, renderTransitionButton, getTransitions };
})();
