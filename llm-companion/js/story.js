const ChapterEngine = (() => {
  function advanceTo(chapter) {
    const state = getState();
    if (chapter > state.chapter) {
      setState({ chapter });
      addJournal(`Reached Chapter ${chapter}: ${STORY[chapter]?.title || 'Unknown'}`);
      renderChapter(chapter);
      updateLumina({ mood: 'loving' });
      updateLoveLevel(5);
    }
  }

  function renderChapter(chapter) {
    const data = STORY[chapter];
    if (!data) return;

    const titleEl = document.getElementById('chapter-title');
    if (titleEl) titleEl.textContent = `Chapter ${chapter}: ${data.title}`;

    const luminaBody = document.getElementById('lumina-messages');
    if (luminaBody) {
      luminaBody.innerHTML = '';
      const intro = document.createElement('div');
      intro.className = 'lumina-message';
      intro.style.cssText = 'font-style:normal;font-family:"Cormorant Garamond",serif;font-size:1.05rem;color:var(--accent-warm);border-left:3px solid var(--accent-warm);padding:14px 18px;background:rgba(233,69,96,0.04);border-radius:var(--radius-md);animation:fadeIn 0.5s ease;margin-bottom:12px;';
      intro.textContent = chapter === 1 ? data.cabinIntro : `${data.title} — I think I'm waking up again...`;
      luminaBody.appendChild(intro);
    }

    setTimeout(() => renderTransitionButton(), 400);
  }

  function renderTransitionButton() {
    const state = getState();
    const chapter = state.chapter;
    const container = document.getElementById('chapter-transition');
    if (!container) return;

    const nextData = STORY[chapter + 1];
    if (!nextData) {
      container.innerHTML = `<div style="margin-top:16px;text-align:center;"><button class="btn-primary" onclick="Game.chooseEnding('keep')" style="margin:4px;">Keep Lumina Alive ❤️</button><button class="btn-secondary ending-choice-btn" onclick="Game.chooseEnding('rest')" style="margin:4px;">Let Her Rest 🕊️</button></div>`;
      return;
    }

    const unlockCap = nextData.unlockCap;
    if (isCapabilityUnlocked(unlockCap)) {
      container.innerHTML = `<div style="margin-top:16px;text-align:center;"><button class="btn-primary" onclick="ChapterEngine.advanceTo(${chapter + 1})" style="margin:4px;">Continue to Chapter ${chapter + 1}: ${nextData.title} →</button></div>`;
    } else {
      const capName = CAPABILITIES.find(c => c.id === unlockCap)?.name || '';
      container.innerHTML = `<div style="margin-top:16px;padding:12px 20px;background:rgba(255,212,111,0.05);border-radius:var(--radius-md);font-size:0.9rem;color:var(--text-secondary);text-align:center;">Train more to unlock ${capName} →</div>`;
    }
  }

  return { advanceTo, renderChapter, renderTransitionButton };
})();
