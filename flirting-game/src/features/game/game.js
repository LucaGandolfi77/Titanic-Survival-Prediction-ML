// Game engine: renders scenes from the dialog graph, drives the reaction
// timer, handles choices/timeouts, swipe gestures and the choice log.
// The ending module is dynamically imported only when the game finishes.

import { els } from '../../ui/dom.js';
import { state, setState, resetRound } from '../../core/state.js';
import { TimerEngine } from '../../core/timer.js';
import { haptic, patterns } from '../../core/haptics.js';
import { showScreen } from '../../ui/router.js';
import { showToast } from '../../ui/toast.js';
import {
  loadDialogues,
  pickCharacter,
  applyMeta,
  firstScene,
  sceneById,
  deriveTone,
  willUnlockSecret,
  nextSceneAfter,
  reactionFor
} from '../../data/dialogues.js';

const RESPONSE_DELAY = 950;

let dialogues = null;
let timer = null;
let advancePending = false;
let timeLeft = 0; // live countdown value, updated by the timer engine

function ensureTimer() {
  if (!timer) {
    timer = new TimerEngine({
      onTick: (secondsLeft, fraction) => {
        timeLeft = secondsLeft;
        updateTimerUI(secondsLeft, fraction);
      },
      onFinish: () => handleTimeout()
    });
  }
  return timer;
}

export function initGame() {
  // Event delegation: one listener for all choice buttons.
  els.choices.addEventListener('click', (event) => {
    const btn = event.target.closest('button[data-index]');
    if (!btn || btn.disabled) return;
    handleChoice(Number(btn.dataset.index));
  });

  initGestures();

  els.historyClose.addEventListener('click', closeHistory);
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape') closeHistory();
  });
}

export async function startGame() {
  ensureTimer();
  timer.cancel();
  timer.cancelDelay();
  advancePending = false;

  if (!dialogues) {
    try {
      dialogues = await loadDialogues();
    } catch (err) {
      console.error('Dialogues failed to load', err);
      showToast('Could not load the story. Reconnect and try again.', { duration: 4000 });
      return;
    }
  }

  applyMeta(dialogues, state);
  const character = pickCharacter(dialogues, state.interestGender);
  if (!character) {
    showToast('No character found for this setup.', { duration: 4000 });
    return;
  }

  resetRound();
  setState({ character, scene: firstScene(character) });

  setAvatar(state.character);
  updateStats();
  showScreen('game');
  renderScene();
}

export function stopGame() {
  if (timer) {
    timer.cancel();
    timer.cancelDelay();
  }
  advancePending = false;
  timeLeft = 0;
  setState({ awaitingChoice: false, responsePhase: false });
  closeHistory();
}

/* ---------- rendering ---------- */

function renderScene() {
  const scene = state.scene;
  if (!scene) {
    finishGame();
    return;
  }

  els.chapterLabel.textContent = `Chapter ${scene.chapter}`;
  els.characterName.textContent = state.character.name;
  els.characterMeta.textContent = state.character.tagline;
  els.dialogueText.textContent = scene.text;

  els.choices.innerHTML = '';
  scene.choices.forEach((choice, index) => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = `choice-btn ${deriveTone(choice.score)}`;
    btn.textContent = choice.text;
    btn.dataset.index = String(index);
    els.choices.appendChild(btn);
  });

  setState({ awaitingChoice: true, responsePhase: false });
  startTimer();
}

function setAvatar(character) {
  els.avatar.textContent = character.name.charAt(0).toUpperCase();
  els.avatar.style.background = `linear-gradient(135deg, ${character.palette[0]}, ${character.palette[1]})`;
  els.avatar.style.boxShadow = `0 18px 40px ${character.palette[0]}55`;
}

function updateStats() {
  els.scoreValue.textContent = state.score;
  els.streakValue.textContent = state.streak;
  els.secretValue.textContent = state.secrets;
}

function updateTimerUI(secondsLeft, fraction) {
  const pct = Math.max(0, Math.min(100, fraction * 100));
  els.timerText.textContent = `${secondsLeft.toFixed(1)}s`;
  els.timerFill.style.width = `${pct}%`;

  if (pct > 60) {
    els.timerFill.style.background = 'linear-gradient(90deg, #6ef3c5, #5aa9ff)';
  } else if (pct > 30) {
    els.timerFill.style.background = 'linear-gradient(90deg, #ffd166, #ff9f43)';
  } else {
    els.timerFill.style.background = 'linear-gradient(90deg, #ff6b6b, #ff3b8d)';
  }
}

function startTimer() {
  timer.start(state.timeLimit);
  timeLeft = state.timeLimit;
  updateTimerUI(state.timeLimit, 1);
}

function disableChoices() {
  els.choices.querySelectorAll('button').forEach((btn) => {
    btn.disabled = true;
  });
}

/* ---------- game flow ---------- */

function handleChoice(index) {
  if (!state.awaitingChoice) return;
  const scene = state.scene;
  const choice = scene?.choices?.[index];
  if (!choice) return;

  const fast = timeLeft / state.timeLimit >= state.fastThreshold && choice.score > 0;

  // Stop the countdown: onFinish can no longer fire after this point.
  timer.cancel();
  setState({ awaitingChoice: false, responsePhase: true });

  state.score += choice.score;

  if (fast) {
    state.score += 1;
    state.streak += 1;
    state.bestStreak = Math.max(state.bestStreak, state.streak);
    haptic(patterns.fast);
  } else {
    state.streak = 0;
    haptic(deriveTone(choice.score) === 'risky' ? patterns.risky : patterns.good);
  }

  if (choice.secretBoost) state.secrets += choice.secretBoost;

  const secretUnlock = fast && willUnlockSecret(scene, state.streak, dialogues.meta);
  if (secretUnlock) state.secrets += 1; // entering the secret scene

  if (choice.ending) setState({ lastEndingTier: choice.ending });

  state.history.push({
    chapter: scene.chapter,
    sceneText: scene.text,
    choiceText: choice.text,
    tone: deriveTone(choice.score),
    fast,
    secret: secretUnlock
  });

  let responseText = reactionFor(choice, state.character);
  if (secretUnlock) responseText += ' ✨ Secret scene unlocked!';

  updateStats();
  disableChoices();
  els.dialogueText.textContent = responseText;

  advanceTo(nextSceneAfter(scene, choice, state.streak, dialogues.meta));
}

function handleTimeout() {
  if (!state.awaitingChoice) return;

  setState({ awaitingChoice: false, responsePhase: true });
  timeLeft = 0;
  state.streak = 0;
  state.score -= 1;

  const scene = state.scene;
  state.history.push({
    chapter: scene.chapter,
    sceneText: scene.text,
    choiceText: null,
    tone: 'timeout',
    fast: false,
    secret: false
  });

  haptic(patterns.timeout);
  updateStats();
  disableChoices();
  els.dialogueText.textContent = `You pause too long. ${state.character.name} smirks and says, "Too slow. Try to keep up."`;

  // The moment moves on anyway: follow the first choice's path.
  const fallback = scene.choices[0];
  advanceTo(fallback ? fallback.next : null);
}

function advanceTo(nextId) {
  advancePending = true;
  timer.delay(RESPONSE_DELAY, () => {
    advancePending = false;
    setState({ responsePhase: false, scene: nextId ? sceneById(state.character, nextId) : null });
    renderScene();
  });
}

async function finishGame() {
  timer.cancel();
  timer.cancelDelay();
  advancePending = false;
  setState({ awaitingChoice: false, responsePhase: false });

  const { renderEnding } = await import('../ending/ending.js');
  renderEnding({ dialogues, meta: dialogues.meta });
}

/* ---------- gestures: swipe history + tap-to-skip ---------- */

function initGestures() {
  const surface = els.gameScreen;
  let startX = 0;
  let startY = 0;
  let tracking = false;

  const onDown = (x, y) => {
    startX = x;
    startY = y;
    tracking = true;
  };

  const onUp = (x, y) => {
    if (!tracking) return;
    tracking = false;
    const dx = x - startX;
    const dy = y - startY;
    const absX = Math.abs(dx);
    const absY = Math.abs(dy);

    if (absX < 10 && absY < 10) {
      // Tap → skip the pending response beat.
      if (advancePending && timer.skipDelay()) haptic(patterns.tap);
      return;
    }

    if (absX > 40 && absX > absY * 1.5) {
      if (dx < 0) openHistory();
      else closeHistory();
    }
  };

  if (window.PointerEvent) {
    surface.addEventListener(
      'pointerdown',
      (e) => {
        if (e.target.closest('button')) return; // buttons handle their own clicks
        onDown(e.clientX, e.clientY);
      },
      { passive: true }
    );
    surface.addEventListener(
      'pointerup',
      (e) => onUp(e.clientX, e.clientY),
      { passive: true }
    );
  } else {
    // Touch Events fallback for older browsers without Pointer Events.
    surface.addEventListener(
      'touchstart',
      (e) => {
        if (e.target.closest('button')) return;
        const t = e.touches[0];
        onDown(t.clientX, t.clientY);
      },
      { passive: true }
    );
    surface.addEventListener(
      'touchend',
      (e) => {
        if (!tracking) return;
        const t = e.changedTouches[0];
        onUp(t.clientX, t.clientY);
      },
      { passive: true }
    );
  }
}

function openHistory() {
  renderHistory();
  els.historyOverlay.classList.add('open');
  els.historyOverlay.setAttribute('aria-hidden', 'false');
  els.historyClose.focus({ preventScroll: true });
}

function closeHistory() {
  if (!els.historyOverlay.classList.contains('open')) return;
  els.historyOverlay.classList.remove('open');
  els.historyOverlay.setAttribute('aria-hidden', 'true');
}

function renderHistory() {
  els.historyList.innerHTML = '';
  const entries = [...state.history].reverse();

  if (!entries.length) {
    const empty = document.createElement('p');
    empty.className = 'history-empty';
    empty.textContent = 'No choices yet — make your first move.';
    els.historyList.appendChild(empty);
    return;
  }

  for (const entry of entries) {
    const item = document.createElement('div');
    item.className = 'history-item';

    const chapter = document.createElement('span');
    chapter.className = 'history-chapter';
    const tags = [entry.fast ? 'fast' : null, entry.secret ? '✨ secret' : null].filter(Boolean);
    chapter.textContent = `Chapter ${entry.chapter}${tags.length ? ` · ${tags.join(' · ')}` : ''}`;

    const sceneText = document.createElement('p');
    sceneText.className = 'history-scene';
    sceneText.textContent = entry.sceneText;

    const choiceText = document.createElement('p');
    choiceText.className = 'history-choice';
    choiceText.textContent = entry.choiceText ? `→ ${entry.choiceText}` : '→ Too slow — no answer.';

    item.append(chapter, sceneText, choiceText);
    els.historyList.appendChild(item);
  }
}
