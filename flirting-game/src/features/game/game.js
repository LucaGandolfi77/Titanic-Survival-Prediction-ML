// Game engine: renders scenes from the dialog graph, drives the reaction
// timer, handles choices/timeouts, swipe gestures and the choice log.
// Phase 2: adaptive difficulty, ambient mood match, procedural AI detours,
// custom flirty lines with sentiment scoring, and voice flirt mode.
// The ending module is dynamically imported only when the game finishes.

import { els } from '../../ui/dom.js';
import { state, setState, resetRound } from '../../core/state.js';
import { storage } from '../../core/storage.js';
import { TimerEngine } from '../../core/timer.js';
import { haptic, patterns } from '../../core/haptics.js';
import { showScreen } from '../../ui/router.js';
import { showToast } from '../../ui/toast.js';
import { t } from '../../core/i18n.js';
import { collectAmbient, moodLine } from '../../core/ambient.js';
import * as difficulty from '../../core/difficulty.js';
import { start as startMotion, stop as stopMotion, requestPermission as requestMotionPermission } from '../../core/motion.js';
import * as duet from '../duet/duet.js';
import { detectCapabilities } from '../../ai/capabilities.js';
import { generateScene, resetGenerator } from '../../ai/generator.js';
import { scoreLine } from '../../ai/sentiment.js';
import * as voice from '../voice/voice.js';
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
let profile = null; // adaptive difficulty profile
let ambient = null; // mood match context for this game
let generated = new Map(); // AI-generated detour scenes
let genCount = 0; // detours injected this game
let genCooldown = 0; // fixed scenes before the next detour is allowed
let currentChapter = 0;
let ambientMode = false; // glanceable "night out" mode
let lastSpikeHandledAt = 0;

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
  initCustomLine();
  initVoice();

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
  generated.clear();
  resetGenerator();
  genCount = 0;
  genCooldown = 0;
  currentChapter = 0;
  toggleCustomInput(false);
  exitAmbientMode();
  lastSpikeHandledAt = 0;

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

  // Share Target: weave the shared text into a custom opening scene.
  let first = firstScene(character);
  if (state.sharedPrompt) {
    first = buildSharedScene(character, state.sharedPrompt, first);
    setState({ sharedPrompt: null });
  }

  // Adaptive difficulty: restore the calibrated profile, or init from meta.
  profile = difficulty.createProfile(dialogues.meta);
  const storedProfile = storage.load().difficulty;
  if (storedProfile?.samples > 0) profile = { ...profile, ...storedProfile };
  state.timeLimit = profile.timeLimit;
  state.fastThreshold = profile.threshold;

  resetRound();
  setState({ character, scene: first });

  if (els.customBtn) els.customBtn.hidden = !state.aiMode;

  setAvatar(state.character);
  updateStats();
  showScreen('game');
  renderScene();

  // Mood Match: one-time ambient toast with the current context.
  ambient = await collectAmbient();
  showToast(moodLine(ambient), { duration: 4500 });
}

export function stopGame() {
  if (timer) {
    timer.cancel();
    timer.cancelDelay();
  }
  advancePending = false;
  timeLeft = 0;
  voice.stopSpeaking();
  voice.cancelListening();
  setState({ awaitingChoice: false, responsePhase: false });
  closeHistory();
  toggleCustomInput(false);
}

/** Play an imported community story pack (validated by the caller). */
export async function playStoryPack(data) {
  dialogues = data;
  const genders = Object.keys(data.characters || {});
  if (!genders.includes(state.interestGender)) {
    setState({ interestGender: genders[0] || 'female' });
  }
  await startGame();
}

/** The last completed playthrough, for the story export. */
export function getLastStory() {
  return {
    history: state.history,
    ending: {
      badge: els.endBadge.textContent,
      title: els.endTitle.textContent
    },
    stats: storage.load()
  };
}

/* ---------- rendering ---------- */

function renderScene() {
  const scene = state.scene;
  if (!scene) {
    finishGame();
    return;
  }

  // Adaptive difficulty: recalibrate when the chapter changes.
  if (scene.chapter !== currentChapter) {
    currentChapter = scene.chapter;
    difficulty.calibrateForChapter(profile);
    state.timeLimit = profile.timeLimit;
    state.fastThreshold = profile.threshold;
  }

  els.chapterLabel.textContent = `${t('game.chapter', { n: scene.chapter })}`;
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
  voice.speak(scene.text, state.character);
  if (state.duetActive) {
    duet.send({
      type: 'scene',
      text: scene.text,
      character: state.character.name,
      choices: scene.choices.map((c) => c.text)
    });
  }
}

/** Share Target: a custom opening scene woven around the shared text. */
function buildSharedScene(character, sharedText, fallback) {
  const prompt = sharedText.length > 120 ? `${sharedText.slice(0, 117)}…` : sharedText;
  return {
    id: 'shared_open',
    chapter: 1,
    text: `${character.name} slides your phone back across the table. "You shared this — '${prompt}'. Convince me it's worth a story."`,
    choices: (fallback?.choices || []).map((c) => ({ ...c }))
  };
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
  applyChoice(choice);
}

/**
 * Shared choice pipeline: used by choice buttons and by custom
 * sentiment-scored lines. `customLine` carries the player's own text.
 */
function applyChoice(choice, { customLine = null } = {}) {
  const scene = state.scene;
  const fast = timeLeft / state.timeLimit >= state.fastThreshold && choice.score > 0;

  // Stop the countdown: onFinish can no longer fire after this point.
  timer.cancel();
  setState({ awaitingChoice: false, responsePhase: true });

  // Adaptive difficulty: fold this reaction into the moving average.
  difficulty.observe(profile, state.timeLimit - timeLeft);

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
    secret: secretUnlock,
    reactionTime: Number((state.timeLimit - timeLeft).toFixed(2)),
    custom: Boolean(customLine)
  });

  let responseText = reactionFor(choice, state.character);
  if (customLine) responseText = `You lean in: “${customLine}” — ${responseText}`;
  if (secretUnlock) responseText += ' ✨ Secret scene unlocked!';

  updateStats();
  disableChoices();

  if (state.duetActive) {
    // Co-op duet: the crush reacts, their line is what we show.
    duet.send({ type: 'choice', text: choice.text });
    showToast('Waiting for your crush\u2019s reaction… 👯', { duration: 2500 });
    duet.waitForReaction(6000).then((reaction) => {
      const text = reaction || reactionFor(choice, state.character);
      els.dialogueText.textContent = customLine ? `You lean in: “${customLine}” — ${text}` : text;
      voice.speak(text, state.character);
      advanceTo(resolveNext(scene, choice, state.streak));
    });
    return;
  }

  els.dialogueText.textContent = responseText;
  voice.speak(responseText, state.character);

  advanceTo(resolveNext(scene, choice, state.streak));
}

function handleTimeout() {
  if (!state.awaitingChoice) return;

  setState({ awaitingChoice: false, responsePhase: true });
  timeLeft = 0;
  state.streak = 0;
  state.score -= 1;

  // Full-window reaction = a "too hard" signal for the difficulty curve.
  difficulty.observe(profile, state.timeLimit);

  const scene = state.scene;
  state.history.push({
    chapter: scene.chapter,
    sceneText: scene.text,
    choiceText: null,
    tone: 'timeout',
    fast: false,
    secret: false,
    reactionTime: state.timeLimit,
    custom: false
  });

  haptic(patterns.timeout);
  updateStats();
  disableChoices();
  const timeoutText = `You pause too long. ${state.character.name} smirks and says, "Too slow. Try to keep up."`;
  els.dialogueText.textContent = timeoutText;
  voice.speak(timeoutText, state.character);

  // The moment moves on anyway: follow the first choice's path.
  const fallback = scene.choices[0];
  advanceTo(fallback ? fallback.next : null);
}

/* ---------- ambient "night out" mode ---------- */

function onMotionSpike() {
  if (!ambientMode) return;
  const now = performance.now();
  if (now - lastSpikeHandledAt < 3000) return; // debounce
  lastSpikeHandledAt = now;
  exitAmbientMode();
  updateAmbientButton();
  showToast('Welcome back ✨', { duration: 2000 });
}

function exitAmbientMode() {
  ambientMode = false;
  document.body.classList.remove('ambient-ui');
  stopMotion(onMotionSpike);
}

function updateAmbientButton() {
  if (!els.ambientBtn) return;
  els.ambientBtn.classList.toggle('active', ambientMode);
  els.ambientBtn.textContent = ambientMode ? '✨' : '☾';
}

/**
 * Resolve the next scene id: fixed secret detour wins, then a mood-biased
 * procedural detour (AI mode, max 2 per game with a cooldown), then the
 * fixed choice.next.
 */
function resolveNext(scene, choice, streak) {
  const fixed = nextSceneAfter(scene, choice, streak, dialogues.meta);
  if (fixed === null) return null; // final scene → game over

  if (state.aiMode && genCount < 2 && genCooldown <= 0) {
    genCount += 1;
    genCooldown = 2;
    const gen = generateScene({
      character: state.character,
      ambient,
      chapter: scene.chapter,
      returnTo: fixed
    });
    generated.set(gen.id, gen);
    return gen.id;
  }

  genCooldown = Math.max(0, genCooldown - 1);
  return fixed;
}

function advanceTo(nextId) {
  advancePending = true;
  timer.delay(RESPONSE_DELAY, () => {
    advancePending = false;
    const scene =
      (nextId && (generated.get(nextId) || sceneById(state.character, nextId))) || null;
    setState({ responsePhase: false, scene });
    renderScene();
  });
}

async function finishGame() {
  timer.cancel();
  timer.cancelDelay();
  advancePending = false;
  setState({ awaitingChoice: false, responsePhase: false });
  voice.stopSpeaking();
  exitAmbientMode();
  if (state.duetActive) {
    duet.send({ type: 'ending', text: els.endTitle.textContent });
    duet.closeSession();
    setState({ duetActive: false });
  }

  // Persist the calibrated difficulty profile for the next session.
  if (profile) storage.update((data) => ({ ...data, difficulty: profile }));

  const { renderEnding } = await import('../ending/ending.js');
  renderEnding({ dialogues, meta: dialogues.meta });
}

/* ---------- custom flirty line (sentiment-scored) ---------- */

function initCustomLine() {
  els.customBtn?.addEventListener('click', () => {
    haptic(patterns.tap);
    toggleCustomInput(!els.customField.classList.contains('open'));
  });
  els.customSend?.addEventListener('click', handleCustomLine);
  els.customInput?.addEventListener('keydown', (event) => {
    if (event.key === 'Enter') {
      event.preventDefault();
      handleCustomLine();
    }
  });
}

function toggleCustomInput(show) {
  els.customField?.classList.toggle('open', show);
  if (show) els.customInput?.focus({ preventScroll: true });
}

function handleCustomLine() {
  if (!state.awaitingChoice) return;
  const text = (els.customInput?.value || '').trim();
  if (!text) return;
  els.customInput.value = '';
  toggleCustomInput(false);

  scoreLine(text).then((verdict) => {
    if (!state.awaitingChoice) return; // the timer may have expired meanwhile
    const isFinal = !state.scene.choices[0]?.next;
    const choice = {
      text: `“${text}”`,
      score: verdict.score,
      // On the final scene, the sentiment decides the ending tier.
      ending: isFinal ? (verdict.tone === 'good' ? 'great' : 'good') : undefined
    };
    applyChoice(choice, { customLine: text });
  });
}

/* ---------- voice flirt mode ---------- */

function initVoice() {
  els.voiceBtn?.addEventListener('click', () => {
    const on = voice.toggle();
    haptic(patterns.tap);
    updateVoiceButtons();
    showToast(on ? 'Voice mode on 🔊' : 'Voice mode off', { duration: 2000 });
  });

  els.micBtn?.addEventListener('click', () => {
    if (!state.awaitingChoice) {
      showToast('Wait for the choices first.', { duration: 2500 });
      return;
    }
    const started = voice.listen({
      onResult: (transcript) => {
        const best = voice.matchChoice(transcript, state.scene?.choices || []);
        if (best >= 0) {
          haptic(patterns.tap);
          state.voiceUsed = true;
          handleChoice(best);
        } else {
          showToast(`“${transcript}” — no matching choice.`, { duration: 3000 });
        }
      },
      onError: (err) =>
        showToast(err === 'not-allowed' ? 'Mic permission denied.' : 'Mic error — try again.', {
          duration: 3000
        }),
      onEnd: updateVoiceButtons
    });
    if (started) {
      updateVoiceButtons();
      showToast(t('toast.listening'), { duration: 4000 });
    }
  });

  // Ambient "night out" mode: glanceable UI + motion wake.
  els.ambientBtn?.addEventListener('click', async () => {
    haptic(patterns.tap);
    if (!ambientMode) {
      const granted = await requestMotionPermission();
      if (!granted) {
        showToast('Motion permission denied — ambient mode stays off.', { duration: 3000 });
        return;
      }
      ambientMode = true;
      document.body.classList.add('ambient-ui');
      startMotion(onMotionSpike);
      showToast('Ambient mode on ☙ — pick up the phone to exit', { duration: 3500 });
    } else {
      exitAmbientMode();
    }
    updateAmbientButton();
  });

  detectCapabilities().then((caps) => {
    voice.init(caps);
    updateVoiceButtons();
  });
}

function updateVoiceButtons() {
  if (!els.voiceBtn) return;
  els.voiceBtn.hidden = !voice.isSupported();
  els.micBtn.hidden = !(voice.isEnabled() && voice.canListen());
  els.voiceBtn.classList.toggle('active', voice.isEnabled());
  els.micBtn.classList.toggle('listening', voice.isListening());
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
    surface.addEventListener('pointerup', (e) => onUp(e.clientX, e.clientY), { passive: true });
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
    const tags = [entry.fast ? 'fast' : null, entry.secret ? '✨ secret' : null, entry.custom ? '✍️ own' : null].filter(
      Boolean
    );
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
