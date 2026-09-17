// Voice flirt mode: the character speaks (SpeechSynthesis with a
// per-character pitch) and the player can answer by voice
// (SpeechRecognition with fuzzy choice matching). Both APIs are
// feature-detected; the mode degrades gracefully to text-only.

let enabled = false;
let caps = null;
let recognition = null;
let listening = false;

export function init(detectedCaps) {
  caps = detectedCaps || {};
}

export function isSupported() {
  return !!(caps?.speechSynthesis || caps?.speechRecognition);
}

export function canSpeak() {
  return !!caps?.speechSynthesis;
}

export function canListen() {
  return !!caps?.speechRecognition;
}

export function isEnabled() {
  return enabled;
}

export function isListening() {
  return listening;
}

export function toggle() {
  enabled = !enabled;
  if (!enabled) {
    stopSpeaking();
    cancelListening();
  }
  return enabled;
}

/* ---- speech synthesis ---- */

function voiceProfileFor(character) {
  // Deterministic per-character pitch/rate derived from the id.
  const id = character?.id || '';
  let hash = 0;
  for (const ch of id) hash = (hash * 31 + ch.charCodeAt(0)) % 997;
  return { pitch: 0.8 + (hash % 60) / 100, rate: 0.95 + (hash % 20) / 100 };
}

export function speak(text, character) {
  if (!enabled || !canSpeak() || !text) return;
  try {
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    const profile = voiceProfileFor(character);
    utterance.pitch = profile.pitch;
    utterance.rate = profile.rate;
    utterance.lang = navigator.language || 'en-US';
    window.speechSynthesis.speak(utterance);
  } catch {
    /* synthesis unsupported — silently ignore */
  }
}

export function stopSpeaking() {
  try {
    window.speechSynthesis?.cancel();
  } catch {
    /* noop */
  }
}

/* ---- speech recognition ---- */

/** Fuzzy-match a transcript to the closest choice (keyword overlap). */
export function matchChoice(transcript, choices) {
  const words = new Set(
    (transcript || '').toLowerCase().split(/\W+/).filter((w) => w.length > 3)
  );
  let best = -1;
  let bestScore = 0;
  choices.forEach((choice, index) => {
    const choiceWords = choice.text.toLowerCase().split(/\W+/).filter((w) => w.length > 3);
    const overlap = choiceWords.filter((w) => words.has(w)).length;
    if (overlap > bestScore) {
      bestScore = overlap;
      best = index;
    }
  });
  return best;
}

export function listen({ onResult, onError, onEnd } = {}) {
  if (!canListen() || listening) return false;
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  try {
    recognition = new SR();
    recognition.lang = navigator.language || 'en-US';
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;
    recognition.onresult = (event) => {
      const transcript = event.results?.[0]?.[0]?.transcript || '';
      onResult?.(transcript);
    };
    recognition.onerror = (event) => onError?.(event?.error || 'error');
    recognition.onend = () => {
      listening = false;
      recognition = null;
      onEnd?.();
    };
    recognition.start();
    listening = true;
    return true;
  } catch {
    listening = false;
    recognition = null;
    return false;
  }
}

export function cancelListening() {
  try {
    recognition?.stop();
  } catch {
    /* noop */
  }
  listening = false;
  recognition = null;
}
