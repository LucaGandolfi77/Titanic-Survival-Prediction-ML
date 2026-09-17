// Ending screen: renders the adaptive ending, persists personal records
// and wires the Web Share button. Loaded dynamically via import() —
// none of this code is needed until the game finishes for the first time.

import { els } from '../../ui/dom.js';
import { state } from '../../core/state.js';
import { storage } from '../../core/storage.js';
import { showScreen } from '../../ui/router.js';
import { haptic, patterns } from '../../core/haptics.js';
import { showToast } from '../../ui/toast.js';
import { getEnding } from '../../data/dialogues.js';

let shareWired = false;

export function renderEnding({ dialogues, meta = {} }) {
  const fizzleBelow = meta.fizzleBelowScore ?? 6;
  const tier = state.score < fizzleBelow ? 'fizzle' : state.lastEndingTier || 'good';
  const ending = getEnding(dialogues, tier, state.character);

  els.endBadge.textContent = ending.badge;
  els.endTitle.textContent = ending.title;
  els.endText.textContent = `${ending.text} Secret scenes unlocked: ${state.secrets}. Best fast streak: ${state.bestStreak}.`;
  els.finalScore.textContent = state.score;
  els.finalStreak.textContent = state.bestStreak;
  els.finalSecret.textContent = state.secrets;

  const records = storage.load();
  const isRecord = state.score > records.bestScore;
  storage.update((data) => ({
    ...data,
    bestScore: Math.max(data.bestScore, state.score),
    bestStreak: Math.max(data.bestStreak, state.bestStreak),
    gamesPlayed: data.gamesPlayed + 1,
    endings: data.endings.includes(tier) ? data.endings : [...data.endings, tier]
  }));

  haptic(tier === 'great' ? patterns.ending : patterns.good);
  showScreen('end');

  if (!shareWired) {
    shareWired = true;
    els.shareBtn.addEventListener('click', handleShare);
  }

  if (isRecord) {
    showToast(`New personal record: ${state.score} charm! 🏆`, { duration: 4000 });
  }
}

async function handleShare() {
  const text = `I scored ${state.score} charm points in Speed Crush! Can you beat me?`;
  const url = window.location.href;

  if (typeof navigator.share === 'function') {
    try {
      await navigator.share({ title: 'Speed Crush', text, url });
    } catch {
      /* user cancelled the share sheet */
    }
  } else if (navigator.clipboard?.writeText) {
    try {
      await navigator.clipboard.writeText(`${text} ${url}`);
      showToast('Link copied to clipboard ✅', { duration: 3000 });
    } catch {
      showToast('Could not copy the link', { duration: 3000 });
    }
  } else {
    showToast(`${text} ${url}`, { duration: 5000 });
  }
}
