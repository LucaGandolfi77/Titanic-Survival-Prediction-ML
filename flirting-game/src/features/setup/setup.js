// Setup screen: restores preferences, shows the personal best, starts the game.

import { els } from '../../ui/dom.js';
import { setState } from '../../core/state.js';
import { storage } from '../../core/storage.js';
import { showScreen } from '../../ui/router.js';
import { haptic, patterns } from '../../core/haptics.js';
import { startGame, stopGame } from '../game/game.js';

export function initSetup() {
  const { prefs } = storage.load();
  els.playerGender.value = prefs.playerGender;
  els.interestGender.value = prefs.interestGender;
  renderBest();

  els.startBtn.addEventListener('click', () => {
    haptic(patterns.tap);
    const playerGender = els.playerGender.value;
    const interestGender = els.interestGender.value;
    setState({ playerGender, interestGender });
    storage.savePrefs(playerGender, interestGender);
    startGame();
  });

  els.restartBtn.addEventListener('click', () => {
    haptic(patterns.tap);
    stopGame();
    showScreen('setup');
    renderBest();
  });

  els.playAgainBtn.addEventListener('click', () => {
    haptic(patterns.tap);
    stopGame();
    showScreen('setup');
    renderBest();
  });
}

function renderBest() {
  const { bestScore } = storage.load();
  els.bestValue.textContent = bestScore > 0 ? `${bestScore} charm` : '—';
}
