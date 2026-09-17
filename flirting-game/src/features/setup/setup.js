// Setup screen: restores preferences + language, shows the personal best
// and daily streak, manages the AI mode toggle + neural model download,
// notifications, duet mode, and opens the stats vault.

import { els } from '../../ui/dom.js';
import { setState } from '../../core/state.js';
import { storage } from '../../core/storage.js';
import { showScreen } from '../../ui/router.js';
import { haptic, patterns } from '../../core/haptics.js';
import { showToast } from '../../ui/toast.js';
import { collectAmbient, moodChip } from '../../core/ambient.js';
import { detectCapabilities } from '../../ai/capabilities.js';
import { ensureNeural } from '../../ai/sentiment.js';
import {
  t,
  setLanguage,
  getLanguage,
  applyTranslations
} from '../../core/i18n.js';
import { enableReminders, isSupported as notificationsSupported } from '../notifications/notifications.js';
import { dailyStreakBadge } from '../achievements/achievements.js';
import { openDuet, initDuet } from '../duet/duet.js';
import { startGame, stopGame } from '../game/game.js';
import { initStats } from '../stats/stats.js';

export function initSetup() {
  const { prefs, aiModelDownloaded } = storage.load();
  setLanguage(prefs.language || undefined);
  applyTranslations();
  els.playerGender.value = prefs.playerGender;
  els.interestGender.value = prefs.interestGender;
  if (els.aiToggle) els.aiToggle.checked = !!prefs.aiMode;
  if (els.languageSelect) els.languageSelect.value = getLanguage();
  updateAiModelButton(aiModelDownloaded);
  renderBest();
  renderStreakBadge();
  renderAmbientChip();

  els.startBtn.addEventListener('click', () => {
    haptic(patterns.tap);
    const playerGender = els.playerGender.value;
    const interestGender = els.interestGender.value;
    const aiMode = !!els.aiToggle?.checked;
    setState({ playerGender, interestGender, aiMode });
    storage.savePrefs(playerGender, interestGender, aiMode, getLanguage());
    startGame();
  });

  els.languageSelect?.addEventListener('change', () => {
    setLanguage(els.languageSelect.value);
    applyTranslations();
    storage.savePrefs(
      els.playerGender.value,
      els.interestGender.value,
      !!els.aiToggle?.checked,
      getLanguage()
    );
  });

  els.aiToggle?.addEventListener('change', () => {
    updateAiModelButton(storage.load().aiModelDownloaded);
  });

  els.aiModelBtn?.addEventListener('click', downloadAiModel);

  els.notifBtn?.addEventListener('click', async () => {
    if (!notificationsSupported()) {
      showToast('Notifications are not supported here.', { duration: 3000 });
      return;
    }
    const result = await enableReminders();
    if (result.ok) {
      showToast(
        result.scheduled ? 'Reminder scheduled for 11pm 🔔' : 'Reminders on 🔔',
        { duration: 3500 }
      );
    } else {
      showToast(
        result.reason === 'denied' ? 'Permission denied.' : 'Notifications unavailable.',
        { duration: 3500 }
      );
    }
  });

  els.duetBtn?.addEventListener('click', () => {
    haptic(patterns.tap);
    openDuet();
  });

  els.restartBtn.addEventListener('click', () => {
    haptic(patterns.tap);
    stopGame();
    showScreen('setup');
    renderBest();
    renderStreakBadge();
  });

  els.playAgainBtn.addEventListener('click', () => {
    haptic(patterns.tap);
    stopGame();
    showScreen('setup');
    renderBest();
    renderStreakBadge();
  });

  initStats();
  initDuet(els);
}

function renderBest() {
  const { bestScore } = storage.load();
  els.bestValue.textContent = bestScore > 0 ? `${bestScore} charm` : '—';
}

function renderStreakBadge() {
  if (!els.streakBadge) return;
  els.streakBadge.textContent = dailyStreakBadge() || '';
  els.streakBadge.hidden = !dailyStreakBadge();
}

function renderAmbientChip() {
  if (!els.ambientChip) return;
  collectAmbient().then((ambient) => {
    els.ambientChip.textContent = moodChip(ambient);
  });
}

function updateAiModelButton(downloaded) {
  if (!els.aiModelBtn) return;
  const aiOn = !!els.aiToggle?.checked;
  els.aiModelBtn.hidden = !aiOn;
  if (aiOn) {
    els.aiModelBtn.textContent = downloaded
      ? t('setup.aiModelReady')
      : t('setup.aiModel');
    els.aiModelBtn.disabled = false;
  }
}

async function downloadAiModel() {
  if (!els.aiModelBtn) return;
  const btn = els.aiModelBtn;
  btn.disabled = true;
  btn.textContent = t('setup.aiModelDownloading');
  try {
    const caps = await detectCapabilities();
    await ensureNeural(caps, (progress) => {
      if (progress?.status === 'progress' && progress.progress != null) {
        btn.textContent = `Downloading… ${Math.round(progress.progress)}%`;
      }
    });
    storage.save({ aiModelDownloaded: true });
    btn.textContent = t('setup.aiModelReady');
    showToast('AI model ready — fully offline from now on ✅', { duration: 4000 });
  } catch (err) {
    console.error('AI model download failed', err);
    btn.disabled = false;
    btn.textContent = 'Download failed — retry';
    showToast('Could not download the model. Check your connection.', { duration: 4000 });
  }
}
