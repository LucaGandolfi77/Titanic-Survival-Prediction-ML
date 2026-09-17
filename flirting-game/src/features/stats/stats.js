// Stats vault: personal records + device context, protected by a Passkey
// (Face ID / Touch ID) when one is registered. Falls back to an open
// vault when WebAuthn is unavailable.

import { els } from '../../ui/dom.js';
import { storage } from '../../core/storage.js';
import { showToast } from '../../ui/toast.js';
import * as webauthn from '../../core/webauthn.js';
import { collectAmbient, moodChip } from '../../core/ambient.js';
import { detectCapabilities, devicePreference } from '../../ai/capabilities.js';
import { achievementStates } from '../achievements/achievements.js';
import { exportStory, importStoryOrPack } from '../story/story.js';
import { getLastStory } from '../game/game.js';

export function initStats() {
  if (!els.statsBtn) return;
  els.statsBtn.addEventListener('click', openStats);
  els.statsClose?.addEventListener('click', closeStats);
  els.biometricBtn?.addEventListener('click', setupBiometric);
  els.exportBtn?.addEventListener('click', () => exportStory(getLastStory()));
  els.importBtn?.addEventListener('click', async () => {
    const result = await importStoryOrPack();
    if (result?.kind === 'pack') {
      // Play the imported story pack immediately.
      const { playStoryPack } = await import('../game/game.js');
      closeStats();
      playStoryPack(result.data);
    }
  });
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape') closeStats();
  });
}

async function openStats() {
  if (webauthn.hasPasskey()) {
    try {
      const ok = await webauthn.verifyPasskey();
      if (!ok) {
        showToast('Verification failed — stats stay locked.', { duration: 3500 });
        return;
      }
    } catch {
      showToast('Verification cancelled.', { duration: 3000 });
      return;
    }
  }
  renderStats();
  els.statsOverlay.classList.add('open');
  els.statsOverlay.setAttribute('aria-hidden', 'false');
  els.statsClose?.focus({ preventScroll: true });
}

function closeStats() {
  if (!els.statsOverlay.classList.contains('open')) return;
  els.statsOverlay.classList.remove('open');
  els.statsOverlay.setAttribute('aria-hidden', 'true');
}

function renderStats() {
  const data = storage.load();

  els.statGames.textContent = String(data.gamesPlayed);
  els.statBestScore.textContent = data.bestScore > 0 ? String(data.bestScore) : '—';
  els.statBestStreak.textContent = String(data.bestStreak);
  els.statEndings.textContent = data.endings.length ? data.endings.join(', ') : '—';

  const d = data.difficulty || {};
  els.statDifficulty.textContent = d.samples
    ? `${d.timeLimit}s window · fast under ${Math.round(d.threshold * 100)}%`
    : 'Standard (8s · 65%)';
  els.statAvgReaction.textContent = d.samples
    ? `${Number(d.avgReaction).toFixed(1)}s avg`
    : '—';

  if (webauthn.isSupported()) {
    els.statsBiometric.hidden = false;
    els.biometricStatus.textContent = webauthn.hasPasskey()
      ? 'Unlocked with Face ID / Touch ID ✓'
      : 'Not set up yet — protect this vault with your biometrics.';
    els.biometricBtn.hidden = webauthn.hasPasskey();
  } else {
    els.statsBiometric.hidden = true;
  }

  collectAmbient().then((ambient) => {
    els.statAmbient.textContent = moodChip(ambient);
  });
  detectCapabilities().then((caps) => {
    els.statDevice.textContent = caps.webnn
      ? `Neural (WebNN) · ${devicePreference(caps)}`
      : caps.webgpu
        ? `Neural (WebGPU) · ${devicePreference(caps)}`
        : 'WebAssembly baseline';
  });

  renderAchievements();
}

function renderAchievements() {
  if (!els.achievementsList) return;
  els.achievementsList.innerHTML = '';
  for (const a of achievementStates()) {
    const item = document.createElement('div');
    item.className = `achievement${a.unlocked ? ' unlocked' : ''}`;
    const icon = document.createElement('span');
    icon.className = 'achievement-icon';
    icon.textContent = a.unlocked ? a.icon : '🔒';
    const copy = document.createElement('div');
    const name = document.createElement('strong');
    name.textContent = a.name;
    const desc = document.createElement('small');
    desc.textContent = a.desc;
    copy.append(name, desc);
    item.append(icon, copy);
    els.achievementsList.appendChild(item);
  }
}

async function setupBiometric() {
  try {
    await webauthn.registerPasskey();
    showToast('Biometric unlock enabled 🔒', { duration: 3500 });
    renderStats();
  } catch {
    showToast('Could not set up the passkey.', { duration: 3500 });
  }
}
