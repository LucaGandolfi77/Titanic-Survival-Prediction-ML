// Push-style rich notifications with quick-reply actions, scheduled via
// the experimental Notification Triggers API when available. Opt-in only;
// the Service Worker handles notification clicks (Play now → open the game).

import { storage } from '../../core/storage.js';
import { collectAmbient } from '../../core/ambient.js';
import { t } from '../../core/i18n.js';

export function isSupported() {
  return typeof window !== 'undefined' && 'Notification' in window;
}

export function permission() {
  return isSupported() ? Notification.permission : 'denied';
}

export function remindersEnabled() {
  return !!storage.load().reminders;
}

export async function enableReminders() {
  if (!isSupported()) return { ok: false, reason: 'unsupported' };
  let perm = Notification.permission;
  if (perm === 'default') {
    try {
      perm = await Notification.requestPermission();
    } catch {
      perm = 'denied';
    }
  }
  if (perm !== 'granted') return { ok: false, reason: perm };
  storage.save({ reminders: true });
  const scheduled = await scheduleContextual();
  return { ok: true, scheduled };
}

/**
 * Schedule the next context-aware reminder ("It's 11pm — the right hour
 * for trouble") via the experimental Notification Triggers API. When the
 * API is unavailable, the Service Worker reminds on the next visit instead.
 */
export async function scheduleContextual() {
  if (typeof window === 'undefined' || typeof window.NotificationTrigger !== 'function') {
    return false;
  }
  try {
    const ambient = await collectAmbient();
    const now = new Date();
    const at = new Date(now);
    at.setHours(23, 0, 0, 0); // next 11pm local time
    if (at <= now) at.setDate(at.getDate() + 1);
    const reg = await navigator.serviceWorker?.ready;
    await reg?.showNotification('Speed Crush 💘', {
      body: `It's ${String(at.getHours()).padStart(2, '0')}:00 — ${t(
        `mood.${ambient.phase.id}.flavor`
      )}. Your crush is waiting.`,
      tag: 'speed-crush-reminder',
      showTrigger: new window.NotificationTrigger({ timestamp: at.getTime() }),
      actions: [
        { action: 'play', title: t('setup.start') },
        { action: 'dismiss', title: t('game.close') }
      ]
    });
    return true;
  } catch {
    return false;
  }
}
