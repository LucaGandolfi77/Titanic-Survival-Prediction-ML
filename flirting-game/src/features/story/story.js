// Story export/import via the File System Access API, with fallbacks.
// Export: the complete playthrough (scenes, choices, ending, stats) as JSON.
// Import: replay a friend's story, or load a community story pack
// (the open marketplace schema, validated before it becomes playable).

import { els } from '../../ui/dom.js';
import { showToast } from '../../ui/toast.js';
import { t } from '../../core/i18n.js';
import { validateStoryPack } from '../../data/validate.js';

export function canUseFileSystem() {
  return (
    typeof window !== 'undefined' &&
    (typeof window.showSaveFilePicker === 'function' || typeof window.showOpenFilePicker === 'function')
  );
}

export async function exportStory({ history, ending, stats }) {
  const payload = {
    format: 'speed-crush-story',
    version: 1,
    exportedAt: new Date().toISOString(),
    ending,
    stats,
    scenes: history
  };
  const json = JSON.stringify(payload, null, 2);

  if (typeof window !== 'undefined' && typeof window.showSaveFilePicker === 'function') {
    try {
      const handle = await window.showSaveFilePicker({
        suggestedName: `speed-crush-story-${Date.now()}.json`,
        types: [{ description: 'Speed Crush story', accept: { 'application/json': ['.json'] } }]
      });
      const writable = await handle.createWritable();
      await writable.write(json);
      await writable.close();
      showToast('Story exported ✅', { duration: 3000 });
      return true;
    } catch {
      /* user cancelled → fall through to the download fallback */
    }
  }

  const blob = new Blob([json], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `speed-crush-story-${Date.now()}.json`;
  a.click();
  URL.revokeObjectURL(url);
  showToast('Story downloaded ✅', { duration: 3000 });
  return true;
}

function pickViaInput() {
  return new Promise((resolve) => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.json,application/json';
    input.onchange = async () => {
      const file = input.files?.[0];
      resolve(file ? await file.text() : null);
    };
    input.click();
  });
}

async function importFile() {
  let text = null;
  if (typeof window !== 'undefined' && typeof window.showOpenFilePicker === 'function') {
    try {
      const [handle] = await window.showOpenFilePicker({
        types: [{ description: 'Speed Crush story / pack', accept: { 'application/json': ['.json'] } }]
      });
      const file = await handle.getFile();
      text = await file.text();
    } catch {
      return null; // cancelled
    }
  } else {
    text = await pickViaInput();
    if (!text) return null;
  }
  try {
    return JSON.parse(text);
  } catch {
    showToast('Invalid JSON file.', { duration: 3000 });
    return null;
  }
}

/**
 * Import + validate: returns playable story-pack data, or null
 * (a friend's playthrough is replayed instead of returned).
 */
export async function importStoryOrPack() {
  const data = await importFile();
  if (!data) return null;

  if (data.format === 'speed-crush-story') {
    replayStory(data);
    return { kind: 'story' };
  }

  const validation = validateStoryPack(data);
  if (!validation.ok) {
    showToast(`Invalid story pack: ${validation.errors[0]}`, { duration: 4500 });
    return { kind: 'invalid' };
  }
  showToast(`Story pack valid — ${validation.scenes} scenes ✅`, { duration: 3500 });
  return { kind: 'pack', data };
}

/** Replay a friend's story in the history sheet (read-only). */
export function replayStory(story) {
  els.historyList.innerHTML = '';

  const titleCard = document.createElement('div');
  titleCard.className = 'history-item story-title';
  const badge = document.createElement('span');
  badge.className = 'history-chapter';
  badge.textContent = `${story.ending?.badge || 'Story'} · ${t('game.historyTitle')}`;
  const title = document.createElement('p');
  title.className = 'history-choice';
  title.textContent = story.ending?.title || 'A shared story';
  titleCard.append(badge, title);
  els.historyList.appendChild(titleCard);

  for (const scene of story.scenes || []) {
    const item = document.createElement('div');
    item.className = 'history-item';

    const chapter = document.createElement('span');
    chapter.className = 'history-chapter';
    const tags = [scene.fast ? 'fast' : null, scene.secret ? '✨ secret' : null, scene.custom ? '✍️ own' : null].filter(
      Boolean
    );
    chapter.textContent = `Chapter ${scene.chapter}${tags.length ? ` · ${tags.join(' · ')}` : ''}`;

    const sceneText = document.createElement('p');
    sceneText.className = 'history-scene';
    sceneText.textContent = scene.sceneText;

    const choiceText = document.createElement('p');
    choiceText.className = 'history-choice';
    choiceText.textContent = scene.choiceText ? `→ ${scene.choiceText}` : '→ Too slow — no answer.';

    item.append(chapter, sceneText, choiceText);
    els.historyList.appendChild(item);
  }

  els.historyOverlay.classList.add('open');
  els.historyOverlay.setAttribute('aria-hidden', 'false');
  els.historyClose?.focus({ preventScroll: true });
}
