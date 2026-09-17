// Dialog engine: loads the graph-based dialogues.json and provides
// traversal helpers (secret detours, tone derivation, endings).
// Content lives in dialogues.json; logic lives here.

let cache = null;

export async function loadDialogues() {
  if (cache) return cache;
  const res = await fetch('./dialogues.json');
  if (!res.ok) throw new Error(`Failed to load dialogues: ${res.status}`);
  cache = await res.json();
  return cache;
}

export function pickCharacter(data, interestGender) {
  const pool = data.characters?.[interestGender] || [];
  return pool[Math.floor(Math.random() * pool.length)] || null;
}

export function applyMeta(data, stateRef) {
  const meta = data.meta || {};
  stateRef.timeLimit = meta.timerSeconds || 8;
  stateRef.fastThreshold = meta.fastThreshold || 0.65;
}

export function firstScene(character) {
  return character?.chapters?.start || null;
}

export function sceneById(character, id) {
  if (!id) return null;
  return Object.values(character.chapters).find((scene) => scene.id === id) || null;
}

/** Map a choice score to the good / safe / risky button tone. */
export function deriveTone(score) {
  if (score >= 4) return 'good';
  if (score >= 3) return 'safe';
  return 'risky';
}

/** True if finishing `scene` now would detour into its secret scene. */
export function willUnlockSecret(scene, streak, meta) {
  if (!scene?.secret?.next) return false;
  const needed = scene.secret.condition?.streakAtLeast ?? meta?.streakNeeded ?? 2;
  return streak >= needed;
}

/** Next scene id after a choice in `scene` (secret detour wins over choice.next). */
export function nextSceneAfter(scene, choice, streak, meta) {
  if (willUnlockSecret(scene, streak, meta)) return scene.secret.next;
  return choice.next || null;
}

const TONE_REACTIONS = {
  good: '{name} smiles — that landed harder than expected.',
  safe: '{name} nods slowly, clearly wanting to hear more.',
  risky: '{name} raises an eyebrow. Smooth recovery needed.'
};

/**
 * Reaction line shown after a choice. Prefers an explicit `choice.response`
 * written by content authors, falls back to a tone-based line.
 */
export function reactionFor(choice, character) {
  const text =
    choice.response ||
    TONE_REACTIONS[deriveTone(choice.score)] ||
    TONE_REACTIONS.safe;
  return text.replaceAll('{name}', character.name);
}

export function getEnding(data, tier, character) {
  const endings = data.endings || {};
  const ending = endings[tier] || endings.good || {};
  return {
    badge: ending.badge || 'Ending',
    title: (ending.title || 'The story ends').replaceAll('{name}', character.name),
    text: (ending.text || '').replaceAll('{name}', character.name)
  };
}
