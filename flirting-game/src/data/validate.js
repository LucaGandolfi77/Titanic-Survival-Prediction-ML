// Story pack schema validation: the open marketplace format, identical to
// the dialogues.json structure. Validates structure + graph integrity
// before an imported pack becomes playable.

export function validateStoryPack(data) {
  const errors = [];
  if (!data || typeof data !== 'object' || Array.isArray(data)) {
    return { ok: false, errors: ['Not a JSON object'], scenes: 0 };
  }
  if (!data.characters || typeof data.characters !== 'object') errors.push('Missing "characters"');
  if (!data.endings || typeof data.endings !== 'object') errors.push('Missing "endings"');

  let scenes = 0;
  if (data.characters && typeof data.characters === 'object') {
    for (const [gender, chars] of Object.entries(data.characters)) {
      if (!Array.isArray(chars) || !chars.length) {
        errors.push(`"${gender}" has no characters`);
        continue;
      }
      for (const ch of chars) {
        if (!ch?.id || !ch?.name || !ch?.chapters?.start) {
          errors.push(`Character missing id/name/start in "${gender}"`);
          continue;
        }
        const ids = new Set(Object.values(ch.chapters).map((s) => s.id));
        scenes += ids.size;
        for (const scene of Object.values(ch.chapters)) {
          if (!scene.text || !Array.isArray(scene.choices) || scene.choices.length < 2) {
            errors.push(`Bad scene ${ch.id}/${scene.id}`);
          }
          for (const choice of scene.choices || []) {
            if (choice.ending) {
              if (!data.endings[choice.ending]) errors.push(`Unknown ending tier "${choice.ending}"`);
            } else if (!choice.next || !ids.has(choice.next)) {
              errors.push(`Broken next "${choice.next}" in ${ch.id}/${scene.id}`);
            }
          }
          if (scene.secret && !ids.has(scene.secret.next)) {
            errors.push(`Broken secret.next in ${ch.id}/${scene.id}`);
          }
        }
      }
    }
  }

  return { ok: errors.length === 0, errors, scenes };
}
