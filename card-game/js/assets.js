/**
 * Asset scanner — detect which card image files exist under assets/cards
 * Provides a small utility to let the UI offer only actually-present files.
 * @module assets
 */

const ASSET_DIR = "assets/cards/";
const EXTS = [".jpg", ".jpeg", ".png", ".webp"];

const pad = (n) => String(n).padStart(3, "0");

const probe = async (url) => {
  try {
    // Try HEAD first for speed; fallback to GET if server doesn't allow HEAD
    let res = await fetch(url, { method: "HEAD" });
    if (res && res.ok) return true;
    res = await fetch(url, { method: "GET" });
    return res && res.ok;
  } catch (e) {
    return false;
  }
};

/**
 * List available card image files for the canonical ids (card_001..card_050).
 * @returns {Promise<string[]>} array of relative paths (e.g. assets/cards/card_001.jpg)
 */
export const listAvailableImages = async () => {
  const found = [];
  for (let i = 1; i <= 50; i++) {
    const id = `card_${pad(i)}`;
    for (const ext of EXTS) {
      const p = `${ASSET_DIR}${id}${ext}`;
      // eslint-disable-next-line no-await-in-loop
      if (await probe(p)) {
        found.push(p);
        break;
      }
    }
  }
  return found;
};

/**
 * Return the first available image path for a specific card id, or null.
 * @param {string} cardId
 * @returns {Promise<string|null>}
 */
export const getImageForCard = async (cardId) => {
  for (const ext of EXTS) {
    const p = `${ASSET_DIR}${cardId}${ext}`;
    if (await probe(p)) return p;
  }
  return null;
};
