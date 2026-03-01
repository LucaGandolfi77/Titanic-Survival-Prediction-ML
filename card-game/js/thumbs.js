/**
 * Thumbnails gallery — probe `assets/cards/thumbs` and `assets/cards/mini`
 * Render clickable tiles that map filenames to `card_###` ids and call `addToDeck`.
 */
import { addToDeck } from './state.js';
import { getCardById } from './database.js';
import { showToast } from './utils.js';

const pad = (n, width = 3) => String(n).padStart(width, '0');

const probe = async (url) => {
  try {
    const res = await fetch(url, { method: 'HEAD' });
    return res && res.ok;
  } catch (e) {
    return false;
  }
};

const extractNumber = (name) => {
  const m = name.match(/(\d{1,4})[^\d]*$/);
  if (!m) return null;
  const n = parseInt(m[1], 10);
  if (Number.isNaN(n)) return null;
  return n;
};

const mapToCardId = (filename) => {
  const n = extractNumber(filename);
  if (!n) return null;
  const id = `card_${pad(n, 3)}`;
  return id;
};

export const initThumbGallery = async (opts = {}) => {
  const gallery = document.getElementById('assets-gallery');
  if (!gallery) return;

  const candidates = [];
  // patterns observed in repo
  for (let i = 1; i <= 156; i++) candidates.push(`a1_${pad(i,4)}.jpg`);
  for (let i = 2; i <= 156; i++) {
    candidates.push(`a_${pad(i,4)}.jpg`);
    candidates.push(`a_${pad(i,4)}_1.jpg`);
  }
  for (let i = 2; i <= 155; i++) candidates.push(`page_${pad(i,4)}.jpg`);

  const baseThumb = 'assets/cards/thumbs/';
  const baseMini = 'assets/cards/mini/';

  for (const name of candidates) {
    const tUrl = baseThumb + name;
    if (await probe(tUrl)) {
      renderTile(gallery, tUrl, name);
      continue;
    }
    const mUrl = baseMini + name;
    if (await probe(mUrl)) renderTile(gallery, mUrl, name);
  }
};

const renderTile = (gallery, src, filename) => {
  const tile = document.createElement('div');
  tile.className = 'assets-gallery__tile';
  tile.tabIndex = 0;
  const img = document.createElement('img');
  img.src = src;
  img.alt = filename;
  img.loading = 'lazy';
  tile.appendChild(img);

  // map filename -> card id if possible
  const cardId = mapToCardId(filename);
  const card = cardId ? getCardById(cardId) : null;
  if (card) {
    tile.dataset.cardId = cardId;
    tile.title = `Add ${card.name} to deck`;
    tile.style.cursor = 'pointer';

    const handleAdd = () => {
      const ok = addToDeck(cardId);
      if (ok) showToast(`Added ${card.name}`, 'success');
      else showToast(`Can't add ${card.name} (deck full or max copies)`, 'warning');
    };

    tile.addEventListener('click', handleAdd);
    tile.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        handleAdd();
      }
    });
  } else {
    tile.title = filename;
  }

  gallery.appendChild(tile);
};

export default initThumbGallery;
