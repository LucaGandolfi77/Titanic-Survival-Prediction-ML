/* ================================================================
   SPLITTING.JS — Text splitting for char-by-char animation
   ================================================================ */

export function initSplitting() {
  if (typeof Splitting === 'undefined') {
    console.warn('Splitting.js not loaded, skipping text split.');
    return;
  }

  /* Split the hero title into individual chars */
  Splitting({ target: '[data-splitting]', by: 'chars' });
}
