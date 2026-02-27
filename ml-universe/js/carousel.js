/* ================================================================
   CAROUSEL – Drag-to-scroll + keyboard navigation
   ================================================================ */

export function initCarousel() {
  const wrapper = document.querySelector('.carousel-wrapper');
  if (!wrapper) return;

  /* ---- drag-to-scroll (desktop) ---- */
  let isDown   = false;
  let startX   = 0;
  let scrollL  = 0;

  wrapper.addEventListener('mousedown', (e) => {
    isDown  = true;
    startX  = e.pageX - wrapper.offsetLeft;
    scrollL = wrapper.scrollLeft;
    wrapper.style.cursor = 'grabbing';
  }, { passive: true });

  wrapper.addEventListener('mouseleave', () => {
    isDown = false;
    wrapper.style.cursor = '';
  }, { passive: true });

  wrapper.addEventListener('mouseup', () => {
    isDown = false;
    wrapper.style.cursor = '';
  }, { passive: true });

  wrapper.addEventListener('mousemove', (e) => {
    if (!isDown) return;
    e.preventDefault();
    const x    = e.pageX - wrapper.offsetLeft;
    const walk = (x - startX) * 1.5;
    wrapper.scrollLeft = scrollL - walk;
  });

  /* ---- keyboard navigation ---- */
  const card = wrapper.querySelector('.algo-card');
  const cardWidth = card ? card.offsetWidth + 24 : 324; // card + gap

  document.addEventListener('keydown', (e) => {
    const sections = document.querySelector('#algorithms');
    if (!sections) return;

    const rect = sections.getBoundingClientRect();
    const inView = rect.top < window.innerHeight && rect.bottom > 0;
    if (!inView) return;

    if (e.key === 'ArrowRight') {
      wrapper.scrollBy({ left: cardWidth, behavior: 'smooth' });
    } else if (e.key === 'ArrowLeft') {
      wrapper.scrollBy({ left: -cardWidth, behavior: 'smooth' });
    }
  });
}
