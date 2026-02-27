/* ================================================================
   TIMELINE – SVG line-draw + card reveal
   ================================================================ */

export function initTimeline() {
  if (typeof gsap === 'undefined' || typeof ScrollTrigger === 'undefined') return;

  const svg  = document.querySelector('#timeline-svg');
  const path = svg?.querySelector('path');
  if (!path) return;

  /* ---- stroke-dashoffset draw animation ---- */
  const totalLength = path.getTotalLength();
  gsap.set(path, {
    strokeDasharray: totalLength,
    strokeDashoffset: totalLength,
  });

  gsap.to(path, {
    strokeDashoffset: 0,
    ease: 'none',
    scrollTrigger: {
      trigger: '#timeline',
      start: 'top 60%',
      end: 'bottom 40%',
      scrub: 1,
    },
  });

  /* ---- timeline cards alternate from left / right ---- */
  const steps = gsap.utils.toArray('.timeline-step');
  steps.forEach((step, i) => {
    const card = step.querySelector('.timeline-card');
    if (!card) return;

    const fromX = i % 2 === 0 ? -50 : 50;

    gsap.fromTo(card,
      { opacity: 0, x: fromX },
      {
        opacity: 1,
        x: 0,
        duration: 0.8,
        ease: 'power3.out',
        scrollTrigger: {
          trigger: step,
          start: 'top 80%',
          toggleActions: 'play none none none',
        },
      }
    );

    /* light up dot */
    const dot = step.querySelector('.timeline-dot');
    if (dot) {
      gsap.fromTo(dot,
        { scale: 0 },
        {
          scale: 1,
          duration: 0.4,
          ease: 'back.out(2)',
          scrollTrigger: {
            trigger: step,
            start: 'top 80%',
            toggleActions: 'play none none none',
          },
        }
      );
    }
  });
}
