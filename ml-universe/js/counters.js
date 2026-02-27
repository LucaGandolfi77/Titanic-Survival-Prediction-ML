/* ================================================================
   ANIMATED STATS COUNTERS
   ================================================================ */

export function initCounters() {
  if (typeof gsap === 'undefined' || typeof ScrollTrigger === 'undefined') return;
  if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;

  const counters = [
    { el: '#stat-params', end: 175000000000, format: 'number',   suffix: '' },
    { el: '#stat-perf',   end: 86,           format: 'percent',  suffix: '%' },
    { el: '#stat-acc',    end: 96.7,         format: 'decimal',  suffix: '%' },
    { el: '#stat-market', end: 200,          format: 'currency', suffix: 'B' },
  ];

  counters.forEach(({ el, end, format, suffix }) => {
    const element = document.querySelector(el);
    if (!element) return;

    ScrollTrigger.create({
      trigger: element,
      start: 'top 85%',
      once: true,
      onEnter: () => {
        const obj = { val: 0 };
        gsap.to(obj, {
          val: end,
          duration: 2.5,
          ease: 'power2.out',
          onUpdate() {
            element.textContent = formatNumber(obj.val, format) + suffix;
          },
        });
      },
    });
  });
}

function formatNumber(n, format) {
  switch (format) {
    case 'number':   return Math.round(n).toLocaleString('en-US');
    case 'percent':  return String(Math.round(n));
    case 'decimal':  return n.toFixed(1);
    case 'currency': return '$' + Math.round(n);
    default:         return String(Math.round(n));
  }
}
