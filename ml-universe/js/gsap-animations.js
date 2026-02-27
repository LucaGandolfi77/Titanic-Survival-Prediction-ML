/* ================================================================
   GSAP + ScrollTrigger — All Scroll Animations
   ================================================================ */

export function initGsapAnimations(lenis) {
  if (typeof gsap === 'undefined' || typeof ScrollTrigger === 'undefined') {
    console.warn('GSAP/ScrollTrigger not loaded, skipping animations.');
    return;
  }

  /* Respect reduced motion */
  if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;

  gsap.registerPlugin(ScrollTrigger);
  ScrollTrigger.defaults({ markers: false });

  /* Connect Lenis to ScrollTrigger */
  if (lenis) {
    lenis.on('scroll', ScrollTrigger.update);
  }

  /* ── 1. Progress bar ────────────────────────────────────── */
  gsap.to('#progress-bar', {
    width: '100%',
    ease: 'none',
    scrollTrigger: {
      scrub: 0.3,
      start: 'top top',
      end: 'bottom bottom',
    },
  });

  /* ── 2. Hero chars (Splitting.js → .char spans) ─────────── */
  const heroChars = document.querySelectorAll('.hero-title .char');
  if (heroChars.length) {
    gsap.fromTo(heroChars,
      { opacity: 0, y: 40, rotateX: -90 },
      {
        opacity: 1, y: 0, rotateX: 0,
        duration: 0.8,
        stagger: 0.03,
        ease: 'back.out(1.7)',
        delay: 0.3,
      }
    );
  }

  /* ── 3. Hero subtitle + CTAs + scroll indicator + stats ── */
  gsap.fromTo(
    ['.hero-subtitle', '.hero-ctas', '.scroll-indicator', '.stats-bar'],
    { opacity: 0, y: 30 },
    {
      opacity: 1, y: 0,
      duration: 0.8,
      stagger: 0.15,
      delay: 1.0,
      ease: 'power2.out',
    }
  );

  /* ── 4. Section headings: clip-path reveal ──────────────── */
  gsap.utils.toArray('.section-title').forEach((title) => {
    gsap.fromTo(title,
      { clipPath: 'inset(100% 0 0 0)' },
      {
        clipPath: 'inset(0% 0 0 0)',
        duration: 1,
        ease: 'power4.out',
        scrollTrigger: { trigger: title, start: 'top 85%' },
      }
    );
  });

  /* ── 5. About text paragraphs ───────────────────────────── */
  gsap.fromTo('#about .fade-in',
    { opacity: 0, y: 40 },
    {
      opacity: 1, y: 0,
      stagger: 0.2,
      duration: 0.8,
      ease: 'power2.out',
      scrollTrigger: { trigger: '#about', start: 'top 70%' },
    }
  );

  /* ── 6. About visual (training loop SVG) ────────────────── */
  gsap.fromTo('.about-visual',
    { opacity: 0, scale: 0.9 },
    {
      opacity: 1, scale: 1,
      duration: 1,
      ease: 'power2.out',
      scrollTrigger: { trigger: '.about-visual', start: 'top 80%' },
    }
  );

  /* ── 7. Algorithm carousel cards ────────────────────────── */
  gsap.fromTo('.algo-card',
    { opacity: 0, x: 60 },
    {
      opacity: 1, x: 0,
      stagger: 0.1,
      duration: 0.7,
      ease: 'power2.out',
      scrollTrigger: { trigger: '#algorithms', start: 'top 80%' },
    }
  );

  /* ── 8. Featured cards ──────────────────────────────────── */
  gsap.fromTo('.featured-card',
    { opacity: 0, y: 50 },
    {
      opacity: 1, y: 0,
      stagger: 0.15,
      duration: 0.8,
      ease: 'power3.out',
      scrollTrigger: { trigger: '#featured', start: 'top 75%' },
    }
  );

  /* ── 9. Bento grid cards ────────────────────────────────── */
  gsap.fromTo('.bento-card',
    { opacity: 0, scale: 0.95, y: 20 },
    {
      opacity: 1, scale: 1, y: 0,
      stagger: 0.08,
      duration: 0.6,
      ease: 'power2.out',
      scrollTrigger: { trigger: '#applications', start: 'top 80%' },
    }
  );

  /* ── 10. Tools section header ───────────────────────────── */
  gsap.fromTo('#tools .section-header',
    { opacity: 0, y: 30 },
    {
      opacity: 1, y: 0,
      duration: 0.8,
      scrollTrigger: { trigger: '#tools', start: 'top 85%' },
    }
  );

  /* ── 11. CTA content ────────────────────────────────────── */
  gsap.fromTo('.cta-content',
    { opacity: 0, y: 40 },
    {
      opacity: 1, y: 0,
      duration: 1,
      ease: 'power2.out',
      scrollTrigger: { trigger: '#cta', start: 'top 80%' },
    }
  );
}
