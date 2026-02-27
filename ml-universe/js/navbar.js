/* ================================================================
   NAVBAR — Scroll behavior + Hamburger + Active section
   ================================================================ */

export function initNavbar(lenis) {
  const header = document.querySelector('.site-header');
  const hamburger = document.querySelector('.hamburger');
  const navOverlay = document.querySelector('.nav-overlay');
  const navLinks = document.querySelectorAll('.nav-link');
  const sections = document.querySelectorAll('section[id]');

  if (!header) return;

  /* ── Scroll: shrink navbar ────────────────────────────────── */
  window.addEventListener('scroll', () => {
    header.classList.toggle('navbar--scrolled', window.scrollY > 50);
  }, { passive: true });

  /* ── Hamburger toggle ─────────────────────────────────────── */
  if (hamburger && navOverlay) {
    hamburger.addEventListener('click', () => {
      const isOpen = hamburger.classList.toggle('open');
      navOverlay.classList.toggle('open', isOpen);
      document.body.style.overflow = isOpen ? 'hidden' : '';
    });

    /* Close on link click */
    navOverlay.querySelectorAll('a').forEach((link) => {
      link.addEventListener('click', () => {
        hamburger.classList.remove('open');
        navOverlay.classList.remove('open');
        document.body.style.overflow = '';
      });
    });

    /* Close on Escape */
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && navOverlay.classList.contains('open')) {
        hamburger.classList.remove('open');
        navOverlay.classList.remove('open');
        document.body.style.overflow = '';
      }
    });
  }

  /* ── Active section via IntersectionObserver ───────────────── */
  if (sections.length && navLinks.length) {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            navLinks.forEach((l) => l.classList.remove('active'));
            const activeLink = document.querySelector(
              `.nav-link[href="#${entry.target.id}"]`
            );
            if (activeLink) activeLink.classList.add('active');
          }
        });
      },
      { threshold: 0.3, rootMargin: '-80px 0px 0px 0px' }
    );

    sections.forEach((s) => observer.observe(s));
  }

  /* ── Smooth scroll for nav links (Lenis) ──────────────────── */
  document.querySelectorAll('a[href^="#"]').forEach((anchor) => {
    anchor.addEventListener('click', (e) => {
      const targetId = anchor.getAttribute('href');
      if (targetId === '#') return;
      const target = document.querySelector(targetId);
      if (target && lenis) {
        e.preventDefault();
        lenis.scrollTo(target, { offset: -70 });
      }
    });
  });
}
