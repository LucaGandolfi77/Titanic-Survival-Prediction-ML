/* ================================================================
   MAIN – Entry point · imports all modules
   ================================================================ */

import { initLenis }           from './lenis.js';
import { initNavbar }          from './navbar.js';
import { initThreeBackground } from './three-bg.js';
import { initSplitting }       from './splitting.js';
import { initGsapAnimations }  from './gsap-animations.js';
import { initCounters }        from './counters.js';
import { initTimeline }        from './timeline.js';
import { initCarousel }        from './carousel.js';
import { initForm }            from './form.js';

document.addEventListener('DOMContentLoaded', async () => {
  /* smooth scroll first — everything else depends on it */
  const lenis = initLenis();

  /* navbar (needs lenis for smooth anchor scroll) */
  initNavbar(lenis);

  /* Three.js particle background */
  initThreeBackground();

  /* Splitting.js text chars */
  initSplitting();

  /* GSAP scroll-driven animations (depends on lenis + ScrollTrigger) */
  initGsapAnimations(lenis);

  /* animated counters in stats section */
  initCounters();

  /* timeline SVG draw + card reveals */
  initTimeline();

  /* algorithm carousel drag & keyboard */
  initCarousel();

  /* newsletter form */
  initForm();

  /* signal ready to CSS / tests */
  document.documentElement.classList.add('js-ready');
});
