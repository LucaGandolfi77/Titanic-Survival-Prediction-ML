/* ================================================================
   NEWSLETTER FORM – ripple · validation · success animation
   ================================================================ */

export function initForm() {
  const form    = document.querySelector('#newsletter-form');
  const input   = form?.querySelector('input[type="email"]');
  const btn     = form?.querySelector('button[type="submit"]');
  const success = form?.closest('.cta-content')?.querySelector('.success-msg');
  if (!form || !input || !btn) return;

  /* ---- ripple on button click ---- */
  btn.addEventListener('click', (e) => {
    const ripple = document.createElement('span');
    ripple.classList.add('ripple');
    const rect = btn.getBoundingClientRect();
    ripple.style.left = `${e.clientX - rect.left}px`;
    ripple.style.top  = `${e.clientY - rect.top}px`;
    btn.appendChild(ripple);
    ripple.addEventListener('animationend', () => ripple.remove());
  });

  /* ---- submit handler ---- */
  form.addEventListener('submit', (e) => {
    e.preventDefault();

    const email = input.value.trim();
    if (!isValidEmail(email)) {
      input.classList.add('input-error');
      input.focus();
      setTimeout(() => input.classList.remove('input-error'), 1200);
      return;
    }

    btn.disabled = true;
    btn.textContent = '';

    /* draw checkmark SVG */
    const svgNS = 'http://www.w3.org/2000/svg';
    const svg   = document.createElementNS(svgNS, 'svg');
    svg.setAttribute('class', 'check-svg');
    svg.setAttribute('viewBox', '0 0 24 24');
    svg.setAttribute('width', '22');
    svg.setAttribute('height', '22');
    const path = document.createElementNS(svgNS, 'path');
    path.setAttribute('class', 'check-path');
    path.setAttribute('d', 'M4 12l5 5L20 7');
    path.setAttribute('fill', 'none');
    path.setAttribute('stroke', 'currentColor');
    path.setAttribute('stroke-width', '2.5');
    path.setAttribute('stroke-linecap', 'round');
    path.setAttribute('stroke-linejoin', 'round');
    svg.appendChild(path);
    btn.appendChild(svg);

    /* show success message */
    if (success) {
      setTimeout(() => {
        success.classList.add('visible');
      }, 900);
    }

    /* reset after 4s */
    setTimeout(() => {
      btn.disabled = false;
      btn.textContent = 'Iscriviti';
      input.value = '';
      if (success) success.classList.remove('visible');
    }, 4000);
  });
}

function isValidEmail(email) {
  return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);
}
