/* ===== Toasts & modals ===== */

let toastCounter = 0;

export function showToast(message, type = 'info', duration = 3500) {
  const container = document.getElementById('toast-container');
  const toast = document.createElement('div');
  toast.className = `toast toast-${type}`;
  toast.id = `toast-${++toastCounter}`;
  toast.innerHTML = `<span>${message}</span>`;
  container.appendChild(toast);
  setTimeout(() => {
    toast.classList.add('out');
    setTimeout(() => toast.remove(), 400);
  }, duration);
}

export function showModal(html, options = {}) {
  const overlay = document.getElementById('modal-overlay');
  const content = document.getElementById('modal-content');
  content.innerHTML = html;
  overlay.classList.remove('hidden');
  overlay.classList.add('visible');

  // Close on overlay click (unless persistent)
  if (!options.persistent) {
    overlay.onclick = (e) => {
      if (e.target === overlay) hideModal();
    };
  }
}

export function hideModal() {
  const overlay = document.getElementById('modal-overlay');
  overlay.classList.remove('visible');
  overlay.classList.add('hidden');
  overlay.onclick = null;
}

export function confirmModal(title, message, onConfirm, onCancel) {
  const html = `
    <h3 class="modal-title">${title}</h3>
    <p class="modal-message">${message}</p>
    <div class="modal-buttons">
      <button class="modal-btn confirm" id="modal-confirm">✔ Confirm</button>
      <button class="modal-btn cancel" id="modal-cancel">✖ Cancel</button>
    </div>
  `;
  showModal(html, { persistent: true });
  document.getElementById('modal-confirm').addEventListener('click', () => {
    hideModal();
    if (onConfirm) onConfirm();
  });
  document.getElementById('modal-cancel').addEventListener('click', () => {
    hideModal();
    if (onCancel) onCancel();
  });
}
