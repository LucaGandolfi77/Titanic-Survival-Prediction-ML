// Toast notifications: SW update prompts, share feedback, records.
// Replaces alert() with a non-blocking, glassmorphism-styled toast.

let toastEl = null;
let hideTimer = 0;

export function showToast(message, { actionLabel = null, onAction = null, duration = 5000 } = {}) {
  if (!toastEl) {
    toastEl = document.createElement('div');
    toastEl.className = 'toast';
    toastEl.setAttribute('role', 'status');

    const text = document.createElement('span');
    text.className = 'toast-text';

    const action = document.createElement('button');
    action.type = 'button';
    action.className = 'toast-action';
    action.hidden = true;

    toastEl.append(text, action);
    document.body.appendChild(toastEl);
  }

  const text = toastEl.querySelector('.toast-text');
  const action = toastEl.querySelector('.toast-action');
  text.textContent = message;

  if (actionLabel && onAction) {
    action.textContent = actionLabel;
    action.hidden = false;
    action.onclick = () => {
      dismiss();
      onAction();
    };
  } else {
    action.hidden = true;
    action.onclick = null;
  }

  toastEl.classList.add('visible');
  clearTimeout(hideTimer);
  if (duration > 0) hideTimer = setTimeout(dismiss, duration);
}

export function dismiss() {
  clearTimeout(hideTimer);
  if (toastEl) toastEl.classList.remove('visible');
}
