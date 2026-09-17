export async function shareModel(model) {
  const layers = model.layers.map((l) => `${l.type}(${l.size})`).join(' → ');
  const summary = `🧠 NN Visualizer\n\nArchitecture: ${layers}\n\nParameters: ${countParameters(model).toLocaleString()}`;

  if (navigator.share) {
    try {
      await navigator.share({
        title: 'My Neural Network',
        text: summary,
      });
      return true;
    } catch (err) {
      console.warn('Share cancelled:', err);
      return false;
    }
  }

  copyToClipboard(summary);
  return true;
}

function countParameters(model) {
  let total = 0;
  for (let L = 1; L < model.layers.length; L++) {
    const W = model.weights[L - 1];
    const B = model.biases[L - 1];
    if (W) total += W.flat().length;
    if (B) total += B.length;
  }
  return total;
}

function copyToClipboard(text) {
  if (navigator.clipboard) {
    navigator.clipboard.writeText(text).then(() => {
      showToast('Architecture copied to clipboard!');
    });
  }
}

function showToast(message) {
  const toast = document.createElement('div');
  toast.textContent = message;
  toast.style.cssText = 'position:fixed;bottom:80px;left:50%;transform:translateX(-50%);background:var(--accent);color:#fff;padding:10px 20px;border-radius:8px;font-size:13px;z-index:10000;animation:slideUp 0.3s ease';
  document.body.appendChild(toast);
  setTimeout(() => toast.remove(), 2500);
}
