// Story trailer: an animated recap of the playthrough (choice highlights +
// ending) rendered on a canvas and recorded via MediaRecorder → a shareable
// WebM video (Web Share with files, download fallback).

import { showToast } from '../../ui/toast.js';

export function isSupported() {
  return (
    typeof window !== 'undefined' &&
    typeof window.MediaRecorder === 'function' &&
    typeof HTMLCanvasElement !== 'undefined' &&
    typeof HTMLCanvasElement.prototype.captureStream === 'function'
  );
}

function pickMime() {
  const candidates = ['video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm', 'video/mp4'];
  for (const mime of candidates) {
    if (MediaRecorder.isTypeSupported?.(mime)) return mime;
  }
  return '';
}

function buildFrames({ ending, scenes }) {
  const frames = [{ kind: 'title', text: 'Speed Crush', sub: 'A flirty story recap', tone: 'risky' }];
  const highlights = (scenes || []).filter((s) => s.choiceText).slice(0, 4);
  for (const scene of highlights) {
    frames.push({
      kind: 'scene',
      text: scene.choiceText,
      sub: `Chapter ${scene.chapter}${scene.fast ? ' · fast' : ''}${scene.secret ? ' · ✨' : ''}`,
      tone: scene.tone
    });
  }
  frames.push({
    kind: 'ending',
    text: ending?.title || 'The end',
    sub: ending?.badge || '',
    tone: 'ending'
  });
  return frames;
}

function wrapText(ctx, text, x, y, maxWidth, lineHeight) {
  const words = (text || '').split(' ');
  const lines = [];
  let line = '';
  for (const word of words) {
    const test = line ? `${line} ${word}` : word;
    if (ctx.measureText(test).width > maxWidth && line) {
      lines.push(line);
      line = word;
    } else {
      line = test;
    }
  }
  if (line) lines.push(line);
  let yy = y - ((lines.length - 1) * lineHeight) / 2;
  for (const l of lines) {
    ctx.fillText(l, x, yy);
    yy += lineHeight;
  }
}

function drawFrame(ctx, canvas, frame) {
  return new Promise((resolve) => {
    const start = performance.now();
    const duration = 900;
    const toneColors = { good: '#8cff77', safe: '#4df3ff', risky: '#ff4fa3', ending: '#ffd44d' };
    const color = toneColors[frame.tone] || '#ff4fa3';
    const size = frame.kind === 'title' ? 84 : frame.kind === 'ending' ? 56 : 44;

    const render = () => {
      const progress = Math.min(1, (performance.now() - start) / duration);
      const ease = 1 - Math.pow(1 - progress, 3);

      const grad = ctx.createLinearGradient(0, 0, canvas.width, canvas.height);
      grad.addColorStop(0, '#180b33');
      grad.addColorStop(1, '#2b0f5f');
      ctx.fillStyle = grad;
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      ctx.globalAlpha = 0.25 * ease;
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(canvas.width / 2, canvas.height / 2, 260 + 40 * ease, 0, Math.PI * 2);
      ctx.fill();
      ctx.globalAlpha = 1;

      ctx.textAlign = 'center';
      ctx.fillStyle = '#ffffff';
      ctx.font = `800 ${Math.round(size * (0.6 + 0.4 * ease))}px Inter, sans-serif`;
      wrapText(ctx, frame.text, canvas.width / 2, canvas.height / 2 - 20, canvas.width - 120, size * 1.2);

      ctx.fillStyle = color;
      ctx.font = `400 ${Math.round(28 * ease)}px Inter, sans-serif`;
      if (frame.sub) ctx.fillText(frame.sub.toUpperCase(), canvas.width / 2, canvas.height / 2 + 90);

      if (progress < 1) requestAnimationFrame(render);
      else resolve();
    };
    requestAnimationFrame(render);
  });
}

async function recordTrailer({ ending, scenes }) {
  const canvas = document.createElement('canvas');
  canvas.width = 720;
  canvas.height = 1280;
  canvas.style.position = 'fixed';
  canvas.style.left = '-9999px';
  document.body.appendChild(canvas);
  const ctx = canvas.getContext('2d');

  const stream = canvas.captureStream(30);
  const chunks = [];
  const recorder = new MediaRecorder(stream, { mimeType: pickMime() });
  recorder.ondataavailable = (event) => {
    if (event.data.size) chunks.push(event.data);
  };
  const done = new Promise((resolve) => {
    recorder.onstop = () => resolve(new Blob(chunks, { type: recorder.mimeType }));
  });

  recorder.start();
  for (const frame of buildFrames({ ending, scenes })) {
    await drawFrame(ctx, canvas, frame);
  }
  recorder.stop();
  stream.getTracks().forEach((track) => track.stop());
  canvas.remove();
  return done;
}

/** Record + share (or download) the playthrough trailer. */
export async function shareTrailer({ ending, scenes }) {
  if (!isSupported()) {
    showToast('Video recording is not supported here.', { duration: 3500 });
    return false;
  }
  showToast('Recording trailer… 🎬', { duration: 2000 });
  try {
    const blob = await recordTrailer({ ending, scenes });
    const file = new File([blob], 'speed-crush-trailer.webm', { type: blob.type });

    if (navigator.canShare?.({ files: [file] })) {
      try {
        await navigator.share({ files: [file], title: 'Speed Crush', text: 'My playthrough trailer 💘' });
        return true;
      } catch {
        /* user cancelled → download instead */
      }
    }

    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'speed-crush-trailer.webm';
    a.click();
    URL.revokeObjectURL(url);
    showToast('Trailer downloaded ✅', { duration: 3000 });
    return true;
  } catch (err) {
    console.error('Trailer recording failed', err);
    showToast('Could not record the trailer.', { duration: 3500 });
    return false;
  }
}
