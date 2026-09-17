// Two-layer sentiment scoring for player-written flirty lines:
// 1. Heuristic keyword scorer — instant, offline, zero dependencies.
// 2. Optional Transformers.js micro-model (explicit opt-in) — neural
//    inference running locally, hardware-accelerated via WebNN/WebGPU when
//    available. Zero server, zero API keys; cached for offline use.

const POSITIVE = [
  'love', 'lovely', 'beautiful', 'beautifully', 'stunning', 'gorgeous', 'kiss',
  'smile', 'favorite', 'favourite', 'magnetic', 'unforgettable', 'adorable',
  'bright', 'music', 'celebrate', 'amazing', 'incredible'
];
const FLIRTY = [
  'dangerous', 'trouble', 'spark', 'chemistry', 'attractive', 'distracting',
  'steal', 'jealous', 'reckless', 'bold', 'crush', 'flirt', 'kiss', 'trouble'
];
const NEGATIVE = ['boring', 'whatever', 'meh', 'nah', 'nope', 'lame', 'dull', 'hate', 'ignore'];

const HEDGE = /^(i |im |i'm )?(maybe|perhaps|probably|kind of|sort of)\b/i;

export function heuristicScore(text) {
  const clean = (text || '').toLowerCase();
  const words = clean.split(/\W+/).filter(Boolean);
  const hits = (list) => words.filter((w) => list.includes(w)).length;
  const pos = hits(POSITIVE);
  const flirty = hits(FLIRTY);
  const neg = hits(NEGATIVE);
  const energy =
    (clean.match(/!/g) || []).length + (clean.match(/[\u2764\u2728\ud83d\ude0d]/u) || []).length;
  const hedged = HEDGE.test(clean);

  const raw = pos * 1.2 + flirty * 1.5 + energy * 0.4 - neg * 1.8 - (hedged ? 0.5 : 0);

  if (raw >= 2.2) return { tone: 'good', score: 4, engine: 'heuristic', confidence: Math.min(1, raw / 4) };
  if (raw >= 1) return { tone: 'safe', score: 3, engine: 'heuristic', confidence: Math.min(1, raw / 3) };
  if (raw < 0) return { tone: 'risky', score: 2, engine: 'heuristic', confidence: Math.min(1, -raw / 3) };
  return { tone: 'safe', score: 2, engine: 'heuristic', confidence: 0.3 };
}

/* ---- optional Transformers.js layer (lazy, opt-in) ---- */

const TRANSFORMERS_URL = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3';
const SENTIMENT_MODEL = 'Xenova/distilbert-base-uncased-finetuned-sst-2-english';

let classifierPromise = null;

export function isNeuralReady() {
  return classifierPromise !== null;
}

/**
 * Lazy-load Transformers.js + the sentiment model, trying the best device
 * first (WebNN → WebGPU → WASM). Only called on explicit opt-in.
 */
export async function ensureNeural(caps, onProgress) {
  if (classifierPromise) return classifierPromise;
  classifierPromise = (async () => {
    const mod = await import(TRANSFORMERS_URL);
    const devices = [caps?.webnn ? 'webnn' : null, caps?.webgpu ? 'webgpu' : null, 'wasm'].filter(
      Boolean
    );
    let lastError = null;
    for (const device of devices) {
      try {
        return await mod.pipeline('sentiment-analysis', SENTIMENT_MODEL, {
          device,
          progress_callback: onProgress
        });
      } catch (err) {
        lastError = err;
      }
    }
    throw lastError || new Error('No inference backend available');
  })();
  try {
    return await classifierPromise;
  } catch (err) {
    classifierPromise = null; // allow a retry later
    throw err;
  }
}

export async function neuralScore(text) {
  const classifier = await classifierPromise;
  if (!classifier) throw new Error('Neural model not loaded');
  const [result] = await classifier(text);
  const positive = result.label === 'POSITIVE';
  const confidence = result.score || 0;
  if (positive && confidence > 0.93) return { tone: 'good', score: 4, engine: 'neural', confidence };
  if (positive) return { tone: 'safe', score: 3, engine: 'neural', confidence };
  return { tone: 'risky', score: 2, engine: 'neural', confidence };
}

/** Score a line with the best available engine (neural → heuristic). */
export async function scoreLine(text) {
  if (isNeuralReady()) {
    try {
      return await neuralScore(text);
    } catch {
      /* fall through to the heuristic */
    }
  }
  return heuristicScore(text);
}
