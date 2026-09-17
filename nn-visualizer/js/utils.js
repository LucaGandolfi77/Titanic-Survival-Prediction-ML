const CACHED_NODES = new Map();

export function getOrCacheNode(key, createFn) {
  if (CACHED_NODES.has(key)) return CACHED_NODES.get(key);
  const node = createFn();
  CACHED_NODES.set(key, node);
  return node;
}

export function clearNodeCache() {
  CACHED_NODES.clear();
}

export function createDomElement(tag, className = '', attrs = {}) {
  const el = document.createElement(tag);
  if (className) el.className = className;
  for (const [key, value] of Object.entries(attrs)) {
    el.setAttribute(key, value);
  }
  return el;
}

export function setText(el, text) {
  el.textContent = text;
  return el;
}
