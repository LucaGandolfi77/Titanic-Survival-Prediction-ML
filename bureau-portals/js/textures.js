import * as THREE from 'three';

function createCanvas(size = 256) {
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  return { canvas, ctx: canvas.getContext('2d') };
}

function toTexture(canvas, repeatX = 1, repeatY = 1) {
  const texture = new THREE.CanvasTexture(canvas);
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.RepeatWrapping;
  texture.repeat.set(repeatX, repeatY);
  texture.needsUpdate = true;
  return texture;
}

export function generateWoodTexture(size = 256, baseColor = '#8b6914', darkColor = '#5a4210') {
  const { canvas, ctx } = createCanvas(size);
  ctx.fillStyle = baseColor;
  ctx.fillRect(0, 0, size, size);

  for (let i = 0; i < 60; i++) {
    const x = Math.random() * size;
    const width = 1 + Math.random() * 3;
    const alpha = 0.1 + Math.random() * 0.3;
    ctx.strokeStyle = darkColor;
    ctx.globalAlpha = alpha;
    ctx.lineWidth = width;
    ctx.beginPath();
    const waveAmp = 2 + Math.random() * 4;
    const waveFreq = 0.02 + Math.random() * 0.03;
    ctx.moveTo(x, 0);
    for (let y = 0; y < size; y += 4) {
      ctx.lineTo(x + Math.sin(y * waveFreq + i) * waveAmp, y);
    }
    ctx.stroke();
  }
  ctx.globalAlpha = 1;
  return toTexture(canvas);
}

export function generateConcreteTexture(size = 256, baseColor = '#8a8a8a', spots = 200) {
  const { canvas, ctx } = createCanvas(size);
  ctx.fillStyle = baseColor;
  ctx.fillRect(0, 0, size, size);

  for (let i = 0; i < spots; i++) {
    const x = Math.random() * size;
    const y = Math.random() * size;
    const r = 1 + Math.random() * 4;
    const alpha = 0.05 + Math.random() * 0.15;
    const shade = Math.random() > 0.5 ? 0 : 255;
    ctx.fillStyle = `rgba(${shade},${shade},${shade},${alpha})`;
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fill();
  }
  return toTexture(canvas);
}

export function generateMetalTexture(size = 256, baseColor = '#7a7a7a', lines = 40) {
  const { canvas, ctx } = createCanvas(size);
  ctx.fillStyle = baseColor;
  ctx.fillRect(0, 0, size, size);

  for (let i = 0; i < lines; i++) {
    const y = (i / lines) * size + Math.random() * 2;
    const alpha = 0.05 + Math.random() * 0.1;
    ctx.strokeStyle = `rgba(255,255,255,${alpha})`;
    ctx.lineWidth = 0.5 + Math.random();
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(size, y + Math.random() * 2 - 1);
    ctx.stroke();
  }
  for (let i = 0; i < 50; i++) {
    const x = Math.random() * size;
    const y = Math.random() * size;
    const alpha = 0.05 + Math.random() * 0.1;
    ctx.fillStyle = `rgba(0,0,0,${alpha})`;
    ctx.fillRect(x, y, 2 + Math.random() * 3, 2 + Math.random() * 3);
  }
  return toTexture(canvas);
}

export function generateTileTexture(size = 256, baseColor = '#6a6a6a', grout = '#3a3a3a') {
  const { canvas, ctx } = createCanvas(size);
  ctx.fillStyle = grout;
  ctx.fillRect(0, 0, size, size);

  const tileSize = size / 4;
  ctx.fillStyle = baseColor;
  for (let x = 0; x < 4; x++) {
    for (let y = 0; y < 4; y++) {
      ctx.fillRect(x * tileSize + 2, y * tileSize + 2, tileSize - 4, tileSize - 4);
    }
  }
  for (let i = 0; i < 100; i++) {
    const x = Math.random() * size;
    const y = Math.random() * size;
    const alpha = 0.05 + Math.random() * 0.1;
    ctx.fillStyle = `rgba(255,255,255,${alpha})`;
    ctx.fillRect(x, y, 1 + Math.random() * 2, 1 + Math.random() * 2);
  }
  return toTexture(canvas);
}

export function generateCarpetTexture(size = 256, baseColor = '#3a3a5a') {
  const { canvas, ctx } = createCanvas(size);
  ctx.fillStyle = baseColor;
  ctx.fillRect(0, 0, size, size);

  const imageData = ctx.getImageData(0, 0, size, size);
  const data = imageData.data;
  for (let i = 0; i < data.length; i += 4) {
    const noise = (Math.random() - 0.5) * 20;
    data[i] = Math.max(0, Math.min(255, data[i] + noise));
    data[i + 1] = Math.max(0, Math.min(255, data[i + 1] + noise));
    data[i + 2] = Math.max(0, Math.min(255, data[i + 2] + noise));
  }
  ctx.putImageData(imageData, 0, 0);
  return toTexture(canvas);
}

export function generateMarbleTexture(size = 256, baseColor = '#d8d8d0') {
  const { canvas, ctx } = createCanvas(size);
  ctx.fillStyle = baseColor;
  ctx.fillRect(0, 0, size, size);

  for (let i = 0; i < 15; i++) {
    ctx.strokeStyle = `rgba(100,100,100,${0.1 + Math.random() * 0.2})`;
    ctx.lineWidth = 0.5 + Math.random() * 2;
    ctx.beginPath();
    const startX = Math.random() * size;
    const startY = Math.random() * size;
    ctx.moveTo(startX, startY);
    let x = startX;
    let y = startY;
    for (let j = 0; j < 20; j++) {
      x += (Math.random() - 0.5) * 30;
      y += (Math.random() - 0.5) * 30;
      ctx.lineTo(x, y);
    }
    ctx.stroke();
  }
  return toTexture(canvas);
}

export function generateNormalMapNoise(size = 256, strength = 1) {
  const { canvas, ctx } = createCanvas(size);
  const imageData = ctx.createImageData(size, size);
  const data = imageData.data;
  for (let i = 0; i < data.length; i += 4) {
    const n = (Math.random() - 0.5) * 255 * strength;
    data[i] = 128 + n;
    data[i + 1] = 128 + n;
    data[i + 2] = 255;
    data[i + 3] = 255;
  }
  ctx.putImageData(imageData, 0, 0);
  return toTexture(canvas);
}

export function createTextureAtlas(textureFns, cols, rows, size = 256) {
  const canvas = document.createElement('canvas');
  canvas.width = size * cols;
  canvas.height = size * rows;
  const ctx = canvas.getContext('2d');

  for (let i = 0; i < textureFns.length; i++) {
    const col = i % cols;
    const row = Math.floor(i / cols);
    const tex = textureFns[i]();
    ctx.drawImage(tex.image, col * size, row * size, size, size);
  }
  return toTexture(canvas);
}

export class TextureManager {
  constructor() {
    this.textures = {};
    this.generated = false;
  }

  generate() {
    if (this.generated) return;
    this.textures.wood = generateWoodTexture(256, '#8b6914', '#5a4210');
    this.textures.woodDark = generateWoodTexture(256, '#5a3a2a', '#3a2a1a');
    this.textures.concrete = generateConcreteTexture(256, '#8a8a8a');
    this.textures.metal = generateMetalTexture(256, '#7a7a7a');
    this.textures.tile = generateTileTexture(256, '#6a6a6a');
    this.textures.carpet = generateCarpetTexture(256, '#3a3a5a');
    this.textures.marble = generateMarbleTexture(256, '#d8d8d0');
    this.textures.normal = generateNormalMapNoise(256, 0.5);
    this.generated = true;
  }

  get(name) {
    this.generate();
    return this.textures[name] || null;
  }

  dispose() {
    Object.values(this.textures).forEach((t) => t.dispose());
    this.textures = {};
    this.generated = false;
  }
}
