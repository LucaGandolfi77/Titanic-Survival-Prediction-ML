import * as THREE from 'three';

export class NonEuclideanEffects {
  constructor(renderer, scene) {
    this.renderer = renderer;
    this.scene = scene;
    this.sanity = 100;

    // Create fullscreen distortion canvas
    this.distortionCanvas = document.getElementById('screen-distortion');
    this.createSanityShader();
  }

  createSanityShader() {
    const canvas = document.createElement('canvas');
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;

    this.distortionCanvas.appendChild(canvas);
    this.ctx = canvas.getContext('2d');
    this.distortionTime = 0;

    const scale = 4;
    this._noiseScale = scale;
    this._noiseW = Math.floor(canvas.width / scale);
    this._noiseH = Math.floor(canvas.height / scale);
    this._noiseCanvas = document.createElement('canvas');
    this._noiseCanvas.width = this._noiseW;
    this._noiseCanvas.height = this._noiseH;
    this._noiseCtx = this._noiseCanvas.getContext('2d');
    this._noiseImageData = this._noiseCtx.createImageData(this._noiseW, this._noiseH);
    this._noiseData = this._noiseImageData.data;
    this._noiseBuffer = new Float32Array(this._noiseData.length / 4);
    for (let i = 0; i < this._noiseBuffer.length; i++) {
      this._noiseBuffer[i] = Math.random();
    }
  }

  updateSanity(value) {
    this.sanity = Math.max(-100, Math.min(100, value));
  }

  update(dt) {
    this.distortionTime += dt;
    this.applyDistortions();
  }

  applyDistortions() {
    const intensity = 1 - this.sanity / 100;

    if (intensity <= 0) return;

    const canvas = this.distortionCanvas.children[0];
    if (!canvas) return;

    // Screen distortion effects based on sanity
    if (this.sanity < 50) {
      this.applyWaveDistortion(intensity);
    }

    if (this.sanity < 20) {
      this.applyChromaticAberration(intensity);
      this.applyVignette(intensity);
    }

    if (this.sanity < 0) {
      this.applyDesaturation(intensity);
      this.applyNoise(intensity);
    }
  }

  applyWaveDistortion(intensity) {
    const canvas = this.distortionCanvas.children[0];
    const ctx = this.ctx;
    const w = canvas.width;
    const h = canvas.height;

    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = `rgba(100, 0, 0, ${intensity * 0.1})`;

    for (let x = 0; x < w; x += 20) {
      const y = Math.sin(x * 0.01 + this.distortionTime * 2) * 20 * intensity;
      ctx.fillRect(x, h / 2 + y, 20, 10);
    }
  }

  applyChromaticAberration(intensity) {
    const canvas = this.distortionCanvas.children[0];
    const ctx = this.ctx;
    const w = canvas.width;
    const h = canvas.height;

    const aberration = intensity * 10;

    ctx.fillStyle = `rgba(255, 0, 0, ${intensity * 0.05})`;
    ctx.fillRect(w - aberration, 0, aberration, h);

    ctx.fillStyle = `rgba(0, 0, 255, ${intensity * 0.05})`;
    ctx.fillRect(0, 0, aberration, h);
  }

  applyVignette(intensity) {
    const canvas = this.distortionCanvas.children[0];
    const ctx = this.ctx;
    const w = canvas.width;
    const h = canvas.height;

    const gradient = ctx.createRadialGradient(w / 2, h / 2, w * 0.3, w / 2, h / 2, Math.max(w, h));
    gradient.addColorStop(0, 'rgba(0, 0, 0, 0)');
    gradient.addColorStop(1, `rgba(0, 0, 0, ${intensity * 0.4})`);

    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, w, h);
  }

  applyDesaturation(intensity) {
    const canvas = this.distortionCanvas.children[0];
    const ctx = this.ctx;
    const w = canvas.width;
    const h = canvas.height;

    const fade = intensity * 0.3;
    ctx.fillStyle = `rgba(100, 100, 100, ${fade})`;
    ctx.fillRect(0, 0, w, h);
  }

  applyNoise(intensity) {
    const ctx = this.ctx;
    const w = this._noiseCanvas.width * this._noiseScale;
    const h = this._noiseCanvas.height * this._noiseScale;
    const data = this._noiseData;

    for (let i = 0; i < data.length; i += 4) {
      const bufIdx = i / 4;
      const noise = this._noiseBuffer[bufIdx % this._noiseBuffer.length] * 255 * intensity * 0.5;
      data[i] = noise;
      data[i + 1] = noise * 0.5;
      data[i + 2] = noise;
    }

    this._noiseCtx.putImageData(this._noiseImageData, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(this._noiseCanvas, 0, 0, w, h);
  }
}
