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
  }

  updateSanity(value) {
    this.sanity = Math.max(-100, Math.min(100, value));
  }

  update(dt) {
    this.distortionTime += dt;
    this.applyDistortions();
  }

  applyDistortions() {
    const intensity = 1 - (this.sanity / 100);
    
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
    gradient.addColorStop(0, `rgba(0, 0, 0, 0)`);
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
    const canvas = this.distortionCanvas.children[0];
    const ctx = this.ctx;
    const w = canvas.width;
    const h = canvas.height;
    const pixels = ctx.getImageData(0, 0, w, h);
    const data = pixels.data;

    for (let i = 0; i < data.length; i += 4) {
      const noise = Math.random() * 255 * intensity * 0.5;
      data[i] += noise;
      data[i + 1] += noise * 0.5;
      data[i + 2] += noise;
    }

    ctx.putImageData(pixels, 0, 0);
  }

  // Breathing walls effect (applied to room materials)
  createBreathingMaterial(baseMaterial) {
    return new THREE.ShaderMaterial({
      uniforms: {
        uTime: { value: 0 },
        uBreathIntensity: { value: 0 },
        map: { value: baseMaterial.map }
      },
      vertexShader: `
        uniform float uTime;
        uniform float uBreathIntensity;
        
        void main() {
          vec3 newPos = position;
          float noise = sin(position.x * 2.0 + uTime) * 
                        cos(position.y * 2.0 + uTime * 0.7) * 0.05;
          newPos += normal * noise * uBreathIntensity;
          
          gl_Position = projectionMatrix * modelViewMatrix * vec4(newPos, 1.0);
        }
      `,
      fragmentShader: `
        void main() {
          gl_FragColor = vec4(0.8, 0.8, 0.8, 1.0);
        }
      `
    });
  }

  // Void proximity effect
  createVoidProximityEffect(proximity) {
    // 0 = far, 1 = very close
    const vignette = document.getElementById('void-vignette');
    if (proximity > 0.5) {
      vignette.classList.remove('hidden');
      vignette.style.opacity = (proximity - 0.5) * 2;
    } else {
      vignette.classList.add('hidden');
    }
  }
}

// Infinite corridor illusion helper
export class InfiniteCorridorRender {
  constructor(scene) {
    this.scene = scene;
    this.cubeCamera = new THREE.CubeCamera(0.1, 1000, 512);
    scene.add(this.cubeCamera);
  }

  updateCorridorView(corridor, position) {
    this.cubeCamera.position.copy(position);
    this.cubeCamera.update(this.scene.renderer, this.scene.scene);
  }
}

// Time lag portal effect
export class TimeLagEffect {
  constructor() {
    this.frameBuffer = [];
    this.bufferSize = 60; // 1 second at 60 FPS
  }

  recordFrame(cameraPos, cameraRot) {
    this.frameBuffer.push({
      pos: cameraPos.clone(),
      rot: cameraRot.clone()
    });

    if (this.frameBuffer.length > this.bufferSize) {
      this.frameBuffer.shift();
    }
  }

  getDelayedFrame() {
    if (this.frameBuffer.length > 0) {
      return this.frameBuffer[0];
    }
    return null;
  }
}