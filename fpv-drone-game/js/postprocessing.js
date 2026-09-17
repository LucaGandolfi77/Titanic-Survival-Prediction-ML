/* ── js/postprocessing.js ── Bloom, Vignette, Tone Mapping ── */

import * as THREE from 'three';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { ShaderPass } from 'three/addons/postprocessing/ShaderPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';

/* ── Vignette Shader ── */
const VignetteShader = {
  uniforms: {
    tDiffuse: { value: null },
    intensity: { value: 0.35 },
  },
  vertexShader: `
    varying vec2 vUv;
    void main() {
      vUv = uv;
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }
  `,
  fragmentShader: `
    uniform sampler2D tDiffuse;
    uniform float intensity;
    varying vec2 vUv;
    void main() {
      vec4 texel = texture2D(tDiffuse, vUv);
      float dist = distance(vUv, vec2(0.5));
      float vignette = smoothstep(0.85, 0.15, dist);
      vec3 darkened = texel.rgb * mix(1.0, vignette, intensity);
      gl_FragColor = vec4(darkened, texel.a);
    }
  `,
};

/* ══════════════════════════════════════════════════════════
   PostProcessingManager — Bloom + Vignette + Output
   ══════════════════════════════════════════════════════════ */
export class PostProcessingManager {
  constructor(renderer, scene, camera) {
    this.renderer = renderer;
    this.scene = scene;
    this.camera = camera;

    const size = renderer.getSize(new THREE.Vector2());

    this.composer = new EffectComposer(renderer);
    this.renderPass = new RenderPass(scene, camera);
    this.composer.addPass(this.renderPass);

    this.bloomPass = new UnrealBloomPass(
      new THREE.Vector2(size.width, size.height),
      0.8,
      0.4,
      0.85
    );
    this.composer.addPass(this.bloomPass);

    this.vignettePass = new ShaderPass(VignetteShader);
    this.composer.addPass(this.vignettePass);

    this.outputPass = new OutputPass();
    this.composer.addPass(this.outputPass);

    this._baseBloom = 0.8;
  }

  /* ── Render frame ── */
  render(delta) {
    if (delta) {
      this.composer.render(delta);
    } else {
      this.composer.render();
    }
  }

  /* ── Set bloom intensity ── */
  setBloom(value) {
    this.bloomPass.strength = value;
  }

  /* ── Boost bloom (for combat, explosions) ── */
  boostBloom(amount) {
    this.bloomPass.strength = this._baseBloom + amount;
  }

  /* ── Reset bloom to base ── */
  resetBloom() {
    this.bloomPass.strength = this._baseBloom;
  }

  /* ── Set vignette intensity ── */
  setVignette(value) {
    this.vignettePass.uniforms.intensity.value = value;
  }

  /* ── Resize handling ── */
  resize(width, height) {
    this.composer.setSize(width, height);
    this.bloomPass.setSize(width, height);
  }
}
