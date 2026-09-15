import * as THREE from 'three';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { BokehPass } from 'three/addons/postprocessing/BokehPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import { Pass } from 'three/addons/postprocessing/Pass.js';

export class PortalRenderPass extends Pass {
  constructor(renderPortalsFn, scene, camera) {
    super();
    this._renderPortals = renderPortalsFn;
    this.scene = scene;
    this.camera = camera;
  }

  render(renderer, writeBuffer, readBuffer, deltaTime, maskActive) {
    renderer.setRenderTarget(this.renderToScreen ? null : writeBuffer);
    renderer.clear();
    this._renderPortals();
    renderer.render(this.scene, this.camera);
  }
}

export function createPostProcessing(renderer, scene, camera, renderPortalsFn) {
  const composer = new EffectComposer(renderer);

  const portalPass = new PortalRenderPass(renderPortalsFn, scene, camera);

  const bloomPass = new UnrealBloomPass(
    new THREE.Vector2(window.innerWidth, window.innerHeight),
    0.3,
    0.4,
    0.85
  );

  const bokehPass = new BokehPass(scene, camera, {
    focus: 15.0,
    aperture: 0.025,
    maxblur: 0.01
  });

  const outputPass = new OutputPass();

  composer.addPass(portalPass);
  composer.addPass(bloomPass);
  composer.addPass(bokehPass);
  composer.addPass(outputPass);

  return { composer, bloomPass, bokehPass, portalPass };
}
