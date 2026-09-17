import * as THREE from 'three';

export function createRenderPipeline(appState, els, lifecycle) {
  let resizeTimer = null;

  function resize() {
    const w = window.innerWidth;
    const h = window.innerHeight;
    els.overlayCanvas.width = w;
    els.overlayCanvas.height = h;
    if (appState.renderer) {
      appState.renderer.setSize(w, h, false);
      if (appState.material?.uniforms?.uResolution) {
        appState.material.uniforms.uResolution.value.set(w, h);
      }
    }
  }

  function debouncedResize() {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(resize, 100);
  }

  function init(canvasEls) {
    appState.renderer = new THREE.WebGLRenderer({
      canvas: canvasEls.glCanvas,
      alpha: false,
      antialias: true,
      preserveDrawingBuffer: true,
    });
    appState.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    appState.scene = new THREE.Scene();
    appState.camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);

    if (!appState.videoTexture) {
      appState.videoTexture = new THREE.VideoTexture(els.camera);
      appState.videoTexture.colorSpace = THREE.SRGBColorSpace;
    }

    appState.mesh = new THREE.Mesh(
      new THREE.PlaneGeometry(2, 2),
      { uniforms: { uTime: { value: 0 }, uMirror: { value: 1 } } }
    );
    appState.material = appState.mesh.material;
    appState.scene.add(appState.mesh);
  }

  function render(time) {
    if (appState.previewing) return;
    if (!els.camera.videoWidth) return;

    if (appState.material?.uniforms?.uTime) {
      appState.material.uniforms.uTime.value = time / 1000;
    }
    if (appState.webglSupported && appState.renderer) {
      appState.renderer.render(appState.scene, appState.camera);
    }
  }

  lifecycle.on('shutdown', () => {
    if (appState.renderer) appState.renderer.dispose();
    if (appState.material) appState.material.dispose();
    if (appState.videoTexture) appState.videoTexture.dispose();
  });

  return { init, render, resize, debouncedResize };
}
