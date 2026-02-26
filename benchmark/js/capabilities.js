/* =====================================================
   CAPABILITIES — Browser API feature detection matrix
   ===================================================== */

const Capabilities = (() => {

  const TESTS = [
    {
      name: 'WebGPU',
      icon: 'fa-bolt',
      test: () => !!navigator.gpu,
      detail: () => navigator.gpu ? 'GPU compute & rendering API' : 'Not supported in this browser'
    },
    {
      name: 'WebGL 2.0',
      icon: 'fa-cube',
      test: () => {
        try {
          const c = document.createElement('canvas');
          return !!c.getContext('webgl2');
        } catch { return false; }
      },
      detail: () => {
        try {
          const c = document.createElement('canvas');
          const gl = c.getContext('webgl2');
          if (!gl) return 'Not available';
          const dbg = gl.getExtension('WEBGL_debug_renderer_info');
          return dbg ? gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL) : 'Available';
        } catch { return 'Error'; }
      }
    },
    {
      name: 'WebAssembly',
      icon: 'fa-microchip',
      test: () => typeof WebAssembly !== 'undefined',
      detail: () => typeof WebAssembly !== 'undefined' ? 'Streaming compilation supported' : 'Not available'
    },
    {
      name: 'WASM SIMD',
      icon: 'fa-layer-group',
      test: () => {
        try {
          return WebAssembly.validate(new Uint8Array([0,97,115,109,1,0,0,0,1,5,1,96,0,1,123,3,2,1,0,10,10,1,8,0,65,0,253,15,253,98,11]));
        } catch { return false; }
      },
      detail: () => 'SIMD 128-bit vector instructions'
    },
    {
      name: 'Web Workers',
      icon: 'fa-users-gear',
      test: () => typeof Worker !== 'undefined',
      detail: () => `${navigator.hardwareConcurrency || '?'} logical cores available`
    },
    {
      name: 'SharedArrayBuffer',
      icon: 'fa-share-nodes',
      test: () => typeof SharedArrayBuffer !== 'undefined',
      detail: () => typeof SharedArrayBuffer !== 'undefined' ? 'Cross-origin isolated' : 'Requires COOP/COEP headers'
    },
    {
      name: 'OffscreenCanvas',
      icon: 'fa-image',
      test: () => typeof OffscreenCanvas !== 'undefined',
      detail: () => 'Canvas rendering in workers'
    },
    {
      name: 'WebCodecs',
      icon: 'fa-film',
      test: () => typeof VideoDecoder !== 'undefined',
      detail: () => 'Low-level video/audio codec access'
    },
    {
      name: 'WebTransport',
      icon: 'fa-tower-broadcast',
      test: () => typeof WebTransport !== 'undefined',
      detail: () => 'QUIC-based bidirectional transport'
    },
    {
      name: 'WebRTC',
      icon: 'fa-phone',
      test: () => typeof RTCPeerConnection !== 'undefined',
      detail: () => 'Real-time peer-to-peer communication'
    },
    {
      name: 'IndexedDB',
      icon: 'fa-database',
      test: () => typeof indexedDB !== 'undefined',
      detail: () => 'Client-side structured storage'
    },
    {
      name: 'Cache API',
      icon: 'fa-box-archive',
      test: () => typeof caches !== 'undefined',
      detail: () => 'Service Worker cache storage'
    },
    {
      name: 'WebXR',
      icon: 'fa-vr-cardboard',
      test: () => typeof navigator.xr !== 'undefined',
      detail: () => 'AR / VR immersive experiences'
    },
    {
      name: 'Clipboard API',
      icon: 'fa-clipboard',
      test: () => !!(navigator.clipboard && navigator.clipboard.writeText),
      detail: () => 'Async clipboard read/write'
    },
    {
      name: 'Gamepad API',
      icon: 'fa-gamepad',
      test: () => typeof navigator.getGamepads === 'function',
      detail: () => 'Game controller input'
    },
    {
      name: 'Performance Observer',
      icon: 'fa-gauge-high',
      test: () => typeof PerformanceObserver !== 'undefined',
      detail: () => 'Performance timing metrics'
    }
  ];

  function render() {
    const container = document.getElementById('capabilities-content');
    if (!container) return;

    let supportedCount = 0;
    let rows = '';

    for (const t of TESTS) {
      const supported = t.test();
      if (supported) supportedCount++;
      const detail = t.detail();

      rows += `
        <tr>
          <td>
            <i class="fa-solid ${t.icon}" style="color:var(--accent);margin-right:8px;width:16px;text-align:center"></i>
            ${t.name}
          </td>
          <td>
            <span class="${supported ? 'cap-yes' : 'cap-no'}">
              <i class="fa-solid ${supported ? 'fa-circle-check' : 'fa-circle-xmark'}"></i>
              ${supported ? 'Supported' : 'Not Available'}
            </span>
          </td>
          <td class="cap-detail">${detail}</td>
        </tr>
      `;
    }

    container.innerHTML = `
      <div class="cap-summary">
        <span class="cap-summary-score">${supportedCount} / ${TESTS.length}</span>
        <span class="cap-summary-label">APIs Supported</span>
      </div>
      <table class="cap-table">
        <thead>
          <tr>
            <th>API / Feature</th>
            <th>Status</th>
            <th>Details</th>
          </tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    `;
  }

  return { render };
})();

if (typeof module !== 'undefined' && module.exports) module.exports = Capabilities;
