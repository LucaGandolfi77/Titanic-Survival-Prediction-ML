/* =====================================================
   SYSTEM INFO — Detect hardware/software via browser APIs
   ===================================================== */

const SystemInfo = (() => {

  async function detect() {
    const info = {
      cpu: detectCPU(),
      gpu: await detectGPU(),
      screen: detectScreen(),
      network: detectNetwork(),
      features: detectFeatures(),
      refreshRate: null
    };
    info.refreshRate = await measureRefreshRate();
    return info;
  }

  function detectCPU() {
    const ua = navigator.userAgent || '';
    const platform = navigator.platform || '';
    let arch = 'unknown';
    if (/x86_64|x86-64|Win64|x64|amd64|AMD64/i.test(ua) || /x86_64|x64/.test(platform)) arch = 'x86_64';
    else if (/aarch64|arm64|ARM64/i.test(ua) || /aarch64|arm/i.test(platform)) arch = 'ARM64';
    else if (/x86|i[3-6]86/i.test(ua)) arch = 'x86';

    let os = 'Unknown';
    let osVersion = '';
    if (navigator.userAgentData && navigator.userAgentData.platform) {
      os = navigator.userAgentData.platform;
    } else {
      if (/Windows NT (\d+\.\d+)/i.test(ua)) { os = 'Windows'; const v = ua.match(/Windows NT (\d+\.\d+)/i)[1]; const map = {'10.0':'10/11','6.3':'8.1','6.2':'8','6.1':'7'}; osVersion = map[v] || v; }
      else if (/Mac OS X ([\d_]+)/i.test(ua)) { os = 'macOS'; osVersion = ua.match(/Mac OS X ([\d_]+)/i)[1].replace(/_/g, '.'); }
      else if (/CrOS/i.test(ua)) os = 'ChromeOS';
      else if (/Android ([\d.]+)/i.test(ua)) { os = 'Android'; osVersion = ua.match(/Android ([\d.]+)/i)[1]; }
      else if (/iPhone|iPad|iPod/i.test(ua)) { os = 'iOS'; const m = ua.match(/OS ([\d_]+)/i); if (m) osVersion = m[1].replace(/_/g, '.'); }
      else if (/Linux/i.test(ua)) os = 'Linux';
    }

    let browser = 'Unknown', browserVersion = '', engine = 'Unknown';
    if (/Edg\/([\d.]+)/i.test(ua)) { browser = 'Edge'; browserVersion = ua.match(/Edg\/([\d.]+)/i)[1]; engine = 'Blink'; }
    else if (/OPR\/([\d.]+)/i.test(ua)) { browser = 'Opera'; browserVersion = ua.match(/OPR\/([\d.]+)/i)[1]; engine = 'Blink'; }
    else if (/Chrome\/([\d.]+)/i.test(ua) && !/Edg/i.test(ua)) { browser = 'Chrome'; browserVersion = ua.match(/Chrome\/([\d.]+)/i)[1]; engine = 'Blink'; }
    else if (/Firefox\/([\d.]+)/i.test(ua)) { browser = 'Firefox'; browserVersion = ua.match(/Firefox\/([\d.]+)/i)[1]; engine = 'Gecko'; }
    else if (/Safari\/([\d.]+)/i.test(ua) && !/Chrome/i.test(ua)) { browser = 'Safari'; browserVersion = ua.match(/Version\/([\d.]+)/i)?.[1] || ''; engine = 'WebKit'; }

    return {
      cores: navigator.hardwareConcurrency || 'N/A',
      architecture: arch,
      platform: platform || 'N/A',
      os: os + (osVersion ? ' ' + osVersion : ''),
      browser: browser + (browserVersion ? ' ' + browserVersion : ''),
      engine,
      deviceMemory: navigator.deviceMemory ? navigator.deviceMemory + ' GB' : 'Unavailable',
      touch: navigator.maxTouchPoints > 0
    };
  }

  async function detectGPU() {
    const gpu = {
      available: false,
      vendor: 'N/A',
      device: 'N/A',
      architecture: 'N/A',
      backend: 'N/A',
      maxBufferSize: 'N/A',
      maxTextureSize: 'N/A',
      maxWorkgroupSize: 'N/A',
      maxInvocations: 'N/A',
      timestampQuery: false,
      shaderF16: false
    };

    if (!navigator.gpu) return gpu;

    try {
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) return gpu;
      gpu.available = true;

      let info = null;
      try { info = await adapter.requestAdapterInfo(); } catch (e) { /* some browsers don't support it */ }

      if (info) {
        gpu.vendor = info.vendor || 'Unknown';
        gpu.device = info.device || info.description || 'Unknown';
        gpu.architecture = info.architecture || 'Unknown';
        gpu.backend = info.description || info.backend || 'Unknown';
      }

      gpu.maxBufferSize = adapter.limits.maxBufferSize;
      gpu.maxTextureSize = adapter.limits.maxTextureDimension2D;
      gpu.maxWorkgroupSize = adapter.limits.maxComputeWorkgroupSizeX;
      gpu.maxInvocations = adapter.limits.maxComputeInvocationsPerWorkgroup;
      gpu.timestampQuery = adapter.features.has('timestamp-query');
      gpu.shaderF16 = adapter.features.has('shader-f16');
    } catch (e) {
      console.warn('WebGPU detection error:', e);
    }
    return gpu;
  }

  function detectScreen() {
    return {
      resolution: screen.width + ' × ' + screen.height,
      viewport: window.innerWidth + ' × ' + window.innerHeight,
      pixelRatio: window.devicePixelRatio || 1,
      colorDepth: screen.colorDepth
    };
  }

  function detectNetwork() {
    const conn = navigator.connection || navigator.mozConnection || navigator.webkitConnection;
    return {
      type: conn?.effectiveType || 'Unknown',
      downlink: conn?.downlink ? conn.downlink + ' Mbps' : 'Unknown'
    };
  }

  function detectFeatures() {
    const features = [
      { name: 'WebGPU', available: typeof navigator.gpu !== 'undefined' },
      { name: 'WebGL2', available: !!document.createElement('canvas').getContext('webgl2') },
      { name: 'WebAssembly', available: typeof WebAssembly !== 'undefined' },
      { name: 'SharedArrayBuffer', available: typeof SharedArrayBuffer !== 'undefined' },
      { name: 'Web Workers', available: typeof Worker !== 'undefined' },
      { name: 'OffscreenCanvas', available: typeof OffscreenCanvas !== 'undefined' },
      { name: 'WebCodecs', available: typeof VideoEncoder !== 'undefined' },
      { name: 'WASM SIMD', available: detectWasmSIMD() }
    ];
    return features;
  }

  function detectWasmSIMD() {
    try {
      return WebAssembly.validate(new Uint8Array([
        0,97,115,109,1,0,0,0,1,5,1,96,0,1,123,3,2,1,0,10,10,1,8,0,65,0,253,15,253,98,11
      ]));
    } catch (e) {
      return false;
    }
  }

  function measureRefreshRate() {
    return new Promise(resolve => {
      let frames = 0;
      let last = performance.now();
      const deltas = [];
      function step(now) {
        frames++;
        if (frames > 1) deltas.push(now - last);
        last = now;
        if (frames < 30) {
          requestAnimationFrame(step);
        } else {
          const avg = deltas.reduce((a, b) => a + b, 0) / deltas.length;
          const hz = Math.round(1000 / avg);
          const rounded = [60, 72, 75, 90, 120, 144, 165, 240].reduce((prev, curr) =>
            Math.abs(curr - hz) < Math.abs(prev - hz) ? curr : prev
          );
          resolve('~' + rounded + 'Hz');
        }
      }
      requestAnimationFrame(step);
    });
  }

  function formatBytes(bytes) {
    if (typeof bytes !== 'number') return 'N/A';
    if (bytes >= 1e9) return (bytes / 1e9).toFixed(2) + ' GB';
    if (bytes >= 1e6) return (bytes / 1e6).toFixed(2) + ' MB';
    return bytes + ' bytes';
  }

  function render(info) {
    const grid = document.getElementById('sysinfo-grid');

    // CPU Card
    const cpuHtml = `
      <div class="sysinfo-card">
        <div class="sysinfo-card-header">
          <i class="fa-solid fa-microchip icon-cpu"></i>
          <h3>CPU & System</h3>
        </div>
        <div class="sysinfo-row"><span class="sysinfo-label">Logical Cores</span><span class="sysinfo-value">${info.cpu.cores}</span></div>
        <div class="sysinfo-row"><span class="sysinfo-label">Architecture</span><span class="sysinfo-value">${info.cpu.architecture}</span></div>
        <div class="sysinfo-row"><span class="sysinfo-label">Platform</span><span class="sysinfo-value">${info.cpu.platform}</span></div>
        <div class="sysinfo-row"><span class="sysinfo-label">OS</span><span class="sysinfo-value">${info.cpu.os}</span></div>
        <div class="sysinfo-row"><span class="sysinfo-label">Browser</span><span class="sysinfo-value">${info.cpu.browser}</span></div>
        <div class="sysinfo-row"><span class="sysinfo-label">Engine</span><span class="sysinfo-value">${info.cpu.engine}</span></div>
        <div class="sysinfo-row"><span class="sysinfo-label">Device Memory</span><span class="sysinfo-value">${info.cpu.deviceMemory}</span></div>
        <div class="sysinfo-row"><span class="sysinfo-label">Touch Support</span><span class="sysinfo-value">${info.cpu.touch ? 'Yes' : 'No'}</span></div>
      </div>
    `;

    // GPU Card
    const gpuClass = info.gpu.available ? 'sysinfo-card gpu-glow' : 'sysinfo-card';
    const gpuHtml = `
      <div class="${gpuClass}">
        <div class="sysinfo-card-header">
          <i class="fa-solid fa-display icon-gpu"></i>
          <h3>GPU (WebGPU)</h3>
        </div>
        ${info.gpu.available ? `
          <div class="sysinfo-row"><span class="sysinfo-label">Vendor</span><span class="sysinfo-value">${info.gpu.vendor}</span></div>
          <div class="sysinfo-row"><span class="sysinfo-label">Device</span><span class="sysinfo-value">${info.gpu.device}</span></div>
          <div class="sysinfo-row"><span class="sysinfo-label">Architecture</span><span class="sysinfo-value">${info.gpu.architecture}</span></div>
          <div class="sysinfo-row"><span class="sysinfo-label">Backend</span><span class="sysinfo-value">${info.gpu.backend}</span></div>
          <div class="sysinfo-row"><span class="sysinfo-label">Max Buffer</span><span class="sysinfo-value">${formatBytes(info.gpu.maxBufferSize)}</span></div>
          <div class="sysinfo-row"><span class="sysinfo-label">Max Texture</span><span class="sysinfo-value">${info.gpu.maxTextureSize}px</span></div>
          <div class="sysinfo-row"><span class="sysinfo-label">Max Workgroup</span><span class="sysinfo-value">${info.gpu.maxWorkgroupSize}</span></div>
          <div class="sysinfo-row"><span class="sysinfo-label">Compute Invocations</span><span class="sysinfo-value">${info.gpu.maxInvocations}</span></div>
          <div class="sysinfo-row"><span class="sysinfo-label">Timestamp Query</span><span class="sysinfo-value">${info.gpu.timestampQuery ? '✓ Yes' : '✗ No'}</span></div>
          <div class="sysinfo-row"><span class="sysinfo-label">Shader F16</span><span class="sysinfo-value">${info.gpu.shaderF16 ? '✓ Yes' : '✗ No'}</span></div>
        ` : '<div style="padding:1rem 0;color:var(--warning);font-size:0.85rem"><i class="fa-solid fa-triangle-exclamation"></i> WebGPU not available</div>'}
      </div>
    `;

    // Screen & Network Card
    const screenHtml = `
      <div class="sysinfo-card">
        <div class="sysinfo-card-header">
          <i class="fa-solid fa-desktop icon-net"></i>
          <h3>Screen & Network</h3>
        </div>
        <div class="sysinfo-row"><span class="sysinfo-label">Resolution</span><span class="sysinfo-value">${info.screen.resolution}</span></div>
        <div class="sysinfo-row"><span class="sysinfo-label">Viewport</span><span class="sysinfo-value">${info.screen.viewport}</span></div>
        <div class="sysinfo-row"><span class="sysinfo-label">Pixel Ratio</span><span class="sysinfo-value">${info.screen.pixelRatio}x</span></div>
        <div class="sysinfo-row"><span class="sysinfo-label">Color Depth</span><span class="sysinfo-value">${info.screen.colorDepth} bit</span></div>
        <div class="sysinfo-row"><span class="sysinfo-label">Refresh Rate</span><span class="sysinfo-value">${info.refreshRate || 'Measuring...'}</span></div>
        <div class="sysinfo-row"><span class="sysinfo-label">Connection</span><span class="sysinfo-value">${info.network.type}</span></div>
        <div class="sysinfo-row"><span class="sysinfo-label">Downlink</span><span class="sysinfo-value">${info.network.downlink}</span></div>
      </div>
    `;

    // Features Card
    const featHtml = `
      <div class="sysinfo-card" style="grid-column: 1 / -1;">
        <div class="sysinfo-card-header">
          <i class="fa-solid fa-puzzle-piece icon-feat"></i>
          <h3>Runtime Features</h3>
        </div>
        <div class="feature-grid">
          ${info.features.map(f => `
            <span class="feature-badge ${f.available ? 'available' : 'unavailable'}">
              <i class="fa-solid ${f.available ? 'fa-check' : 'fa-xmark'}"></i> ${f.name}
            </span>
          `).join('')}
        </div>
      </div>
    `;

    grid.innerHTML = cpuHtml + gpuHtml + screenHtml + featHtml;
  }

  return { detect, render };
})();

if (typeof module !== 'undefined' && module.exports) module.exports = SystemInfo;
