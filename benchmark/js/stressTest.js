/* =====================================================
   STRESS TEST — Continuous CPU/GPU stress with live charts
   ===================================================== */

const StressTest = (() => {

  let running = false;
  let cpuWorkers = [];
  let gpuRunning = false;
  let cpuChart = null;
  let gpuChart = null;
  let duration = 60; // seconds
  let startTime = 0;
  let cpuHistory = [];
  let gpuHistory = [];
  let peakCpuOps = 0;
  let peakGpuGflops = 0;
  let _throttleWarned = false;

  function render() {
    const el = document.getElementById('stress-content');
    el.innerHTML = `
      <div class="stress-controls">
        <span style="font-size:0.8rem;color:var(--text-muted);font-weight:600">Duration:</span>
        <div class="stress-duration-btns">
          <button class="btn btn-outline btn-sm stress-dur-btn ${duration === 60 ? 'active' : ''}" data-dur="60">1 min</button>
          <button class="btn btn-outline btn-sm stress-dur-btn ${duration === 300 ? 'active' : ''}" data-dur="300">5 min</button>
          <button class="btn btn-outline btn-sm stress-dur-btn ${duration === 600 ? 'active' : ''}" data-dur="600">10 min</button>
          <button class="btn btn-outline btn-sm stress-dur-btn ${duration === 0 ? 'active' : ''}" data-dur="0">∞</button>
        </div>
        <button id="btn-start-stress" class="btn ${running ? 'btn-danger' : 'btn-accent'} btn-sm">
          <i class="fa-solid ${running ? 'fa-stop' : 'fa-play'}"></i>
          ${running ? 'Stop Stress Test' : 'Start Stress Test'}
        </button>
      </div>

      <div class="stress-grid">
        <!-- CPU Stress -->
        <div class="stress-card">
          <h3><i class="fa-solid fa-microchip" style="color:var(--accent3)"></i> CPU Stress</h3>
          <div class="stress-live-value" id="stress-cpu-value">—</div>
          <div class="stress-live-label">Giga-ops/sec</div>
          <div class="stress-chart-wrap"><canvas id="stress-cpu-chart"></canvas></div>
          <div id="stress-cpu-warning" class="stress-warning" style="display:none">
            <i class="fa-solid fa-triangle-exclamation"></i>
            <span>Possible thermal throttling detected — performance dropped &gt;15% from peak</span>
          </div>
          <div class="stress-stats">
            <div class="stress-stat"><div class="stress-stat-value" id="stress-cpu-peak">—</div><div class="stress-stat-label">Peak</div></div>
            <div class="stress-stat"><div class="stress-stat-value" id="stress-cpu-avg">—</div><div class="stress-stat-label">Average</div></div>
            <div class="stress-stat"><div class="stress-stat-value" id="stress-cpu-min">—</div><div class="stress-stat-label">Min</div></div>
          </div>
        </div>

        <!-- GPU Stress -->
        <div class="stress-card">
          <h3><i class="fa-solid fa-display" style="color:var(--accent)"></i> GPU Stress</h3>
          <div class="stress-live-value" id="stress-gpu-value">—</div>
          <div class="stress-live-label">GFLOPS</div>
          <div class="stress-chart-wrap"><canvas id="stress-gpu-chart"></canvas></div>
          <div id="stress-gpu-warning" class="stress-warning" style="display:none">
            <i class="fa-solid fa-triangle-exclamation"></i>
            <span>Possible thermal throttling detected — performance dropped &gt;15% from peak</span>
          </div>
          <div class="stress-stats">
            <div class="stress-stat"><div class="stress-stat-value" id="stress-gpu-peak">—</div><div class="stress-stat-label">Peak</div></div>
            <div class="stress-stat"><div class="stress-stat-value" id="stress-gpu-avg">—</div><div class="stress-stat-label">Average</div></div>
            <div class="stress-stat"><div class="stress-stat-value" id="stress-gpu-min">—</div><div class="stress-stat-label">Min</div></div>
          </div>
        </div>
      </div>
    `;

    _bindControls();
  }

  function _bindControls() {
    document.querySelectorAll('.stress-dur-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        if (running) return;
        duration = parseInt(btn.dataset.dur, 10);
        document.querySelectorAll('.stress-dur-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
      });
    });

    document.getElementById('btn-start-stress')?.addEventListener('click', () => {
      if (running) stop();
      else start();
    });
  }

  function start() {
    running = true;
    cpuHistory = [];
    gpuHistory = [];
    peakCpuOps = 0;
    peakGpuGflops = 0;
    _throttleWarned = false;
    startTime = performance.now();

    _initCharts();
    _startCpuStress();
    _startGpuStress();
  }

  function stop() {
    running = false;
    cpuWorkers.forEach(w => w.terminate());
    cpuWorkers = [];
    gpuRunning = false;
    render();
  }

  function _initCharts() {
    const cpuCtx = document.getElementById('stress-cpu-chart');
    const gpuCtx = document.getElementById('stress-gpu-chart');
    if (!cpuCtx || !gpuCtx) return;

    const chartOpts = {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      scales: {
        x: { display: false },
        y: {
          grid: { color: 'rgba(42,42,61,0.4)' },
          ticks: { color: '#64748b', font: { size: 10, family: 'JetBrains Mono' } }
        }
      },
      plugins: { legend: { display: false } },
      elements: { point: { radius: 0 }, line: { borderWidth: 2 } }
    };

    if (cpuChart) cpuChart.destroy();
    cpuChart = new Chart(cpuCtx, {
      type: 'line',
      data: {
        labels: [],
        datasets: [{
          data: [],
          borderColor: '#f59e0b',
          backgroundColor: 'rgba(245,158,11,0.1)',
          fill: true,
          tension: 0.3
        }]
      },
      options: chartOpts
    });

    if (gpuChart) gpuChart.destroy();
    gpuChart = new Chart(gpuCtx, {
      type: 'line',
      data: {
        labels: [],
        datasets: [{
          data: [],
          borderColor: '#00d4ff',
          backgroundColor: 'rgba(0,212,255,0.1)',
          fill: true,
          tension: 0.3
        }]
      },
      options: chartOpts
    });
  }

  function _startCpuStress() {
    const numWorkers = navigator.hardwareConcurrency || 4;

    const workerCode = `
      self.onmessage = function(e) {
        function run() {
          const end = performance.now() + 1000; // 1 second batches
          let ops = 0, a = 1, b = 2, c = 0;
          while (performance.now() < end) {
            c = a + b; c = c * 3; c = c - 1; c = c ^ 0xFF;
            c = (c << 2) | (c >> 6); c = c & 0xFFFF;
            c = c % 97; c = c + a; c = c * b; c = c - 7;
            ops++;
          }
          self.postMessage({ ops: ops * 10 });
          setTimeout(run, 10);
        }
        run();
      };
    `;

    let blob, url;
    try {
      blob = new Blob([workerCode], { type: 'application/javascript' });
      url = URL.createObjectURL(blob);
    } catch (e) { return; }

    let lastUpdate = performance.now();
    let accumulated = 0;
    let workerReports = 0;

    for (let i = 0; i < numWorkers; i++) {
      const w = new Worker(url);
      w.onmessage = (e) => {
        if (!running) return;
        accumulated += e.data.ops;
        workerReports++;

        if (workerReports >= numWorkers) {
          const now = performance.now();
          const dt = (now - lastUpdate) / 1000;
          const gops = accumulated / dt / 1e9;

          cpuHistory.push(gops);
          if (cpuHistory.length > 60) cpuHistory.shift();
          if (gops > peakCpuOps) peakCpuOps = gops;

          // Thermal throttle check
          if (cpuHistory.length > 5 && gops < peakCpuOps * 0.85) {
            const warn = document.getElementById('stress-cpu-warning');
            if (warn) warn.style.display = 'flex';
          }

          _updateCpuUI(gops);
          accumulated = 0;
          workerReports = 0;
          lastUpdate = now;

          // Duration check
          if (duration > 0 && (now - startTime) / 1000 >= duration) {
            stop();
          }
        }
      };
      w.postMessage({});
      cpuWorkers.push(w);
    }

    URL.revokeObjectURL(url);
  }

  async function _startGpuStress() {
    if (!navigator.gpu) return;
    gpuRunning = true;

    try {
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) return;
      const device = await adapter.requestDevice();

      const N = 512; // Smaller for continuous stress
      const matrixSize = N * N * 4;
      const bufA = device.createBuffer({ size: matrixSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      const bufB = device.createBuffer({ size: matrixSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      const bufC = device.createBuffer({ size: matrixSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
      const bufDims = device.createBuffer({ size: 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

      const dataA = new Float32Array(N * N).map(() => Math.random());
      const dataB = new Float32Array(N * N).map(() => Math.random());
      device.queue.writeBuffer(bufA, 0, dataA);
      device.queue.writeBuffer(bufB, 0, dataB);
      device.queue.writeBuffer(bufDims, 0, new Uint32Array([N]));

      const shaderModule = device.createShaderModule({
        code: `
          @group(0) @binding(0) var<storage, read> A : array<f32>;
          @group(0) @binding(1) var<storage, read> B : array<f32>;
          @group(0) @binding(2) var<storage, read_write> C : array<f32>;
          @group(0) @binding(3) var<uniform> dims : u32;
          const TILE : u32 = 16u;
          var<workgroup> tA : array<array<f32, 16>, 16>;
          var<workgroup> tB : array<array<f32, 16>, 16>;
          @compute @workgroup_size(16, 16)
          fn main(@builtin(global_invocation_id) gid : vec3<u32>, @builtin(local_invocation_id) lid : vec3<u32>) {
            let NN = dims;
            let row = gid.y; let col = gid.x;
            var sum : f32 = 0.0;
            for (var t : u32 = 0u; t < NN / TILE; t = t + 1u) {
              tA[lid.y][lid.x] = A[row * NN + t * TILE + lid.x];
              tB[lid.y][lid.x] = B[(t * TILE + lid.y) * NN + col];
              workgroupBarrier();
              for (var k : u32 = 0u; k < TILE; k = k + 1u) { sum = sum + tA[lid.y][k] * tB[k][lid.x]; }
              workgroupBarrier();
            }
            if (row < NN && col < NN) { C[row * NN + col] = sum; }
          }
        `
      });

      const pipeline = device.createComputePipeline({ layout: 'auto', compute: { module: shaderModule, entryPoint: 'main' } });
      const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: bufA } },
          { binding: 1, resource: { buffer: bufB } },
          { binding: 2, resource: { buffer: bufC } },
          { binding: 3, resource: { buffer: bufDims } }
        ]
      });
      const wg = Math.ceil(N / 16);

      const loop = async () => {
        if (!running || !gpuRunning) {
          device.destroy();
          return;
        }

        const iters = 5;
        const t0 = performance.now();
        for (let i = 0; i < iters; i++) {
          const enc = device.createCommandEncoder();
          const pass = enc.beginComputePass();
          pass.setPipeline(pipeline);
          pass.setBindGroup(0, bindGroup);
          pass.dispatchWorkgroups(wg, wg);
          pass.end();
          device.queue.submit([enc.finish()]);
        }
        await device.queue.onSubmittedWorkDone();
        const elapsed = (performance.now() - t0) / 1000;

        const flops = 2 * N * N * N * iters;
        const gflops = flops / elapsed / 1e9;

        gpuHistory.push(gflops);
        if (gpuHistory.length > 60) gpuHistory.shift();
        if (gflops > peakGpuGflops) peakGpuGflops = gflops;

        if (gpuHistory.length > 5 && gflops < peakGpuGflops * 0.85) {
          const warn = document.getElementById('stress-gpu-warning');
          if (warn) warn.style.display = 'flex';
        }

        _updateGpuUI(gflops);
        setTimeout(loop, 50);
      };

      loop();
    } catch (e) {
      console.warn('GPU stress error:', e);
    }
  }

  function _updateCpuUI(gops) {
    const valEl = document.getElementById('stress-cpu-value');
    const peakEl = document.getElementById('stress-cpu-peak');
    const avgEl = document.getElementById('stress-cpu-avg');
    const minEl = document.getElementById('stress-cpu-min');

    if (valEl) valEl.textContent = gops.toFixed(2);
    if (peakEl) peakEl.textContent = peakCpuOps.toFixed(2);

    if (cpuHistory.length > 0) {
      const avg = cpuHistory.reduce((a, b) => a + b, 0) / cpuHistory.length;
      const min = Math.min(...cpuHistory);
      if (avgEl) avgEl.textContent = avg.toFixed(2);
      if (minEl) minEl.textContent = min.toFixed(2);
    }

    if (cpuChart) {
      cpuChart.data.labels = cpuHistory.map((_, i) => i + 's');
      cpuChart.data.datasets[0].data = [...cpuHistory];
      cpuChart.update('none');
    }
  }

  function _updateGpuUI(gflops) {
    const valEl = document.getElementById('stress-gpu-value');
    const peakEl = document.getElementById('stress-gpu-peak');
    const avgEl = document.getElementById('stress-gpu-avg');
    const minEl = document.getElementById('stress-gpu-min');

    if (valEl) valEl.textContent = gflops.toFixed(1);
    if (peakEl) peakEl.textContent = peakGpuGflops.toFixed(1);

    if (gpuHistory.length > 0) {
      const avg = gpuHistory.reduce((a, b) => a + b, 0) / gpuHistory.length;
      const min = Math.min(...gpuHistory);
      if (avgEl) avgEl.textContent = avg.toFixed(1);
      if (minEl) minEl.textContent = min.toFixed(1);
    }

    if (gpuChart) {
      gpuChart.data.labels = gpuHistory.map((_, i) => i + 's');
      gpuChart.data.datasets[0].data = [...gpuHistory];
      gpuChart.update('none');
    }
  }

  return { render, start, stop, isRunning: () => running };
})();

if (typeof module !== 'undefined' && module.exports) module.exports = StressTest;
