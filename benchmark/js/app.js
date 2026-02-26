/* =====================================================
   APP — Main entry point, benchmark card rendering,
         Run-All orchestration, UI wiring
   ===================================================== */

const App = (() => { // eslint-disable-line no-unused-vars

  let sysInfo = null;
  const benchResults = {};
  let running = false;

  /* ---- Benchmark definitions ---- */
  const BENCHMARKS = [
    {
      id: 'cpuInt', section: 'cpu', label: 'CPU Integer',
      icon: 'fa-calculator', color: 'var(--accent3)',
      run: (p) => CpuBenchmarks.integerBenchmark(5000, p),
      format: r => `${r.gops.toFixed(2)} Gops/s`,
      tier: r => CpuBenchmarks.getTierInt(r.gops),
      key: 'gops', refVal: Results.REF.cpuInt, refLabel: 'Avg Desktop'
    },
    {
      id: 'cpuFloat', section: 'cpu', label: 'CPU Float',
      icon: 'fa-wave-square', color: 'var(--accent3)',
      run: (p) => CpuBenchmarks.floatBenchmark(5000, p),
      format: r => `${r.mflops.toFixed(1)} Mflops/s`,
      tier: r => CpuBenchmarks.getTierFloat(r.mflops),
      key: 'mflops', refVal: Results.REF.cpuFloat, refLabel: 'Avg Desktop'
    },
    {
      id: 'cpuMulti', section: 'cpu', label: 'CPU Multi-Thread',
      icon: 'fa-layer-group', color: 'var(--accent3)',
      run: (p) => CpuBenchmarks.multiThreadBenchmark(p),
      format: r => `${r.gops.toFixed(2)} Gops/s (${r.numWorkers} threads)`,
      tier: r => CpuBenchmarks.getTierMulti(r.gops),
      key: 'gops', refVal: Results.REF.cpuMulti, refLabel: 'Avg Desktop'
    },
    {
      id: 'memBW', section: 'cpu', label: 'Memory Bandwidth',
      icon: 'fa-memory', color: '#a78bfa',
      run: (p) => CpuBenchmarks.memoryBandwidthBenchmark(p),
      format: r => `R: ${(r.readBW / 1000).toFixed(1)} GB/s  W: ${(r.writeBW / 1000).toFixed(1)} GB/s`,
      tier: r => CpuBenchmarks.getTierMemory(r.readBW / 1000),
      key: 'readBW', refVal: Results.REF.memBW * 1000, refLabel: 'Avg Desktop'
    },
    {
      id: 'json', section: 'cpu', label: 'JSON Processing',
      icon: 'fa-code', color: 'var(--accent3)',
      run: (p) => CpuBenchmarks.jsonBenchmark(5000, p),
      format: r => `${r.opsPerSec.toFixed(0)} ops/s`,
      tier: r => CpuBenchmarks.getTierJSON(r.opsPerSec),
      key: 'opsPerSec', refVal: Results.REF.json, refLabel: 'Avg Desktop'
    },
    {
      id: 'gpuCompute', section: 'gpu', label: 'GPU Compute (MatMul)',
      icon: 'fa-bolt', color: 'var(--accent)',
      run: (p) => GpuBenchmarks.matMulBenchmark(p),
      format: r => `${r.gflops.toFixed(1)} GFLOPS (${r.tflops.toFixed(3)} TFLOPS)`,
      tier: r => GpuBenchmarks.getTierMatMul(r.tflops),
      key: 'tflops', refVal: Results.REF.gpuCompute, refLabel: 'Avg Desktop'
    },
    {
      id: 'gpuMem', section: 'gpu', label: 'GPU Memory BW',
      icon: 'fa-hard-drive', color: 'var(--accent)',
      run: (p) => GpuBenchmarks.gpuMemoryBandwidthBenchmark(p),
      format: r => `${r.gbps.toFixed(1)} GB/s`,
      tier: r => GpuBenchmarks.getTierGpuMem(r.gbps),
      key: 'gbps', refVal: Results.REF.gpuMem, refLabel: 'Avg Desktop'
    },
    {
      id: 'render', section: 'gpu', label: 'WebGL Render',
      icon: 'fa-display', color: '#34d399',
      run: (p) => GpuBenchmarks.webglRenderBenchmark(p),
      format: r => `${r.fps.toFixed(1)} FPS  (${r.mTriPerSec.toFixed(2)} M tri/s)`,
      tier: r => GpuBenchmarks.getTierRender(r.fps),
      key: 'fps', refVal: Results.REF.render, refLabel: 'Avg Desktop'
    },
    // --- AI / ML Benchmarks ---
    {
      id: 'mlNNInf', section: 'ml', label: 'NN Inference',
      icon: 'fa-brain', color: '#ec4899',
      run: (p) => MlBenchmarks.nnInferenceBenchmark(5000, p),
      format: r => `${r.infPerSec.toLocaleString()} inf/s  (${r.mflops.toFixed(1)} Mflops)`,
      tier: r => MlBenchmarks.getTierNNInference(r.infPerSec),
      key: 'infPerSec', refVal: Results.REF.mlNNInf, refLabel: 'Avg Desktop'
    },
    {
      id: 'mlConv2D', section: 'ml', label: 'Conv2D (3×3)',
      icon: 'fa-table-cells', color: '#ec4899',
      run: (p) => MlBenchmarks.conv2dBenchmark(5000, p),
      format: r => `${r.convPerSec.toFixed(1)} conv/s  (${r.mpixPerSec.toFixed(1)} Mpix/s)`,
      tier: r => MlBenchmarks.getTierConv2D(r.convPerSec),
      key: 'convPerSec', refVal: Results.REF.mlConv2D, refLabel: 'Avg Desktop'
    },
    {
      id: 'mlTensorOps', section: 'ml', label: 'Tensor Ops',
      icon: 'fa-cubes', color: '#ec4899',
      run: (p) => MlBenchmarks.tensorOpsBenchmark(5000, p),
      format: r => `${r.gflops.toFixed(3)} GFLOPS  (${r.opsPerSec.toFixed(0)} ops/s)`,
      tier: r => MlBenchmarks.getTierTensorOps(r.gflops),
      key: 'gflops', refVal: Results.REF.mlTensorOps, refLabel: 'Avg Desktop'
    },
    {
      id: 'mlKMeans', section: 'ml', label: 'K-Means Clustering',
      icon: 'fa-circle-nodes', color: '#ec4899',
      run: (p) => MlBenchmarks.kMeansBenchmark(5000, p),
      format: r => `${r.itersPerSec.toFixed(1)} iter/s  (${r.gflops.toFixed(3)} GFLOPS)`,
      tier: r => MlBenchmarks.getTierKMeans(r.itersPerSec),
      key: 'itersPerSec', refVal: Results.REF.mlKMeans, refLabel: 'Avg Desktop'
    }
  ];

  /* ---- Toast ---- */
  function toast(msg, type = 'info') {
    const container = document.getElementById('toast-container');
    const div = document.createElement('div');
    div.className = `toast toast-${type} anim-slide-up`;
    const icons = { info: 'fa-circle-info', success: 'fa-circle-check', error: 'fa-circle-xmark', warning: 'fa-triangle-exclamation' };
    div.innerHTML = `<i class="fa-solid ${icons[type] || icons.info}"></i> ${msg}`;
    container.appendChild(div);
    setTimeout(() => { div.classList.add('toast-exit'); setTimeout(() => div.remove(), 400); }, 3500);
  }

  /* ---- Build benchmark cards ---- */
  function _renderBenchCards() {
    const grid = document.getElementById('bench-grid');
    grid.innerHTML = '';

    for (const b of BENCHMARKS) {
      const card = document.createElement('div');
      card.className = 'bench-card anim-fade-in';
      card.id = `bench-${b.id}`;
      card.innerHTML = `
        <div class="bench-card-header">
          <div class="bench-card-icon" style="color:${b.color}">
            <i class="fa-solid ${b.icon}"></i>
          </div>
          <div class="bench-card-title">${b.label}</div>
          <button class="btn btn-outline btn-sm bench-run-btn" data-id="${b.id}">
            <i class="fa-solid fa-play"></i> Run
          </button>
        </div>
        <div class="bench-progress-wrap">
          <div class="bench-progress-bar progress-${b.section}" id="progress-${b.id}" style="width:0%"></div>
        </div>
        <div class="bench-result" id="result-${b.id}">
          <span class="bench-result-placeholder">Not yet run</span>
        </div>
      `;
      grid.appendChild(card);
    }

    // Bind individual run buttons
    grid.querySelectorAll('.bench-run-btn').forEach(btn => {
      btn.addEventListener('click', () => _runSingleBench(btn.dataset.id));
    });
  }

  /* ---- Run single benchmark ---- */
  async function _runSingleBench(id) {
    if (running) return;
    const b = BENCHMARKS.find(x => x.id === id);
    if (!b) return;

    const card = document.getElementById(`bench-${id}`);
    const progBar = document.getElementById(`progress-${id}`);
    const resultEl = document.getElementById(`result-${id}`);
    const runBtn = card.querySelector('.bench-run-btn');

    card.classList.add('running');
    card.classList.remove('completed', 'error');
    runBtn.disabled = true;
    runBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Running';
    progBar.style.width = '0%';
    resultEl.innerHTML = '<span class="bench-result-placeholder"><i class="fa-solid fa-spinner fa-spin"></i> Running...</span>';

    try {
      const result = await b.run(pct => {
        progBar.style.width = `${Math.round(pct * 100)}%`;
      });

      benchResults[id] = result;
      const tier = b.tier(result);
      const compareRatio = Math.min(2, (result[b.key] || 0) / b.refVal);

      card.classList.remove('running');
      card.classList.add('completed');
      progBar.style.width = '100%';

      resultEl.innerHTML = `
        <div class="bench-score-row">
          <span class="score-value">${b.format(result)}</span>
          <span class="tier-badge tier-${tier}">${tier}</span>
        </div>
        <div class="bench-compare">
          <div class="compare-bar-bg">
            <div class="compare-bar-fill" style="width:${Math.round(compareRatio * 50)}%"></div>
            <div class="compare-avg-marker" style="left:50%"></div>
          </div>
          <div class="compare-label">${b.refLabel}: ${_formatRef(b)}</div>
        </div>
      `;
    } catch (err) {
      card.classList.remove('running');
      card.classList.add('error');
      resultEl.innerHTML = `<span class="bench-result-error"><i class="fa-solid fa-circle-xmark"></i> ${err.message || 'Failed'}</span>`;
    }

    runBtn.disabled = false;
    runBtn.innerHTML = '<i class="fa-solid fa-rotate-right"></i> Re-run';
  }

  function _formatRef(b) {
    if (b.id === 'memBW') return `${Results.REF.memBW} GB/s`;
    if (b.id === 'cpuFloat') return `${Results.REF.cpuFloat} Mflops`;
    if (b.id === 'json') return `${Results.REF.json} ops/s`;
    if (b.id === 'gpuCompute') return `${Results.REF.gpuCompute} TFLOPS`;
    if (b.id === 'gpuMem') return `${Results.REF.gpuMem} GB/s`;
    if (b.id === 'render') return `${Results.REF.render} FPS`;
    if (b.id === 'cpuInt') return `${Results.REF.cpuInt} Gops`;
    if (b.id === 'cpuMulti') return `${Results.REF.cpuMulti} Gops`;
    if (b.id === 'mlNNInf') return `${Results.REF.mlNNInf} inf/s`;
    if (b.id === 'mlConv2D') return `${Results.REF.mlConv2D} conv/s`;
    if (b.id === 'mlTensorOps') return `${Results.REF.mlTensorOps} GFLOPS`;
    if (b.id === 'mlKMeans') return `${Results.REF.mlKMeans} iter/s`;
    return '';
  }

  /* ---- Run All ---- */
  async function runAll() {
    if (running) return;
    running = true;

    const btn = document.getElementById('btn-run-all');
    btn.disabled = true;
    btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Running...';

    const overallBar = document.getElementById('overall-progress-bar');
    overallBar.style.width = '0%';
    document.getElementById('overall-progress').classList.remove('hidden');

    toast('Starting all benchmarks...', 'info');

    for (let i = 0; i < BENCHMARKS.length; i++) {
      const b = BENCHMARKS[i];
      const baseProgress = i / BENCHMARKS.length;
      const stepSize = 1 / BENCHMARKS.length;

      const card = document.getElementById(`bench-${b.id}`);
      const progBar = document.getElementById(`progress-${b.id}`);
      const resultEl = document.getElementById(`result-${b.id}`);
      const runBtn = card.querySelector('.bench-run-btn');

      card.classList.add('running');
      card.classList.remove('completed', 'error');
      runBtn.disabled = true;
      runBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i>';
      progBar.style.width = '0%';
      resultEl.innerHTML = '<span class="bench-result-placeholder"><i class="fa-solid fa-spinner fa-spin"></i> Running...</span>';

      try {
        const result = await b.run(pct => {
          progBar.style.width = `${Math.round(pct * 100)}%`;
          overallBar.style.width = `${Math.round((baseProgress + pct * stepSize) * 100)}%`;
        });

        benchResults[b.id] = result;
        const tier = b.tier(result);
        const compareRatio = Math.min(2, (result[b.key] || 0) / b.refVal);

        card.classList.remove('running');
        card.classList.add('completed');
        progBar.style.width = '100%';

        resultEl.innerHTML = `
          <div class="bench-score-row">
            <span class="score-value">${b.format(result)}</span>
            <span class="tier-badge tier-${tier}">${tier}</span>
          </div>
          <div class="bench-compare">
            <div class="compare-bar-bg">
              <div class="compare-bar-fill" style="width:${Math.round(compareRatio * 50)}%"></div>
              <div class="compare-avg-marker" style="left:50%"></div>
            </div>
            <div class="compare-label">${b.refLabel}: ${_formatRef(b)}</div>
          </div>
        `;
      } catch (err) {
        card.classList.remove('running');
        card.classList.add('error');
        resultEl.innerHTML = `<span class="bench-result-error"><i class="fa-solid fa-circle-xmark"></i> ${err.message || 'Failed'}</span>`;
      }

      runBtn.disabled = false;
      runBtn.innerHTML = '<i class="fa-solid fa-rotate-right"></i> Re-run';
    }

    overallBar.style.width = '100%';

    // Render results dashboard
    Results.render(benchResults);
    toast('All benchmarks complete!', 'success');

    running = false;
    btn.disabled = false;
    btn.innerHTML = '<i class="fa-solid fa-play"></i> Run All Benchmarks';
  }

  /* ---- Section toggle ---- */
  function _bindSectionToggles() {
    document.querySelectorAll('.section-header').forEach(header => {
      header.addEventListener('click', () => {
        const toggle = header.getAttribute('data-toggle');
        if (!toggle) return;
        const body = document.getElementById(toggle);
        if (!body) return;
        body.classList.toggle('collapsed');
        const icon = header.querySelector('.toggle-icon');
        if (icon) icon.style.transform = body.classList.contains('collapsed') ? 'rotate(180deg)' : '';
      });
    });
  }

  /* ---- Export button wiring ---- */
  function _bindExportButtons() {
    const copyBtn = document.getElementById('btn-copy');
    const dlBtn = document.getElementById('btn-download');
    const shareBtn = document.getElementById('btn-share');

    if (copyBtn) copyBtn.addEventListener('click', () => {
      if (Object.keys(benchResults).length === 0) { toast('Run benchmarks first!', 'warning'); return; }
      ExportUtils.copyToClipboard(benchResults, sysInfo);
    });
    if (dlBtn) dlBtn.addEventListener('click', () => {
      if (Object.keys(benchResults).length === 0) { toast('Run benchmarks first!', 'warning'); return; }
      ExportUtils.downloadJSON(benchResults, sysInfo);
    });
    if (shareBtn) shareBtn.addEventListener('click', () => {
      if (Object.keys(benchResults).length === 0) { toast('Run benchmarks first!', 'warning'); return; }
      ExportUtils.shareViaURL(benchResults);
    });
  }

  /* ---- Init ---- */
  async function init() {
    // Detect system info
    sysInfo = await SystemInfo.detect();
    SystemInfo.render(sysInfo);

    // Render benchmark cards
    _renderBenchCards();

    // Render stress test UI
    StressTest.render();

    // Render capabilities
    Capabilities.render();

    // Bind Run All
    document.getElementById('btn-run-all').addEventListener('click', runAll);

    // Bind section toggles
    _bindSectionToggles();

    // Bind export buttons
    _bindExportButtons();

    // Check URL hash for shared results
    _checkSharedResults();

    toast('System info detected. Ready to benchmark!', 'info');
  }

  function _checkSharedResults() {
    const hash = location.hash;
    if (!hash.startsWith('#results=')) return;
    try {
      const data = JSON.parse(atob(hash.slice(9)));
      toast(`Viewing shared result: Score ${data.s}`, 'info');
    } catch { /* ignore */ }
  }

  /* ---- Bootstrap ---- */
  document.addEventListener('DOMContentLoaded', () => init());

  return { toast, runAll, benchResults: () => benchResults, sysInfo: () => sysInfo };
})();
