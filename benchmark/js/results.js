/* =====================================================
   RESULTS — Dashboard, score calculation, Chart.js charts
   ===================================================== */

const Results = (() => {

  // Average desktop 2024 reference values
  const REF = {
    cpuInt: 1.0,      // Gops
    cpuFloat: 20,     // Mflops
    cpuMulti: 5.0,    // Gops
    memBW: 5.0,       // GB/s read
    json: 1500,       // ops/s
    gpuCompute: 1.0,  // TFLOPS
    gpuMem: 50,       // GB/s
    render: 60,       // FPS
    mlNNInf: 2000,    // inferences/sec
    mlConv2D: 20,     // convolutions/sec
    mlTensorOps: 2.0, // GFLOPS
    mlKMeans: 10      // iterations/sec
  };

  const WEIGHTS = {
    cpuInt: 0.08,
    cpuFloat: 0.08,
    cpuMulti: 0.14,
    memBW: 0.08,
    json: 0.04,
    gpuCompute: 0.18,
    gpuMem: 0.08,
    render: 0.07,
    mlNNInf: 0.07,
    mlConv2D: 0.06,
    mlTensorOps: 0.06,
    mlKMeans: 0.06
  };

  function calculateScore(benchResults) {
    // Normalize each to 0-100 range (ref = 50), then weight
    const scores = {};
    const normalize = (val, ref) => Math.min(100, (val / ref) * 50);

    scores.cpuInt     = benchResults.cpuInt     ? normalize(benchResults.cpuInt.gops, REF.cpuInt) : 0;
    scores.cpuFloat   = benchResults.cpuFloat   ? normalize(benchResults.cpuFloat.mflops, REF.cpuFloat) : 0;
    scores.cpuMulti   = benchResults.cpuMulti   ? normalize(benchResults.cpuMulti.gops, REF.cpuMulti) : 0;
    scores.memBW      = benchResults.memBW      ? normalize(benchResults.memBW.readBW / 1000, REF.memBW) : 0;
    scores.json       = benchResults.json       ? normalize(benchResults.json.opsPerSec, REF.json) : 0;
    scores.gpuCompute = benchResults.gpuCompute ? normalize(benchResults.gpuCompute.tflops, REF.gpuCompute) : 0;
    scores.gpuMem     = benchResults.gpuMem     ? normalize(benchResults.gpuMem.gbps, REF.gpuMem) : 0;
    scores.render     = benchResults.render     ? normalize(benchResults.render.fps, REF.render) : 0;
    scores.mlNNInf    = benchResults.mlNNInf    ? normalize(benchResults.mlNNInf.infPerSec, REF.mlNNInf) : 0;
    scores.mlConv2D   = benchResults.mlConv2D   ? normalize(benchResults.mlConv2D.convPerSec, REF.mlConv2D) : 0;
    scores.mlTensorOps = benchResults.mlTensorOps ? normalize(benchResults.mlTensorOps.gflops, REF.mlTensorOps) : 0;
    scores.mlKMeans   = benchResults.mlKMeans   ? normalize(benchResults.mlKMeans.itersPerSec, REF.mlKMeans) : 0;

    let composite = 0;
    for (const key of Object.keys(WEIGHTS)) {
      composite += (scores[key] || 0) * WEIGHTS[key];
    }
    // Scale to 0-10000
    const totalScore = Math.round(composite * 100);

    return { scores, totalScore };
  }

  function getTotalTier(score) {
    if (score >= 7000) return { label: 'GAMING PC', cls: 'tier-gaming', icon: 'fa-gamepad' };
    if (score >= 5000) return { label: 'WORKSTATION', cls: 'tier-workstation', icon: 'fa-server' };
    if (score >= 3000) return { label: 'OFFICE PC', cls: 'tier-office', icon: 'fa-desktop' };
    if (score >= 1500) return { label: 'BUDGET', cls: 'tier-budget', icon: 'fa-laptop' };
    return { label: 'MOBILE / LOW-END', cls: 'tier-mobile', icon: 'fa-mobile-screen' };
  }

  function render(benchResults) {
    const section = document.getElementById('section-results');
    section.classList.remove('hidden');

    const { scores, totalScore } = calculateScore(benchResults);
    const tier = getTotalTier(totalScore);

    const content = document.getElementById('results-content');
    content.innerHTML = `
      <div class="results-grid">
        <!-- Overall Score -->
        <div class="result-score-card anim-slide-up">
          <div class="result-overall-label">Overall Score</div>
          <div class="result-overall-score" id="result-score-num">0</div>
          <div class="result-overall-max">/ 10,000</div>
          <div class="result-tier-badge ${tier.cls}">
            <i class="fa-solid ${tier.icon}"></i> ${tier.label}
          </div>
          <div class="result-breakdown">
            <div class="breakdown-item">
              <div class="breakdown-value">${Math.round(scores.cpuInt + scores.cpuFloat + scores.cpuMulti + scores.json)}</div>
              <div class="breakdown-label">CPU Score</div>
              <div class="breakdown-pct">${Math.round((WEIGHTS.cpuInt + WEIGHTS.cpuFloat + WEIGHTS.cpuMulti + WEIGHTS.json) * 100)}% weight</div>
            </div>
            <div class="breakdown-item">
              <div class="breakdown-value">${Math.round(scores.gpuCompute + scores.gpuMem + scores.render)}</div>
              <div class="breakdown-label">GPU Score</div>
              <div class="breakdown-pct">${Math.round((WEIGHTS.gpuCompute + WEIGHTS.gpuMem + WEIGHTS.render) * 100)}% weight</div>
            </div>
            <div class="breakdown-item">
              <div class="breakdown-value">${Math.round(scores.memBW)}</div>
              <div class="breakdown-label">Memory Score</div>
              <div class="breakdown-pct">${Math.round(WEIGHTS.memBW * 100)}% weight</div>
            </div>
            <div class="breakdown-item">
              <div class="breakdown-value">${Math.round(scores.mlNNInf + scores.mlConv2D + scores.mlTensorOps + scores.mlKMeans)}</div>
              <div class="breakdown-label">AI/ML Score</div>
              <div class="breakdown-pct">${Math.round((WEIGHTS.mlNNInf + WEIGHTS.mlConv2D + WEIGHTS.mlTensorOps + WEIGHTS.mlKMeans) * 100)}% weight</div>
            </div>
          </div>
        </div>

        <!-- Radar Chart -->
        <div class="result-chart-card anim-slide-up">
          <h3><i class="fa-solid fa-chart-radar"></i> Performance Radar</h3>
          <div class="chart-wrap"><canvas id="radar-chart"></canvas></div>
        </div>

        <!-- CPU Bar Chart -->
        <div class="result-chart-card anim-slide-up">
          <h3><i class="fa-solid fa-microchip"></i> CPU Benchmarks</h3>
          <div class="chart-wrap"><canvas id="cpu-bar-chart"></canvas></div>
        </div>

        <!-- GPU Bar Chart -->
        <div class="result-chart-card anim-slide-up">
          <h3><i class="fa-solid fa-display"></i> GPU Benchmarks</h3>
          <div class="chart-wrap"><canvas id="gpu-bar-chart"></canvas></div>
        </div>

        <!-- AI/ML Bar Chart -->
        <div class="result-chart-card anim-slide-up">
          <h3><i class="fa-solid fa-brain"></i> AI / ML Benchmarks</h3>
          <div class="chart-wrap"><canvas id="ml-bar-chart"></canvas></div>
        </div>

        <!-- Per-core chart (if multi-thread data) -->
        ${benchResults.cpuMulti ? `
        <div class="result-chart-card anim-slide-up">
          <h3><i class="fa-solid fa-layer-group"></i> Per-Core Performance</h3>
          <div class="chart-wrap"><canvas id="percore-chart"></canvas></div>
        </div>
        ` : ''}

        <!-- History -->
        <div class="result-chart-card anim-slide-up" ${!benchResults.cpuMulti ? 'style="grid-column:1/-1"' : ''}>
          <h3><i class="fa-solid fa-clock-rotate-left"></i> Score History</h3>
          <div id="history-content"></div>
        </div>
      </div>
    `;

    // Animate score count-up
    _animateCountUp('result-score-num', totalScore, 1500);

    // Build charts
    setTimeout(() => {
      _buildRadar(scores);
      _buildCpuBar(benchResults);
      _buildGpuBar(benchResults);
      _buildMlBar(benchResults);
      if (benchResults.cpuMulti) _buildPercoreBar(benchResults.cpuMulti);
      _renderHistory(benchResults, totalScore);
    }, 100);

    // Save to history
    _saveRun(benchResults, totalScore);

    // Scroll to results
    section.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  function _animateCountUp(elementId, target, durationMs) {
    const el = document.getElementById(elementId);
    if (!el) return;
    const start = performance.now();
    function step(now) {
      const elapsed = now - start;
      const progress = Math.min(elapsed / durationMs, 1);
      const eased = 1 - Math.pow(1 - progress, 4); // easeOutQuart
      el.textContent = Math.round(target * eased).toLocaleString();
      if (progress < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
  }

  function _buildRadar(scores) {
    const ctx = document.getElementById('radar-chart');
    if (!ctx) return;

    new Chart(ctx, {
      type: 'radar',
      data: {
        labels: ['CPU Speed', 'Multi-Core', 'Memory', 'GPU Compute', 'GPU Render', 'AI / ML', 'Overall'],
        datasets: [
          {
            label: 'Your PC',
            data: [
              scores.cpuInt,
              scores.cpuMulti,
              scores.memBW,
              scores.gpuCompute,
              scores.render,
              (scores.mlNNInf + scores.mlConv2D + scores.mlTensorOps + scores.mlKMeans) / 4,
              (scores.cpuInt + scores.cpuMulti + scores.memBW + scores.gpuCompute + scores.render +
                (scores.mlNNInf + scores.mlConv2D + scores.mlTensorOps + scores.mlKMeans) / 4) / 6
            ],
            borderColor: '#00d4ff',
            backgroundColor: 'rgba(0, 212, 255, 0.15)',
            borderWidth: 2,
            pointBackgroundColor: '#00d4ff'
          },
          {
            label: 'Average Desktop 2024',
            data: [50, 50, 50, 50, 50, 50, 50],
            borderColor: '#64748b',
            backgroundColor: 'rgba(100, 116, 139, 0.05)',
            borderWidth: 1,
            borderDash: [5, 5],
            pointBackgroundColor: '#64748b'
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          r: {
            beginAtZero: true,
            max: 100,
            grid: { color: 'rgba(42,42,61,0.5)' },
            angleLines: { color: 'rgba(42,42,61,0.5)' },
            pointLabels: { color: '#e2e8f0', font: { size: 11, family: 'Inter' } },
            ticks: { display: false }
          }
        },
        plugins: {
          legend: {
            labels: { color: '#e2e8f0', font: { size: 11 } }
          }
        }
      }
    });
  }

  function _buildCpuBar(r) {
    const ctx = document.getElementById('cpu-bar-chart');
    if (!ctx) return;

    const labels = ['Integer', 'Float', 'Multi-Thread', 'Memory BW', 'JSON'];
    const yours = [
      r.cpuInt ? r.cpuInt.gops : 0,
      r.cpuFloat ? r.cpuFloat.mflops / 10 : 0, // scale
      r.cpuMulti ? r.cpuMulti.gops : 0,
      r.memBW ? r.memBW.readBW / 1000 : 0,
      r.json ? r.json.opsPerSec / 1000 : 0
    ];
    const avg = [
      REF.cpuInt,
      REF.cpuFloat / 10,
      REF.cpuMulti,
      REF.memBW,
      REF.json / 1000
    ];

    new Chart(ctx, {
      type: 'bar',
      data: {
        labels,
        datasets: [
          {
            label: 'Your PC',
            data: yours,
            backgroundColor: 'rgba(245, 158, 11, 0.7)',
            borderColor: '#f59e0b',
            borderWidth: 1
          },
          {
            label: 'Average 2024',
            data: avg,
            backgroundColor: 'rgba(100, 116, 139, 0.3)',
            borderColor: '#64748b',
            borderWidth: 1
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: { grid: { display: false }, ticks: { color: '#64748b', font: { size: 10 } } },
          y: { grid: { color: 'rgba(42,42,61,0.4)' }, ticks: { color: '#64748b', font: { size: 10 } } }
        },
        plugins: {
          legend: { labels: { color: '#e2e8f0', font: { size: 11 } } }
        }
      }
    });
  }

  function _buildGpuBar(r) {
    const ctx = document.getElementById('gpu-bar-chart');
    if (!ctx) return;

    const labels = ['Compute (TFLOPS)', 'Mem BW (GB/s)', 'Render (FPS)'];
    const yours = [
      r.gpuCompute ? r.gpuCompute.tflops : 0,
      r.gpuMem ? r.gpuMem.gbps / 10 : 0,
      r.render ? r.render.fps / 10 : 0
    ];
    const avg = [REF.gpuCompute, REF.gpuMem / 10, REF.render / 10];

    new Chart(ctx, {
      type: 'bar',
      data: {
        labels,
        datasets: [
          {
            label: 'Your PC',
            data: yours,
            backgroundColor: 'rgba(0, 212, 255, 0.7)',
            borderColor: '#00d4ff',
            borderWidth: 1
          },
          {
            label: 'Average 2024',
            data: avg,
            backgroundColor: 'rgba(100, 116, 139, 0.3)',
            borderColor: '#64748b',
            borderWidth: 1
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: { grid: { display: false }, ticks: { color: '#64748b', font: { size: 10 } } },
          y: { grid: { color: 'rgba(42,42,61,0.4)' }, ticks: { color: '#64748b', font: { size: 10 } } }
        },
        plugins: {
          legend: { labels: { color: '#e2e8f0', font: { size: 11 } } }
        }
      }
    });
  }

  function _buildMlBar(r) {
    const ctx = document.getElementById('ml-bar-chart');
    if (!ctx) return;

    const labels = ['NN Inference', 'Conv2D', 'Tensor Ops', 'K-Means'];
    // Normalize values to similar scale for visual comparison
    const yours = [
      r.mlNNInf ? r.mlNNInf.infPerSec / 100 : 0,
      r.mlConv2D ? r.mlConv2D.convPerSec : 0,
      r.mlTensorOps ? r.mlTensorOps.gflops : 0,
      r.mlKMeans ? r.mlKMeans.itersPerSec : 0
    ];
    const avg = [
      REF.mlNNInf / 100,
      REF.mlConv2D,
      REF.mlTensorOps,
      REF.mlKMeans
    ];

    new Chart(ctx, {
      type: 'bar',
      data: {
        labels,
        datasets: [
          {
            label: 'Your PC',
            data: yours,
            backgroundColor: 'rgba(236, 72, 153, 0.7)',
            borderColor: '#ec4899',
            borderWidth: 1
          },
          {
            label: 'Average 2024',
            data: avg,
            backgroundColor: 'rgba(100, 116, 139, 0.3)',
            borderColor: '#64748b',
            borderWidth: 1
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: { grid: { display: false }, ticks: { color: '#64748b', font: { size: 10 } } },
          y: { grid: { color: 'rgba(42,42,61,0.4)' }, ticks: { color: '#64748b', font: { size: 10 } } }
        },
        plugins: {
          legend: { labels: { color: '#e2e8f0', font: { size: 11 } } }
        }
      }
    });
  }

  function _buildPercoreBar(multi) {
    const ctx = document.getElementById('percore-chart');
    if (!ctx) return;

    const labels = multi.perWorker.map((_, i) => 'Core ' + (i + 1));
    new Chart(ctx, {
      type: 'bar',
      data: {
        labels,
        datasets: [{
          label: 'Ops/sec per core',
          data: multi.perWorker.map(v => v / 1e6),
          backgroundColor: multi.perWorker.map((_, i) => {
            const hue = (i / multi.numWorkers) * 360;
            return `hsla(${hue}, 70%, 55%, 0.7)`;
          }),
          borderWidth: 0
        }]
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: { grid: { color: 'rgba(42,42,61,0.4)' }, ticks: { color: '#64748b', font: { size: 10, family: 'JetBrains Mono' } } },
          y: { grid: { display: false }, ticks: { color: '#64748b', font: { size: 10, family: 'JetBrains Mono' } } }
        },
        plugins: {
          legend: { display: false }
        }
      }
    });
  }

  function _saveRun(benchResults, totalScore) {
    const history = JSON.parse(localStorage.getItem('benchmark_history') || '[]');
    history.push({
      date: new Date().toISOString(),
      score: totalScore,
      cpuScore: benchResults.cpuInt ? Math.round(benchResults.cpuInt.gops * 1000) : 0,
      gpuScore: benchResults.gpuCompute ? Math.round(benchResults.gpuCompute.gflops) : 0,
      browser: navigator.userAgent.split(' ').pop()
    });
    if (history.length > 5) history.shift();
    localStorage.setItem('benchmark_history', JSON.stringify(history));
  }

  function _renderHistory() {
    const el = document.getElementById('history-content');
    if (!el) return;
    const history = JSON.parse(localStorage.getItem('benchmark_history') || '[]');

    if (history.length === 0) {
      el.innerHTML = '<p style="color:var(--text-muted);font-size:0.8rem">No previous runs</p>';
      return;
    }

    let rows = '';
    for (let i = history.length - 1; i >= 0; i--) {
      const h = history[i];
      const date = new Date(h.date).toLocaleString();
      let delta = '';
      if (i > 0) {
        const prev = history[i - 1];
        const diff = h.score - prev.score;
        const pct = ((diff / prev.score) * 100).toFixed(1);
        const cls = diff >= 0 ? 'delta-positive' : 'delta-negative';
        delta = `<span class="${cls}">${diff >= 0 ? '+' : ''}${pct}%</span>`;
      }
      rows += `<tr>
        <td>${date}</td>
        <td>${h.cpuScore}</td>
        <td>${h.gpuScore}</td>
        <td>${h.score.toLocaleString()}</td>
        <td>${delta || '—'}</td>
      </tr>`;
    }

    el.innerHTML = `
      <table class="history-table">
        <thead><tr><th>Date</th><th>CPU</th><th>GPU</th><th>Total</th><th>Δ</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
    `;
  }

  return { calculateScore, getTotalTier, render, REF };
})();

if (typeof module !== 'undefined' && module.exports) module.exports = Results;
