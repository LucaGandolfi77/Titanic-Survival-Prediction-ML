/* =====================================================
   EXPORT — Clipboard, JSON download, URL sharing
   ===================================================== */

const ExportUtils = (() => {

  function _formatTextReport(benchResults, sysInfo) {
    const { totalScore } = Results.calculateScore(benchResults);
    const tier = Results.getTotalTier(totalScore);
    const lines = [];

    lines.push('═══════════════════════════════════════');
    lines.push('  PC BENCHMARK SUITE — Results Report  ');
    lines.push('═══════════════════════════════════════');
    lines.push(`  Date: ${new Date().toLocaleString()}`);
    lines.push(`  Overall Score: ${totalScore.toLocaleString()} / 10,000`);
    lines.push(`  Tier: ${tier.label}`);
    lines.push('');

    if (sysInfo) {
      lines.push('── System Info ──');
      if (sysInfo.cpu) {
        lines.push(`  CPU Cores: ${sysInfo.cpu.cores}`);
        lines.push(`  Architecture: ${sysInfo.cpu.architecture}`);
        lines.push(`  OS: ${sysInfo.cpu.os}`);
        lines.push(`  Browser: ${sysInfo.cpu.browser} ${sysInfo.cpu.browserVersion}`);
        if (sysInfo.cpu.deviceMemory) lines.push(`  Device Memory: ${sysInfo.cpu.deviceMemory} GB`);
      }
      if (sysInfo.gpu && sysInfo.gpu.available) {
        lines.push(`  GPU: ${sysInfo.gpu.device || 'Unknown'}`);
        lines.push(`  GPU Vendor: ${sysInfo.gpu.vendor || 'Unknown'}`);
        lines.push(`  GPU Backend: ${sysInfo.gpu.backend || 'Unknown'}`);
      }
      lines.push('');
    }

    lines.push('── CPU Benchmarks ──');
    if (benchResults.cpuInt) {
      lines.push(`  Integer: ${benchResults.cpuInt.gops.toFixed(2)} Gops/s`);
    }
    if (benchResults.cpuFloat) {
      lines.push(`  Float: ${benchResults.cpuFloat.mflops.toFixed(1)} Mflops/s`);
    }
    if (benchResults.cpuMulti) {
      lines.push(`  Multi-Thread: ${benchResults.cpuMulti.gops.toFixed(2)} Gops/s (${benchResults.cpuMulti.numWorkers} workers)`);
    }
    if (benchResults.memBW) {
      lines.push(`  Memory Read: ${benchResults.memBW.readBW.toFixed(0)} MB/s`);
      lines.push(`  Memory Write: ${benchResults.memBW.writeBW.toFixed(0)} MB/s`);
    }
    if (benchResults.json) {
      lines.push(`  JSON: ${benchResults.json.opsPerSec.toFixed(0)} ops/s`);
    }
    lines.push('');

    lines.push('── GPU Benchmarks ──');
    if (benchResults.gpuCompute) {
      lines.push(`  Matrix Multiply: ${benchResults.gpuCompute.gflops.toFixed(1)} GFLOPS (${benchResults.gpuCompute.tflops.toFixed(3)} TFLOPS)`);
    }
    if (benchResults.gpuMem) {
      lines.push(`  Memory Bandwidth: ${benchResults.gpuMem.gbps.toFixed(1)} GB/s`);
    }
    if (benchResults.render) {
      lines.push(`  WebGL Render: ${benchResults.render.fps.toFixed(1)} FPS (${benchResults.render.mTriPerSec.toFixed(2)} M tri/s)`);
    }
    lines.push('');
    lines.push('═══════════════════════════════════════');

    return lines.join('\n');
  }

  async function copyToClipboard(benchResults, sysInfo) {
    const text = _formatTextReport(benchResults, sysInfo);
    try {
      await navigator.clipboard.writeText(text);
      App.toast('Results copied to clipboard!', 'success');
    } catch {
      // Fallback
      const ta = document.createElement('textarea');
      ta.value = text;
      document.body.appendChild(ta);
      ta.select();
      document.execCommand('copy');
      document.body.removeChild(ta);
      App.toast('Results copied to clipboard!', 'success');
    }
  }

  function downloadJSON(benchResults, sysInfo) {
    const data = {
      meta: {
        app: 'PC Benchmark Suite',
        version: '1.0.0',
        date: new Date().toISOString(),
        url: location.href
      },
      system: sysInfo || {},
      benchmarks: benchResults,
      scores: Results.calculateScore(benchResults)
    };

    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `benchmark_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
    App.toast('JSON downloaded!', 'success');
  }

  function shareViaURL(benchResults) {
    const { totalScore, scores } = Results.calculateScore(benchResults);
    const mini = {
      s: totalScore,
      ci: Math.round(scores.cpuInt),
      cf: Math.round(scores.cpuFloat),
      cm: Math.round(scores.cpuMulti),
      mb: Math.round(scores.memBW),
      gc: Math.round(scores.gpuCompute),
      gm: Math.round(scores.gpuMem),
      r: Math.round(scores.render)
    };
    const encoded = btoa(JSON.stringify(mini));
    const shareURL = `${location.origin}${location.pathname}#results=${encoded}`;

    navigator.clipboard.writeText(shareURL).then(() => {
      App.toast('Share URL copied to clipboard!', 'success');
    }).catch(() => {
      prompt('Copy this URL:', shareURL);
    });
  }

  return { copyToClipboard, downloadJSON, shareViaURL };
})();

if (typeof module !== 'undefined' && module.exports) module.exports = ExportUtils;
