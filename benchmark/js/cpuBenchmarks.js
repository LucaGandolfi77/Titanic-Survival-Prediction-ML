/* =====================================================
   CPU BENCHMARKS — Tests 1-5 (integer, float, multi, mem, json)
   ===================================================== */

const CpuBenchmarks = (() => {

  // =============== TEST 1: Single-Thread Integer ===============
  function integerBenchmark(durationMs = 2000, onProgress) {
    return new Promise(resolve => {
      const startTime = performance.now();
      let ops = 0;
      const a = 1, b = 2; let c = 0;
      const end = startTime + durationMs;

      function chunk() {
        const chunkEnd = Math.min(performance.now() + 50, end);
        while (performance.now() < chunkEnd) {
          // 100 integer ops per iteration
          c = a + b; c = c * 3; c = c - 1; c = c ^ 0xFF;
          c = (c << 2) | (c >> 6); c = c & 0xFFFF;
          c = c % 97; c = c + a; c = c * b; c = c - 7;
          c = a + b + 1; c = c * 2; c = c - 3; c = c ^ 0xAA;
          c = (c << 1) | (c >> 7); c = c & 0xFFF;
          c = c % 53; c = c + b; c = c * a; c = c - 2;
          c = a + 5; c = c * 7; c = c - 11; c = c ^ 0x55;
          c = (c << 3) | (c >> 5); c = c & 0xFF;
          c = c % 31; c = c + a; c = c * 4; c = c - 9;
          c = b + 3; c = c * 6; c = c - 8; c = c ^ 0xCC;
          c = (c << 4) | (c >> 4); c = c & 0xFFFF;
          c = c % 43; c = c + b; c = c * a; c = c - 5;
          c = a * b; c = c + 13; c = c - a; c = c ^ 0x77;
          c = (c << 2) | (c >> 3); c = c & 0xFFF;
          c = c % 67; c = c + 7; c = c * 3; c = c - b;
          c = a + b; c = c * 3; c = c - 1; c = c ^ 0xFF;
          c = (c << 2) | (c >> 6); c = c & 0xFFFF;
          c = c % 97; c = c + a; c = c * b; c = c - 7;
          c = a + b + 1; c = c * 2; c = c - 3; c = c ^ 0xAA;
          c = (c << 1) | (c >> 7); c = c & 0xFFF;
          c = c % 53; c = c + b; c = c * a; c = c - 2;
          c = a + 5; c = c * 7; c = c - 11; c = c ^ 0x55;
          c = (c << 3) | (c >> 5); c = c & 0xFF;
          c = c % 31; c = c + a; c = c * 4; c = c - 9;
          c = b + 3; c = c * 6; c = c - 8; c = c ^ 0xCC;
          ops++;
        }
        const elapsed = performance.now() - startTime;
        if (onProgress) onProgress(Math.min(elapsed / durationMs, 1));

        if (performance.now() < end) {
          setTimeout(chunk, 0);
        } else {
          const seconds = elapsed / 1000;
          const totalOps = ops * 100;
          const gops = totalOps / seconds / 1e9;
          resolve({ opsPerSec: totalOps / seconds, gops, checksum: c });
        }
      }
      chunk();
    });
  }

  // =============== TEST 2: Single-Thread Float (FP64) ===============
  function floatBenchmark(durationMs = 2000, onProgress) {
    return new Promise(resolve => {
      const startTime = performance.now();
      let ops = 0;
      let x = Math.PI, y = Math.E, z = 0;
      const end = startTime + durationMs;

      function chunk() {
        const chunkEnd = Math.min(performance.now() + 50, end);
        while (performance.now() < chunkEnd) {
          z = Math.sin(x) + Math.cos(y);
          z = Math.sqrt(Math.abs(z) + 0.001);
          z = Math.log(z + 1) * Math.exp(-z);
          z = Math.pow(x, 0.5) + Math.atan2(y, x);
          z = (z + Math.PI) % (2 * Math.PI);
          x = Math.sin(z + 0.1) + Math.PI;
          y = Math.cos(z * 0.5) + Math.E;
          z = Math.tan(x * 0.01) + Math.sqrt(y);
          z = Math.log2(Math.abs(z) + 1) + Math.log10(y + 1);
          z = Math.hypot(x, y) * 0.1;
          ops += 10;
        }
        const elapsed = performance.now() - startTime;
        if (onProgress) onProgress(Math.min(elapsed / durationMs, 1));

        if (performance.now() < end) {
          setTimeout(chunk, 0);
        } else {
          const seconds = elapsed / 1000;
          const mflops = ops / seconds / 1e6;
          resolve({ opsPerSec: ops / seconds, mflops, checksum: z });
        }
      }
      chunk();
    });
  }

  // =============== TEST 3: Multi-Thread (Web Workers) ===============
  function multiThreadBenchmark(onProgress) {
    return new Promise((resolve, reject) => {
      const numWorkers = navigator.hardwareConcurrency || 4;
      const duration = 3000;

      const workerCode = `
        self.onmessage = function(e) {
          const end = performance.now() + ${duration};
          let ops = 0, a = 1, b = 2, c = 0;
          while (performance.now() < end) {
            c = a + b; c = c * 3; c = c - 1; c = c ^ 0xFF;
            c = (c << 2) | (c >> 6); c = c & 0xFFFF;
            c = c % 97; c = c + a; c = c * b; c = c - 7;
            ops++;
          }
          self.postMessage({ ops: ops * 10 });
        };
      `;

      let blob, url;
      try {
        blob = new Blob([workerCode], { type: 'application/javascript' });
        url = URL.createObjectURL(blob);
      } catch (e) {
        reject(new Error('Cannot create Web Workers'));
        return;
      }

      // Simulate progress
      let progressInterval = null;
      let fakeProgress = 0;
      if (onProgress) {
        progressInterval = setInterval(() => {
          fakeProgress = Math.min(fakeProgress + 0.03, 0.95);
          onProgress(fakeProgress);
        }, 100);
      }

      const promises = Array.from({ length: numWorkers }, () => {
        return new Promise(res => {
          const w = new Worker(url);
          w.onmessage = e => { w.terminate(); res(e.data.ops); };
          w.onerror = () => { w.terminate(); res(0); };
          w.postMessage({});
        });
      });

      Promise.all(promises).then(results => {
        URL.revokeObjectURL(url);
        if (progressInterval) clearInterval(progressInterval);
        if (onProgress) onProgress(1);

        const totalOps = results.reduce((a, b) => a + b, 0);
        const perSecond = totalOps / (duration / 1000);

        resolve({
          totalOpsPerSec: perSecond,
          gops: perSecond / 1e9,
          perWorker: results.map(r => r / (duration / 1000)),
          numWorkers
        });
      }).catch(reject);
    });
  }

  // =============== TEST 4: Memory Bandwidth ===============
  function memoryBandwidthBenchmark(onProgress) {
    return new Promise(resolve => {
      if (onProgress) onProgress(0.1);

      const SIZE = 16 * 1024 * 1024; // 16M floats = 64 MB
      const arr = new Float32Array(SIZE);

      if (onProgress) onProgress(0.2);

      // Sequential write
      const t0 = performance.now();
      for (let i = 0; i < SIZE; i++) arr[i] = i * 0.001;
      const writeTime = performance.now() - t0;

      if (onProgress) onProgress(0.5);

      // Sequential read
      const t1 = performance.now();
      let sum = 0;
      for (let i = 0; i < SIZE; i++) sum += arr[i];
      const readTime = performance.now() - t1;

      if (onProgress) onProgress(0.8);

      // Random access
      const t2 = performance.now();
      let rsum = 0;
      for (let i = 0; i < SIZE; i++) {
        rsum += arr[(i * 7 + 13) & (SIZE - 1)];
      }
      const randomTime = performance.now() - t2;

      if (onProgress) onProgress(1);

      const bytes = SIZE * 4;
      resolve({
        writeBW: bytes / writeTime / 1e6,   // MB/s -> will display as GB/s
        readBW: bytes / readTime / 1e6,
        randomBW: bytes / randomTime / 1e6,
        checksum: sum + rsum
      });
    });
  }

  // =============== TEST 5: JSON / String Processing ===============
  function jsonBenchmark(durationMs = 2000, onProgress) {
    return new Promise(resolve => {
      // Build a large nested object (100 keys)
      const obj = {};
      for (let i = 0; i < 100; i++) {
        obj['key_' + i] = {
          id: i,
          name: 'item_' + i,
          value: Math.random() * 1000,
          active: i % 2 === 0,
          tags: ['alpha', 'beta', 'gamma'],
          nested: { x: i * 1.5, y: i * 2.5, label: 'nested_' + i }
        };
      }

      const startTime = performance.now();
      let ops = 0;
      const end = startTime + durationMs;

      function chunk() {
        const chunkEnd = Math.min(performance.now() + 50, end);
        while (performance.now() < chunkEnd) {
          const s = JSON.stringify(obj);
          JSON.parse(s);
          ops++;
        }
        const elapsed = performance.now() - startTime;
        if (onProgress) onProgress(Math.min(elapsed / durationMs, 1));

        if (performance.now() < end) {
          setTimeout(chunk, 0);
        } else {
          const seconds = elapsed / 1000;
          resolve({ opsPerSec: ops / seconds, totalOps: ops });
        }
      }
      chunk();
    });
  }

  // =============== Tier calculation ===============
  function getTierInt(gops) {
    if (gops > 2.0) return 'S';
    if (gops > 1.0) return 'A';
    if (gops > 0.5) return 'B';
    if (gops > 0.2) return 'C';
    return 'D';
  }

  function getTierFloat(mflops) {
    if (mflops > 50) return 'S';
    if (mflops > 25) return 'A';
    if (mflops > 10) return 'B';
    if (mflops > 3)  return 'C';
    return 'D';
  }

  function getTierMulti(gops) {
    if (gops > 10) return 'S';
    if (gops > 5)  return 'A';
    if (gops > 2)  return 'B';
    if (gops > 0.5) return 'C';
    return 'D';
  }

  function getTierMemory(readGBs) {
    if (readGBs > 10) return 'S';
    if (readGBs > 5)  return 'A';
    if (readGBs > 2)  return 'B';
    if (readGBs > 0.5) return 'C';
    return 'D';
  }

  function getTierJSON(ops) {
    if (ops > 5000)  return 'S';
    if (ops > 2000)  return 'A';
    if (ops > 800)   return 'B';
    if (ops > 200)   return 'C';
    return 'D';
  }

  return {
    integerBenchmark,
    floatBenchmark,
    multiThreadBenchmark,
    memoryBandwidthBenchmark,
    jsonBenchmark,
    getTierInt, getTierFloat, getTierMulti, getTierMemory, getTierJSON
  };
})();

if (typeof module !== 'undefined' && module.exports) module.exports = CpuBenchmarks;
