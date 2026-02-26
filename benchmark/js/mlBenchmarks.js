/* =====================================================
   ML BENCHMARKS — AI / Machine Learning performance tests
   Tests 9-12: NN inference, Conv2D, Tensor ops, K-Means
   ===================================================== */

const MlBenchmarks = (() => { // eslint-disable-line no-unused-vars

  // =============== TEST 9: Neural Network Inference ===============
  // Multi-layer perceptron: 784 → 256 → 128 → 64 → 10
  // Simulates MNIST-style classification inference
  function nnInferenceBenchmark(durationMs = 3000, onProgress) {
    return new Promise(resolve => {
      // Build network weights (random init)
      const layers = [
        { w: _randomMatrix(256, 784), b: _randomVector(256) },
        { w: _randomMatrix(128, 256), b: _randomVector(128) },
        { w: _randomMatrix(64, 128),  b: _randomVector(64) },
        { w: _randomMatrix(10, 64),   b: _randomVector(10) }
      ];

      const input = _randomVector(784); // 28×28 flattened image
      let inferences = 0;
      const startTime = performance.now();
      const end = startTime + durationMs;

      function chunk() {
        const chunkEnd = Math.min(performance.now() + 50, end);
        while (performance.now() < chunkEnd) {
          _forwardPass(layers, input);
          inferences++;
        }
        const elapsed = performance.now() - startTime;
        if (onProgress) onProgress(Math.min(elapsed / durationMs, 1));

        if (performance.now() < end) {
          setTimeout(chunk, 0);
        } else {
          const totalElapsed = performance.now() - startTime;
          const infPerSec = (inferences / totalElapsed) * 1000;
          // Estimate FLOPS: each layer = 2*rows*cols (multiply-add) + rows (bias+ReLU)
          const flopsPerInf = layers.reduce((sum, l) =>
            sum + 2 * l.w.length * l.w[0].length + l.w.length, 0);
          const mflops = (infPerSec * flopsPerInf) / 1e6;

          resolve({
            infPerSec: Math.round(infPerSec),
            totalInferences: inferences,
            mflops: Math.round(mflops * 100) / 100,
            layers: layers.length,
            architecture: '784→256→128→64→10'
          });
        }
      }
      setTimeout(chunk, 0);
    });
  }

  function _forwardPass(layers, input) {
    let x = input;
    for (const layer of layers) {
      x = _denseLayer(layer.w, layer.b, x);
    }
    // Softmax on final layer
    return _softmax(x);
  }

  function _denseLayer(weights, bias, input) {
    const out = new Array(weights.length);
    for (let i = 0; i < weights.length; i++) {
      let sum = bias[i];
      const row = weights[i];
      for (let j = 0; j < row.length; j++) {
        sum += row[j] * input[j];
      }
      out[i] = sum > 0 ? sum : 0; // ReLU
    }
    return out;
  }

  function _softmax(x) {
    let max = -Infinity;
    for (let i = 0; i < x.length; i++) {
      if (x[i] > max) max = x[i];
    }
    const out = new Array(x.length);
    let sum = 0;
    for (let i = 0; i < x.length; i++) {
      out[i] = Math.exp(x[i] - max);
      sum += out[i];
    }
    for (let i = 0; i < out.length; i++) {
      out[i] /= sum;
    }
    return out;
  }

  // =============== TEST 10: 2D Convolution ===============
  // Applies 3×3 convolution kernels on 512×512 synthetic image
  function conv2dBenchmark(durationMs = 3000, onProgress) {
    return new Promise(resolve => {
      const H = 512, W = 512;
      const image = new Float32Array(H * W);
      for (let i = 0; i < H * W; i++) image[i] = Math.random();

      // Convolution kernels
      const kernels = {
        edgeDetect: [-1, -1, -1, -1, 8, -1, -1, -1, -1],
        sharpen:    [0, -1, 0, -1, 5, -1, 0, -1, 0],
        gaussBlur:  [1/16, 2/16, 1/16, 2/16, 4/16, 2/16, 1/16, 2/16, 1/16],
        emboss:     [-2, -1, 0, -1, 1, 1, 0, 1, 2],
        sobelX:     [-1, 0, 1, -2, 0, 2, -1, 0, 1]
      };
      const kernelList = Object.values(kernels);

      const output = new Float32Array(H * W);
      let convolutions = 0;
      const startTime = performance.now();
      const end = startTime + durationMs;

      function chunk() {
        const chunkEnd = Math.min(performance.now() + 50, end);
        while (performance.now() < chunkEnd) {
          const kernel = kernelList[convolutions % kernelList.length];
          _convolve2d(image, output, W, H, kernel, 3);
          convolutions++;
        }
        const elapsed = performance.now() - startTime;
        if (onProgress) onProgress(Math.min(elapsed / durationMs, 1));

        if (performance.now() < end) {
          setTimeout(chunk, 0);
        } else {
          const totalElapsed = performance.now() - startTime;
          const convPerSec = (convolutions / totalElapsed) * 1000;
          // Each pixel: 9 multiplies + 8 adds = 17 FLOPS, (H-2)*(W-2) pixels
          const pixelsPerConv = (H - 2) * (W - 2);
          const flopsPerConv = pixelsPerConv * 17;
          const mflops = (convPerSec * flopsPerConv) / 1e6;
          const mpixPerSec = (convPerSec * pixelsPerConv) / 1e6;

          resolve({
            convPerSec: Math.round(convPerSec * 100) / 100,
            totalConvolutions: convolutions,
            mflops: Math.round(mflops * 100) / 100,
            mpixPerSec: Math.round(mpixPerSec * 100) / 100,
            imageSize: `${W}×${H}`,
            kernelCount: kernelList.length
          });
        }
      }
      setTimeout(chunk, 0);
    });
  }

  function _convolve2d(input, output, w, h, kernel, kSize) {
    const half = (kSize - 1) / 2;
    for (let y = half; y < h - half; y++) {
      for (let x = half; x < w - half; x++) {
        let sum = 0;
        let ki = 0;
        for (let ky = -half; ky <= half; ky++) {
          for (let kx = -half; kx <= half; kx++) {
            sum += input[(y + ky) * w + (x + kx)] * kernel[ki++];
          }
        }
        output[y * w + x] = sum;
      }
    }
  }

  // =============== TEST 11: Tensor Operations ===============
  // Element-wise, transpose, batch normalization, softmax, GELU
  function tensorOpsBenchmark(durationMs = 3000, onProgress) {
    return new Promise(resolve => {
      const N = 1024;
      const a = new Float32Array(N * N);
      const b = new Float32Array(N * N);
      const c = new Float32Array(N * N);
      for (let i = 0; i < N * N; i++) {
        a[i] = Math.random() * 2 - 1;
        b[i] = Math.random() * 2 - 1;
      }

      let ops = 0;
      const startTime = performance.now();
      const end = startTime + durationMs;

      function chunk() {
        const chunkEnd = Math.min(performance.now() + 50, end);
        while (performance.now() < chunkEnd) {
          // Cycle through tensor operations
          switch (ops % 5) {
          case 0: _tensorAdd(a, b, c, N * N); break;
          case 1: _tensorMul(a, b, c, N * N); break;
          case 2: _tensorTranspose(a, c, N, N); break;
          case 3: _batchNorm(a, c, N, N); break;
          case 4: _tensorGelu(a, c, N * N); break;
          }
          ops++;
        }
        const elapsed = performance.now() - startTime;
        if (onProgress) onProgress(Math.min(elapsed / durationMs, 1));

        if (performance.now() < end) {
          setTimeout(chunk, 0);
        } else {
          const totalElapsed = performance.now() - startTime;
          const opsPerSec = (ops / totalElapsed) * 1000;
          // Each op processes N*N elements with varying FLOPS
          const elementsPerOp = N * N;
          const avgFlopsPerElement = 3; // average across ops
          const gflops = (opsPerSec * elementsPerOp * avgFlopsPerElement) / 1e9;

          resolve({
            opsPerSec: Math.round(opsPerSec * 100) / 100,
            totalOps: ops,
            gflops: Math.round(gflops * 1000) / 1000,
            tensorSize: `${N}×${N}`,
            elements: N * N
          });
        }
      }
      setTimeout(chunk, 0);
    });
  }

  function _tensorAdd(a, b, c, n) {
    for (let i = 0; i < n; i++) c[i] = a[i] + b[i];
  }

  function _tensorMul(a, b, c, n) {
    for (let i = 0; i < n; i++) c[i] = a[i] * b[i];
  }

  function _tensorTranspose(a, c, rows, cols) {
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        c[j * rows + i] = a[i * cols + j];
      }
    }
  }

  function _batchNorm(input, output, batchSize, features) {
    // Compute mean and variance per feature, then normalize
    for (let f = 0; f < features; f++) {
      let mean = 0;
      for (let b = 0; b < batchSize; b++) mean += input[b * features + f];
      mean /= batchSize;

      let variance = 0;
      for (let b = 0; b < batchSize; b++) {
        const diff = input[b * features + f] - mean;
        variance += diff * diff;
      }
      variance /= batchSize;
      const invStd = 1.0 / Math.sqrt(variance + 1e-5);

      for (let b = 0; b < batchSize; b++) {
        output[b * features + f] = (input[b * features + f] - mean) * invStd;
      }
    }
  }

  function _tensorGelu(input, output, n) {
    // GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
    const SQRT_2_PI = 0.7978845608; // √(2/π)
    for (let i = 0; i < n; i++) {
      const x = input[i];
      const cdf = 0.5 * (1.0 + Math.tanh(SQRT_2_PI * (x + 0.044715 * x * x * x)));
      output[i] = x * cdf;
    }
  }

  // =============== TEST 12: K-Means Clustering ===============
  // K-Means on 50,000 points in 16 dimensions with K=32
  function kMeansBenchmark(durationMs = 3000, onProgress) {
    return new Promise(resolve => {
      const N = 50000;
      const D = 16;
      const K = 32;

      // Generate synthetic data
      const data = new Float32Array(N * D);
      for (let i = 0; i < N * D; i++) data[i] = Math.random() * 10 - 5;

      // Initialize centroids (first K points)
      const centroids = new Float32Array(K * D);
      for (let i = 0; i < K * D; i++) centroids[i] = data[i];

      const assignments = new Int32Array(N);
      const counts = new Int32Array(K);
      const newCentroids = new Float32Array(K * D);

      let iterations = 0;
      const startTime = performance.now();
      const end = startTime + durationMs;

      function chunk() {
        const chunkEnd = Math.min(performance.now() + 80, end);
        while (performance.now() < chunkEnd) {
          // Assignment step
          for (let i = 0; i < N; i++) {
            let bestDist = Infinity;
            let bestK = 0;
            for (let k = 0; k < K; k++) {
              let dist = 0;
              for (let d = 0; d < D; d++) {
                const diff = data[i * D + d] - centroids[k * D + d];
                dist += diff * diff;
              }
              if (dist < bestDist) {
                bestDist = dist;
                bestK = k;
              }
            }
            assignments[i] = bestK;
          }

          // Update step
          newCentroids.fill(0);
          counts.fill(0);
          for (let i = 0; i < N; i++) {
            const k = assignments[i];
            counts[k]++;
            for (let d = 0; d < D; d++) {
              newCentroids[k * D + d] += data[i * D + d];
            }
          }
          for (let k = 0; k < K; k++) {
            if (counts[k] > 0) {
              for (let d = 0; d < D; d++) {
                centroids[k * D + d] = newCentroids[k * D + d] / counts[k];
              }
            }
          }
          iterations++;
        }
        const elapsed = performance.now() - startTime;
        if (onProgress) onProgress(Math.min(elapsed / durationMs, 1));

        if (performance.now() < end) {
          setTimeout(chunk, 0);
        } else {
          const totalElapsed = performance.now() - startTime;
          const itersPerSec = (iterations / totalElapsed) * 1000;
          // Flops: N*K*D*2 (distance) + N*D (update) per iteration
          const flopsPerIter = N * K * D * 2 + N * D;
          const gflops = (itersPerSec * flopsPerIter) / 1e9;

          resolve({
            itersPerSec: Math.round(itersPerSec * 100) / 100,
            totalIterations: iterations,
            gflops: Math.round(gflops * 1000) / 1000,
            dataPoints: N,
            dimensions: D,
            clusters: K
          });
        }
      }
      setTimeout(chunk, 0);
    });
  }

  // =============== Helpers ===============
  function _randomMatrix(rows, cols) {
    const m = new Array(rows);
    const scale = Math.sqrt(2.0 / cols); // He initialization
    for (let i = 0; i < rows; i++) {
      m[i] = new Float32Array(cols);
      for (let j = 0; j < cols; j++) {
        m[i][j] = _gaussianRandom() * scale;
      }
    }
    return m;
  }

  function _randomVector(n) {
    const v = new Float32Array(n);
    for (let i = 0; i < n; i++) v[i] = (Math.random() - 0.5) * 0.1;
    return v;
  }

  let _hasSpare = false;
  let _spare = 0;
  function _gaussianRandom() {
    if (_hasSpare) { _hasSpare = false; return _spare; }
    let u, v, s;
    do {
      u = Math.random() * 2 - 1;
      v = Math.random() * 2 - 1;
      s = u * u + v * v;
    } while (s >= 1 || s === 0);
    s = Math.sqrt(-2.0 * Math.log(s) / s);
    _spare = v * s;
    _hasSpare = true;
    return u * s;
  }

  // =============== Tier calculation ===============
  function getTierNNInference(infPerSec) {
    if (infPerSec > 8000) return 'S';
    if (infPerSec > 4000) return 'A';
    if (infPerSec > 1500) return 'B';
    if (infPerSec > 500)  return 'C';
    return 'D';
  }

  function getTierConv2D(convPerSec) {
    if (convPerSec > 100)  return 'S';
    if (convPerSec > 40)   return 'A';
    if (convPerSec > 15)   return 'B';
    if (convPerSec > 5)    return 'C';
    return 'D';
  }

  function getTierTensorOps(gflops) {
    if (gflops > 10)  return 'S';
    if (gflops > 4)   return 'A';
    if (gflops > 1.5) return 'B';
    if (gflops > 0.5) return 'C';
    return 'D';
  }

  function getTierKMeans(itersPerSec) {
    if (itersPerSec > 50) return 'S';
    if (itersPerSec > 20) return 'A';
    if (itersPerSec > 8)  return 'B';
    if (itersPerSec > 3)  return 'C';
    return 'D';
  }

  return {
    nnInferenceBenchmark,
    conv2dBenchmark,
    tensorOpsBenchmark,
    kMeansBenchmark,
    getTierNNInference,
    getTierConv2D,
    getTierTensorOps,
    getTierKMeans
  };
})();

if (typeof module !== 'undefined' && module.exports) module.exports = MlBenchmarks;
