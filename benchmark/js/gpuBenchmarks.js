/* =====================================================
   GPU BENCHMARKS — Tests 6-8 (WebGPU matmul, GPU mem BW, WebGL render)
   ===================================================== */

const GpuBenchmarks = (() => {

  // =============== TEST 6: WebGPU Compute — Matrix Multiplication ===============
  async function matMulBenchmark(onProgress) {
    if (!navigator.gpu) throw new Error('WebGPU not supported in this browser');

    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error('No WebGPU adapter found');

    const device = await adapter.requestDevice({
      requiredLimits: {
        maxBufferSize: Math.min(adapter.limits.maxBufferSize, 256 * 1024 * 1024),
        maxStorageBufferBindingSize: Math.min(adapter.limits.maxStorageBufferBindingSize, 256 * 1024 * 1024)
      }
    });

    const N = 1024; // 1024×1024 matrix
    const matrixSize = N * N * 4; // bytes (Float32)
    if (onProgress) onProgress(0.1);

    // Create buffers
    const bufA = device.createBuffer({ size: matrixSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const bufB = device.createBuffer({ size: matrixSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const bufC = device.createBuffer({ size: matrixSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const bufDims = device.createBuffer({ size: 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

    // Fill random data
    const dataA = new Float32Array(N * N);
    const dataB = new Float32Array(N * N);
    for (let i = 0; i < N * N; i++) {
      dataA[i] = Math.random() * 2 - 1;
      dataB[i] = Math.random() * 2 - 1;
    }
    device.queue.writeBuffer(bufA, 0, dataA);
    device.queue.writeBuffer(bufB, 0, dataB);
    device.queue.writeBuffer(bufDims, 0, new Uint32Array([N]));

    if (onProgress) onProgress(0.2);

    // WGSL tiled matrix multiplication shader
    const shaderModule = device.createShaderModule({
      code: `
        @group(0) @binding(0) var<storage, read> A : array<f32>;
        @group(0) @binding(1) var<storage, read> B : array<f32>;
        @group(0) @binding(2) var<storage, read_write> C : array<f32>;
        @group(0) @binding(3) var<uniform> dims : u32;

        const TILE_SIZE : u32 = 16u;

        var<workgroup> tileA : array<array<f32, 16>, 16>;
        var<workgroup> tileB : array<array<f32, 16>, 16>;

        @compute @workgroup_size(16, 16)
        fn main(
          @builtin(global_invocation_id) gid : vec3<u32>,
          @builtin(local_invocation_id)  lid : vec3<u32>
        ) {
          let N = dims;
          let row = gid.y;
          let col = gid.x;
          var sum : f32 = 0.0;

          let numTiles = N / TILE_SIZE;

          for (var t : u32 = 0u; t < numTiles; t = t + 1u) {
            let tileCol = t * TILE_SIZE + lid.x;
            let tileRow = t * TILE_SIZE + lid.y;

            if (row < N && tileCol < N) {
              tileA[lid.y][lid.x] = A[row * N + tileCol];
            } else {
              tileA[lid.y][lid.x] = 0.0;
            }

            if (tileRow < N && col < N) {
              tileB[lid.y][lid.x] = B[tileRow * N + col];
            } else {
              tileB[lid.y][lid.x] = 0.0;
            }

            workgroupBarrier();

            for (var k : u32 = 0u; k < TILE_SIZE; k = k + 1u) {
              sum = sum + tileA[lid.y][k] * tileB[k][lid.x];
            }

            workgroupBarrier();
          }

          if (row < N && col < N) {
            C[row * N + col] = sum;
          }
        }
      `
    });

    const pipeline = device.createComputePipeline({
      layout: 'auto',
      compute: { module: shaderModule, entryPoint: 'main' }
    });

    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: bufA } },
        { binding: 1, resource: { buffer: bufB } },
        { binding: 2, resource: { buffer: bufC } },
        { binding: 3, resource: { buffer: bufDims } }
      ]
    });

    const workgroupsX = Math.ceil(N / 16);
    const workgroupsY = Math.ceil(N / 16);

    if (onProgress) onProgress(0.3);

    // Warmup: 3 runs
    for (let i = 0; i < 3; i++) {
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(workgroupsX, workgroupsY);
      pass.end();
      device.queue.submit([encoder.finish()]);
    }
    await device.queue.onSubmittedWorkDone();

    if (onProgress) onProgress(0.4);

    // Timed: 10 runs
    const iterations = 10;
    const t0 = performance.now();
    for (let i = 0; i < iterations; i++) {
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(workgroupsX, workgroupsY);
      pass.end();
      device.queue.submit([encoder.finish()]);
      if (onProgress) onProgress(0.4 + (i / iterations) * 0.5);
    }
    await device.queue.onSubmittedWorkDone();
    const elapsed = (performance.now() - t0) / 1000; // seconds

    // 2 * N^3 FLOPs per matmul
    const flopsPerRun = 2 * N * N * N;
    const totalFlops = flopsPerRun * iterations;
    const gflops = totalFlops / elapsed / 1e9;
    const tflops = gflops / 1000;

    // Cleanup
    bufA.destroy();
    bufB.destroy();
    bufC.destroy();
    bufDims.destroy();
    device.destroy();

    if (onProgress) onProgress(1);

    return { gflops, tflops, matrixSize: N, iterations, elapsed, backend: 'WebGPU' };
  }

  // =============== TEST 7: GPU Memory Bandwidth ===============
  async function gpuMemoryBandwidthBenchmark(onProgress) {
    if (!navigator.gpu) throw new Error('WebGPU not supported in this browser');

    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error('No WebGPU adapter found');

    const maxBuf = Math.min(adapter.limits.maxBufferSize, 128 * 1024 * 1024);
    const device = await adapter.requestDevice({
      requiredLimits: {
        maxBufferSize: maxBuf,
        maxStorageBufferBindingSize: maxBuf
      }
    });

    const SIZE = Math.min(64 * 1024 * 1024, Math.floor(maxBuf / 4)); // number of f32 elements
    const byteSize = SIZE * 4;

    if (onProgress) onProgress(0.1);

    const bufSrc = device.createBuffer({ size: byteSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const bufDst = device.createBuffer({ size: byteSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const bufSize = device.createBuffer({ size: 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

    // Fill source
    const srcData = new Float32Array(SIZE);
    for (let i = 0; i < SIZE; i++) srcData[i] = i * 0.001;
    device.queue.writeBuffer(bufSrc, 0, srcData);
    device.queue.writeBuffer(bufSize, 0, new Uint32Array([SIZE]));

    if (onProgress) onProgress(0.2);

    // Copy shader
    const shaderModule = device.createShaderModule({
      code: `
        @group(0) @binding(0) var<storage, read> src : array<f32>;
        @group(0) @binding(1) var<storage, read_write> dst : array<f32>;
        @group(0) @binding(2) var<uniform> count : u32;

        @compute @workgroup_size(256)
        fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
          let idx = gid.x;
          if (idx < count) {
            dst[idx] = src[idx];
          }
        }
      `
    });

    const pipeline = device.createComputePipeline({
      layout: 'auto',
      compute: { module: shaderModule, entryPoint: 'main' }
    });

    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: bufSrc } },
        { binding: 1, resource: { buffer: bufDst } },
        { binding: 2, resource: { buffer: bufSize } }
      ]
    });

    const workgroups = Math.ceil(SIZE / 256);

    // Warmup
    for (let i = 0; i < 3; i++) {
      const enc = device.createCommandEncoder();
      const pass = enc.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(workgroups);
      pass.end();
      device.queue.submit([enc.finish()]);
    }
    await device.queue.onSubmittedWorkDone();

    if (onProgress) onProgress(0.4);

    // Timed: 10 runs
    const iters = 10;
    const t0 = performance.now();
    for (let i = 0; i < iters; i++) {
      const enc = device.createCommandEncoder();
      const pass = enc.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(workgroups);
      pass.end();
      device.queue.submit([enc.finish()]);
      if (onProgress) onProgress(0.4 + (i / iters) * 0.5);
    }
    await device.queue.onSubmittedWorkDone();
    const elapsed = (performance.now() - t0) / 1000;

    // Read + write = 2 * byteSize per iteration
    const totalBytes = 2 * byteSize * iters;
    const gbps = totalBytes / elapsed / 1e9;

    bufSrc.destroy();
    bufDst.destroy();
    bufSize.destroy();
    device.destroy();

    if (onProgress) onProgress(1);

    return { gbps, totalBytes, elapsed, iterations: iters };
  }

  // =============== TEST 8: WebGL Render Performance ===============
  function webglRenderBenchmark(onProgress) {
    return new Promise((resolve, reject) => {
      let canvas;
      try {
        canvas = new OffscreenCanvas(1920, 1080);
      } catch (e) {
        canvas = document.createElement('canvas');
        canvas.width = 1920;
        canvas.height = 1080;
      }

      const gl = canvas.getContext('webgl2');
      if (!gl) {
        reject(new Error('WebGL2 not supported'));
        return;
      }

      // Vertex shader
      const vsSource = `#version 300 es
        in vec2 aPosition;
        uniform vec2 uOffset;
        uniform float uScale;
        void main() {
          gl_Position = vec4(aPosition * uScale + uOffset, 0.0, 1.0);
        }
      `;

      // Fragment shader
      const fsSource = `#version 300 es
        precision mediump float;
        uniform vec4 uColor;
        out vec4 fragColor;
        void main() {
          fragColor = uColor;
        }
      `;

      function compileShader(src, type) {
        const s = gl.createShader(type);
        gl.shaderSource(s, src);
        gl.compileShader(s);
        if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
          throw new Error(gl.getShaderInfoLog(s));
        }
        return s;
      }

      const vs = compileShader(vsSource, gl.VERTEX_SHADER);
      const fs = compileShader(fsSource, gl.FRAGMENT_SHADER);
      const prog = gl.createProgram();
      gl.attachShader(prog, vs);
      gl.attachShader(prog, fs);
      gl.linkProgram(prog);
      if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
        reject(new Error(gl.getProgramInfoLog(prog)));
        return;
      }
      gl.useProgram(prog);

      const aPos = gl.getAttribLocation(prog, 'aPosition');
      const uOffset = gl.getUniformLocation(prog, 'uOffset');
      const uScale = gl.getUniformLocation(prog, 'uScale');
      const uColor = gl.getUniformLocation(prog, 'uColor');

      // Quad (2 triangles)
      const quadVerts = new Float32Array([
        -1, -1,  1, -1,  1,  1,
        -1, -1,  1,  1, -1,  1
      ]);
      const vbo = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
      gl.bufferData(gl.ARRAY_BUFFER, quadVerts, gl.STATIC_DRAW);
      gl.enableVertexAttribArray(aPos);
      gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 0, 0);

      const DRAW_CALLS_PER_FRAME = 5000;
      const durationMs = 5000;
      let frames = 0;
      let drawCalls = 0;
      const startTime = performance.now();

      // Use pseudo-random for deterministic pattern
      let seed = 12345;
      function pseudoRandom() {
        seed = (seed * 16807 + 0) % 2147483647;
        return seed / 2147483647;
      }

      function renderFrame() {
        gl.clearColor(0.04, 0.04, 0.06, 1);
        gl.clear(gl.COLOR_BUFFER_BIT);

        for (let i = 0; i < DRAW_CALLS_PER_FRAME; i++) {
          const ox = pseudoRandom() * 2 - 1;
          const oy = pseudoRandom() * 2 - 1;
          gl.uniform2f(uOffset, ox, oy);
          gl.uniform1f(uScale, 0.02 + pseudoRandom() * 0.05);
          gl.uniform4f(uColor, pseudoRandom(), pseudoRandom(), pseudoRandom(), 0.8);
          gl.drawArrays(gl.TRIANGLES, 0, 6);
          drawCalls++;
        }
        gl.finish();
        frames++;

        const elapsed = performance.now() - startTime;
        if (onProgress) onProgress(Math.min(elapsed / durationMs, 1));

        if (elapsed < durationMs) {
          requestAnimationFrame(renderFrame);
        } else {
          const seconds = elapsed / 1000;
          const fps = frames / seconds;
          const dcPerSec = drawCalls / seconds;
          const trisPerSec = (drawCalls * 2) / seconds; // 2 triangles per quad

          resolve({
            fps: Math.round(fps),
            drawCallsPerSec: Math.round(dcPerSec),
            trianglesPerSec: trisPerSec,
            mTriPerSec: trisPerSec / 1e6,
            frames,
            drawCalls
          });
        }
      }

      requestAnimationFrame(renderFrame);
    });
  }

  // =============== Tier calculation ===============
  function getTierMatMul(tflops) {
    if (tflops > 5.0) return 'S';
    if (tflops > 1.0) return 'A';
    if (tflops > 0.1) return 'B';
    if (tflops > 0.01) return 'C';
    return 'D';
  }

  function getTierGpuMem(gbps) {
    if (gbps > 200) return 'S';
    if (gbps > 50)  return 'A';
    if (gbps > 10)  return 'B';
    if (gbps > 2)   return 'C';
    return 'D';
  }

  function getTierRender(fps) {
    if (fps > 120)  return 'S';
    if (fps > 60)   return 'A';
    if (fps > 30)   return 'B';
    if (fps > 15)   return 'C';
    return 'D';
  }

  return {
    matMulBenchmark,
    gpuMemoryBandwidthBenchmark,
    webglRenderBenchmark,
    getTierMatMul, getTierGpuMem, getTierRender
  };
})();

if (typeof module !== 'undefined' && module.exports) module.exports = GpuBenchmarks;
