/**
 * Tests for GpuBenchmarks tier functions
 * (GPU hardware tests are skipped — no WebGPU/WebGL in jsdom)
 */
const GpuBenchmarks = require('../js/gpuBenchmarks');

describe('GpuBenchmarks', () => {
  // ------------------------------------------------------------------
  // Module structure
  // ------------------------------------------------------------------
  test('exports all expected functions', () => {
    expect(typeof GpuBenchmarks.matMulBenchmark).toBe('function');
    expect(typeof GpuBenchmarks.gpuMemoryBandwidthBenchmark).toBe('function');
    expect(typeof GpuBenchmarks.webglRenderBenchmark).toBe('function');
    expect(typeof GpuBenchmarks.getTierMatMul).toBe('function');
    expect(typeof GpuBenchmarks.getTierGpuMem).toBe('function');
    expect(typeof GpuBenchmarks.getTierRender).toBe('function');
  });

  // ------------------------------------------------------------------
  // MatMul tier
  // ------------------------------------------------------------------
  describe('getTierMatMul', () => {
    test.each([
      [10.0, 'S'],
      [5.1, 'S'],
      [3.0, 'A'],
      [1.01, 'A'],
      [0.5, 'B'],
      [0.11, 'B'],
      [0.05, 'C'],
      [0.011, 'C'],
      [0.005, 'D'],
      [0, 'D']
    ])('getTierMatMul(%f) => %s', (tflops, expected) => {
      expect(GpuBenchmarks.getTierMatMul(tflops)).toBe(expected);
    });
  });

  // ------------------------------------------------------------------
  // GPU Memory tier
  // ------------------------------------------------------------------
  describe('getTierGpuMem', () => {
    test.each([
      [300, 'S'],
      [201, 'S'],
      [100, 'A'],
      [51, 'A'],
      [25, 'B'],
      [10.1, 'B'],
      [5, 'C'],
      [2.1, 'C'],
      [1, 'D'],
      [0, 'D']
    ])('getTierGpuMem(%f) => %s', (gbps, expected) => {
      expect(GpuBenchmarks.getTierGpuMem(gbps)).toBe(expected);
    });
  });

  // ------------------------------------------------------------------
  // Render tier
  // ------------------------------------------------------------------
  describe('getTierRender', () => {
    test.each([
      [144, 'S'],
      [121, 'S'],
      [90, 'A'],
      [61, 'A'],
      [45, 'B'],
      [31, 'B'],
      [20, 'C'],
      [15.1, 'C'],
      [10, 'D'],
      [0, 'D']
    ])('getTierRender(%f) => %s', (fps, expected) => {
      expect(GpuBenchmarks.getTierRender(fps)).toBe(expected);
    });
  });

  // ------------------------------------------------------------------
  // Tier functions integrity
  // ------------------------------------------------------------------
  describe('tier functions always return valid tiers', () => {
    const validTiers = ['S', 'A', 'B', 'C', 'D'];

    test('getTierMatMul across range', () => {
      for (let v = 0; v <= 20; v += 0.1) {
        expect(validTiers).toContain(GpuBenchmarks.getTierMatMul(v));
      }
    });

    test('getTierGpuMem across range', () => {
      for (let v = 0; v <= 500; v += 1) {
        expect(validTiers).toContain(GpuBenchmarks.getTierGpuMem(v));
      }
    });

    test('getTierRender across range', () => {
      for (let v = 0; v <= 300; v += 1) {
        expect(validTiers).toContain(GpuBenchmarks.getTierRender(v));
      }
    });
  });

  // ------------------------------------------------------------------
  // GPU benchmarks should throw/reject without WebGPU
  // ------------------------------------------------------------------
  test('matMulBenchmark rejects without WebGPU', async () => {
    await expect(GpuBenchmarks.matMulBenchmark()).rejects.toThrow(/WebGPU/);
  });

  test('gpuMemoryBandwidthBenchmark rejects without WebGPU', async () => {
    await expect(GpuBenchmarks.gpuMemoryBandwidthBenchmark()).rejects.toThrow(/WebGPU/);
  });
});
