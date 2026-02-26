/**
 * Tests for MlBenchmarks tier functions and benchmark contracts
 */
const MlBenchmarks = require('../js/mlBenchmarks');

describe('MlBenchmarks', () => {
  // ------------------------------------------------------------------
  // Module structure
  // ------------------------------------------------------------------
  test('exports all expected functions', () => {
    expect(typeof MlBenchmarks.nnInferenceBenchmark).toBe('function');
    expect(typeof MlBenchmarks.conv2dBenchmark).toBe('function');
    expect(typeof MlBenchmarks.tensorOpsBenchmark).toBe('function');
    expect(typeof MlBenchmarks.kMeansBenchmark).toBe('function');
    expect(typeof MlBenchmarks.getTierNNInference).toBe('function');
    expect(typeof MlBenchmarks.getTierConv2D).toBe('function');
    expect(typeof MlBenchmarks.getTierTensorOps).toBe('function');
    expect(typeof MlBenchmarks.getTierKMeans).toBe('function');
  });

  // ------------------------------------------------------------------
  // NN Inference tier
  // ------------------------------------------------------------------
  describe('getTierNNInference', () => {
    test.each([
      [10000, 'S'],
      [8001,  'S'],
      [6000,  'A'],
      [4001,  'A'],
      [2500,  'B'],
      [1501,  'B'],
      [800,   'C'],
      [501,   'C'],
      [200,   'D'],
      [0,     'D']
    ])('getTierNNInference(%f) => %s', (infPerSec, expected) => {
      expect(MlBenchmarks.getTierNNInference(infPerSec)).toBe(expected);
    });

    test('boundary: exactly 8000 is A, not S', () => {
      expect(MlBenchmarks.getTierNNInference(8000)).toBe('A');
    });

    test('boundary: exactly 4000 is B, not A', () => {
      expect(MlBenchmarks.getTierNNInference(4000)).toBe('B');
    });

    test('boundary: exactly 1500 is C, not B', () => {
      expect(MlBenchmarks.getTierNNInference(1500)).toBe('C');
    });

    test('boundary: exactly 500 is D, not C', () => {
      expect(MlBenchmarks.getTierNNInference(500)).toBe('D');
    });
  });

  // ------------------------------------------------------------------
  // Conv2D tier
  // ------------------------------------------------------------------
  describe('getTierConv2D', () => {
    test.each([
      [200, 'S'],
      [101, 'S'],
      [60,  'A'],
      [41,  'A'],
      [25,  'B'],
      [16,  'B'],
      [8,   'C'],
      [6,   'C'],
      [3,   'D'],
      [0,   'D']
    ])('getTierConv2D(%f) => %s', (convPerSec, expected) => {
      expect(MlBenchmarks.getTierConv2D(convPerSec)).toBe(expected);
    });

    test('boundary: exactly 100 is A, not S', () => {
      expect(MlBenchmarks.getTierConv2D(100)).toBe('A');
    });

    test('boundary: exactly 40 is B, not A', () => {
      expect(MlBenchmarks.getTierConv2D(40)).toBe('B');
    });

    test('boundary: exactly 15 is C, not B', () => {
      expect(MlBenchmarks.getTierConv2D(15)).toBe('C');
    });

    test('boundary: exactly 5 is D, not C', () => {
      expect(MlBenchmarks.getTierConv2D(5)).toBe('D');
    });
  });

  // ------------------------------------------------------------------
  // Tensor Ops tier
  // ------------------------------------------------------------------
  describe('getTierTensorOps', () => {
    test.each([
      [15,  'S'],
      [10.1, 'S'],
      [6,   'A'],
      [4.1, 'A'],
      [2.5, 'B'],
      [1.6, 'B'],
      [1.0, 'C'],
      [0.6, 'C'],
      [0.3, 'D'],
      [0,   'D']
    ])('getTierTensorOps(%f) => %s', (gflops, expected) => {
      expect(MlBenchmarks.getTierTensorOps(gflops)).toBe(expected);
    });

    test('boundary: exactly 10 is A, not S', () => {
      expect(MlBenchmarks.getTierTensorOps(10)).toBe('A');
    });

    test('boundary: exactly 4 is B, not A', () => {
      expect(MlBenchmarks.getTierTensorOps(4)).toBe('B');
    });

    test('boundary: exactly 1.5 is C, not B', () => {
      expect(MlBenchmarks.getTierTensorOps(1.5)).toBe('C');
    });

    test('boundary: exactly 0.5 is D, not C', () => {
      expect(MlBenchmarks.getTierTensorOps(0.5)).toBe('D');
    });
  });

  // ------------------------------------------------------------------
  // K-Means tier
  // ------------------------------------------------------------------
  describe('getTierKMeans', () => {
    test.each([
      [100, 'S'],
      [51,  'S'],
      [35,  'A'],
      [21,  'A'],
      [12,  'B'],
      [9,   'B'],
      [5,   'C'],
      [4,   'C'],
      [2,   'D'],
      [0,   'D']
    ])('getTierKMeans(%f) => %s', (itersPerSec, expected) => {
      expect(MlBenchmarks.getTierKMeans(itersPerSec)).toBe(expected);
    });

    test('boundary: exactly 50 is A, not S', () => {
      expect(MlBenchmarks.getTierKMeans(50)).toBe('A');
    });

    test('boundary: exactly 20 is B, not A', () => {
      expect(MlBenchmarks.getTierKMeans(20)).toBe('B');
    });

    test('boundary: exactly 8 is C, not B', () => {
      expect(MlBenchmarks.getTierKMeans(8)).toBe('C');
    });

    test('boundary: exactly 3 is D, not C', () => {
      expect(MlBenchmarks.getTierKMeans(3)).toBe('D');
    });
  });

  // ------------------------------------------------------------------
  // Benchmark return shape contracts (short duration)
  // ------------------------------------------------------------------
  describe('nnInferenceBenchmark', () => {
    test('returns expected shape', async () => {
      const result = await MlBenchmarks.nnInferenceBenchmark(200);
      expect(result).toHaveProperty('infPerSec');
      expect(result).toHaveProperty('totalInferences');
      expect(result).toHaveProperty('mflops');
      expect(result).toHaveProperty('layers', 4);
      expect(result).toHaveProperty('architecture', '784→256→128→64→10');
      expect(typeof result.infPerSec).toBe('number');
      expect(result.infPerSec).toBeGreaterThan(0);
      expect(result.totalInferences).toBeGreaterThan(0);
      expect(result.mflops).toBeGreaterThan(0);
    }, 10000);

    test('calls onProgress callback', async () => {
      const progressValues = [];
      await MlBenchmarks.nnInferenceBenchmark(200, (pct) => progressValues.push(pct));
      expect(progressValues.length).toBeGreaterThan(0);
      expect(progressValues[progressValues.length - 1]).toBeCloseTo(1, 0);
    }, 10000);
  });

  describe('conv2dBenchmark', () => {
    test('returns expected shape', async () => {
      const result = await MlBenchmarks.conv2dBenchmark(200);
      expect(result).toHaveProperty('convPerSec');
      expect(result).toHaveProperty('totalConvolutions');
      expect(result).toHaveProperty('mflops');
      expect(result).toHaveProperty('mpixPerSec');
      expect(result).toHaveProperty('imageSize', '512×512');
      expect(result).toHaveProperty('kernelCount', 5);
      expect(typeof result.convPerSec).toBe('number');
      expect(result.convPerSec).toBeGreaterThan(0);
    }, 10000);
  });

  describe('tensorOpsBenchmark', () => {
    test('returns expected shape', async () => {
      const result = await MlBenchmarks.tensorOpsBenchmark(200);
      expect(result).toHaveProperty('opsPerSec');
      expect(result).toHaveProperty('totalOps');
      expect(result).toHaveProperty('gflops');
      expect(result).toHaveProperty('tensorSize', '1024×1024');
      expect(result).toHaveProperty('elements', 1048576);
      expect(typeof result.gflops).toBe('number');
      expect(result.gflops).toBeGreaterThan(0);
    }, 10000);
  });

  describe('kMeansBenchmark', () => {
    test('returns expected shape', async () => {
      const result = await MlBenchmarks.kMeansBenchmark(200);
      expect(result).toHaveProperty('itersPerSec');
      expect(result).toHaveProperty('totalIterations');
      expect(result).toHaveProperty('gflops');
      expect(result).toHaveProperty('dataPoints', 50000);
      expect(result).toHaveProperty('dimensions', 16);
      expect(result).toHaveProperty('clusters', 32);
      expect(typeof result.itersPerSec).toBe('number');
      expect(result.itersPerSec).toBeGreaterThan(0);
    }, 10000);
  });
});
