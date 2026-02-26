/**
 * Tests for Results — score calculation, tier assignment, REF constants
 */
const Results = require('../js/results');

describe('Results', () => {
  // ------------------------------------------------------------------
  // Module structure
  // ------------------------------------------------------------------
  test('exports all expected members', () => {
    expect(typeof Results.calculateScore).toBe('function');
    expect(typeof Results.getTotalTier).toBe('function');
    expect(typeof Results.render).toBe('function');
    expect(Results.REF).toBeDefined();
    expect(typeof Results.REF).toBe('object');
  });

  // ------------------------------------------------------------------
  // REF constants
  // ------------------------------------------------------------------
  describe('REF constants', () => {
    test('all reference values are positive numbers', () => {
      for (const [key, val] of Object.entries(Results.REF)) {
        expect(typeof val).toBe('number');
        expect(val).toBeGreaterThan(0);
      }
    });

    test('contains all 12 benchmark keys', () => {
      const expectedKeys = ['cpuInt', 'cpuFloat', 'cpuMulti', 'memBW', 'json', 'gpuCompute', 'gpuMem', 'render', 'mlNNInf', 'mlConv2D', 'mlTensorOps', 'mlKMeans'];
      expectedKeys.forEach(k => {
        expect(Results.REF).toHaveProperty(k);
      });
    });
  });

  // ------------------------------------------------------------------
  // calculateScore — empty input
  // ------------------------------------------------------------------
  test('calculateScore returns 0 for empty bench results', () => {
    const { scores, totalScore } = Results.calculateScore({});
    expect(totalScore).toBe(0);
    Object.values(scores).forEach(s => expect(s).toBe(0));
  });

  // ------------------------------------------------------------------
  // calculateScore — all at reference (should = 50 normalized → 5000 total)
  // ------------------------------------------------------------------
  test('calculateScore: all benchmarks at reference value => score ~5000', () => {
    const bench = {
      cpuInt: { gops: Results.REF.cpuInt },
      cpuFloat: { mflops: Results.REF.cpuFloat },
      cpuMulti: { gops: Results.REF.cpuMulti },
      memBW: { readBW: Results.REF.memBW * 1000 },
      json: { opsPerSec: Results.REF.json },
      gpuCompute: { tflops: Results.REF.gpuCompute },
      gpuMem: { gbps: Results.REF.gpuMem },
      render: { fps: Results.REF.render },
      mlNNInf: { infPerSec: Results.REF.mlNNInf },
      mlConv2D: { convPerSec: Results.REF.mlConv2D },
      mlTensorOps: { gflops: Results.REF.mlTensorOps },
      mlKMeans: { itersPerSec: Results.REF.mlKMeans }
    };
    const { totalScore } = Results.calculateScore(bench);
    expect(totalScore).toBe(5000);
  });

  // ------------------------------------------------------------------
  // calculateScore — double reference => ~10000 (capped at 100 per)
  // ------------------------------------------------------------------
  test('calculateScore: double reference values => score 10000', () => {
    const bench = {
      cpuInt: { gops: Results.REF.cpuInt * 2 },
      cpuFloat: { mflops: Results.REF.cpuFloat * 2 },
      cpuMulti: { gops: Results.REF.cpuMulti * 2 },
      memBW: { readBW: Results.REF.memBW * 2 * 1000 },
      json: { opsPerSec: Results.REF.json * 2 },
      gpuCompute: { tflops: Results.REF.gpuCompute * 2 },
      gpuMem: { gbps: Results.REF.gpuMem * 2 },
      render: { fps: Results.REF.render * 2 },
      mlNNInf: { infPerSec: Results.REF.mlNNInf * 2 },
      mlConv2D: { convPerSec: Results.REF.mlConv2D * 2 },
      mlTensorOps: { gflops: Results.REF.mlTensorOps * 2 },
      mlKMeans: { itersPerSec: Results.REF.mlKMeans * 2 }
    };
    const { totalScore } = Results.calculateScore(bench);
    expect(totalScore).toBe(10000);
  });

  // ------------------------------------------------------------------
  // calculateScore — extreme values are capped
  // ------------------------------------------------------------------
  test('calculateScore: extreme values capped at 10000', () => {
    const bench = {
      cpuInt: { gops: 999 },
      cpuFloat: { mflops: 999 },
      cpuMulti: { gops: 999 },
      memBW: { readBW: 999999 },
      json: { opsPerSec: 999999 },
      gpuCompute: { tflops: 999 },
      gpuMem: { gbps: 999 },
      render: { fps: 999 },
      mlNNInf: { infPerSec: 999999 },
      mlConv2D: { convPerSec: 999999 },
      mlTensorOps: { gflops: 999999 },
      mlKMeans: { itersPerSec: 999999 }
    };
    const { totalScore } = Results.calculateScore(bench);
    expect(totalScore).toBe(10000);
  });

  // ------------------------------------------------------------------
  // calculateScore — partial results
  // ------------------------------------------------------------------
  test('calculateScore: partial results (CPU only)', () => {
    const bench = {
      cpuInt: { gops: Results.REF.cpuInt },
      cpuFloat: { mflops: Results.REF.cpuFloat }
    };
    const { totalScore, scores } = Results.calculateScore(bench);
    expect(totalScore).toBeGreaterThan(0);
    expect(totalScore).toBeLessThan(5000);
    expect(scores.cpuInt).toBe(50);
    expect(scores.cpuFloat).toBe(50);
    expect(scores.gpuCompute).toBe(0);
  });

  // ------------------------------------------------------------------
  // calculateScore — weights sum to 1.0
  // ------------------------------------------------------------------
  test('weights sum to 1.0', () => {
    // We can infer weights from score calculation:
    // A score of 5000 when all components are at reference (50) means weights sum to 1.0
    const bench = {
      cpuInt: { gops: Results.REF.cpuInt },
      cpuFloat: { mflops: Results.REF.cpuFloat },
      cpuMulti: { gops: Results.REF.cpuMulti },
      memBW: { readBW: Results.REF.memBW * 1000 },
      json: { opsPerSec: Results.REF.json },
      gpuCompute: { tflops: Results.REF.gpuCompute },
      gpuMem: { gbps: Results.REF.gpuMem },
      render: { fps: Results.REF.render },
      mlNNInf: { infPerSec: Results.REF.mlNNInf },
      mlConv2D: { convPerSec: Results.REF.mlConv2D },
      mlTensorOps: { gflops: Results.REF.mlTensorOps },
      mlKMeans: { itersPerSec: Results.REF.mlKMeans }
    };
    const { totalScore } = Results.calculateScore(bench);
    // 50 (normalized) * 1.0 (sum weights) * 100 = 5000
    expect(totalScore).toBe(5000);
  });

  // ------------------------------------------------------------------
  // getTotalTier
  // ------------------------------------------------------------------
  describe('getTotalTier', () => {
    test.each([
      [10000, 'GAMING PC'],
      [7000, 'GAMING PC'],
      [6999, 'WORKSTATION'],
      [5000, 'WORKSTATION'],
      [4999, 'OFFICE PC'],
      [3000, 'OFFICE PC'],
      [2999, 'BUDGET'],
      [1500, 'BUDGET'],
      [1499, 'MOBILE / LOW-END'],
      [0, 'MOBILE / LOW-END']
    ])('getTotalTier(%i) => %s', (score, expectedLabel) => {
      const tier = Results.getTotalTier(score);
      expect(tier.label).toBe(expectedLabel);
    });

    test('tier objects have label, cls, and icon', () => {
      const tier = Results.getTotalTier(8000);
      expect(tier).toHaveProperty('label');
      expect(tier).toHaveProperty('cls');
      expect(tier).toHaveProperty('icon');
      expect(typeof tier.label).toBe('string');
      expect(typeof tier.cls).toBe('string');
      expect(typeof tier.icon).toBe('string');
    });

    test('each tier has a distinct CSS class', () => {
      const classes = new Set();
      [0, 1500, 3000, 5000, 7000].forEach(score => {
        classes.add(Results.getTotalTier(score).cls);
      });
      expect(classes.size).toBe(5);
    });
  });

  // ------------------------------------------------------------------
  // calculateScore — individual normalized scores are in [0, 100]
  // ------------------------------------------------------------------
  test('individual scores are bounded [0, 100]', () => {
    const bench = {
      cpuInt: { gops: 999 },
      cpuFloat: { mflops: 0.001 },
      cpuMulti: { gops: Results.REF.cpuMulti },
      memBW: { readBW: 0 },
      json: { opsPerSec: 999999 },
      gpuCompute: { tflops: 0 },
      gpuMem: { gbps: 999 },
      render: { fps: 0 },
      mlNNInf: { infPerSec: 999999 },
      mlConv2D: { convPerSec: 0 },
      mlTensorOps: { gflops: 999 },
      mlKMeans: { itersPerSec: 0 }
    };
    const { scores } = Results.calculateScore(bench);
    for (const [, val] of Object.entries(scores)) {
      expect(val).toBeGreaterThanOrEqual(0);
      expect(val).toBeLessThanOrEqual(100);
    }
  });

  // ------------------------------------------------------------------
  // Monotonicity: higher input always means higher or equal score
  // ------------------------------------------------------------------
  test('calculateScore is monotonic with input increases', () => {
    const makeBench = (factor) => ({
      cpuInt: { gops: Results.REF.cpuInt * factor },
      cpuFloat: { mflops: Results.REF.cpuFloat * factor },
      cpuMulti: { gops: Results.REF.cpuMulti * factor },
      memBW: { readBW: Results.REF.memBW * factor * 1000 },
      json: { opsPerSec: Results.REF.json * factor },
      gpuCompute: { tflops: Results.REF.gpuCompute * factor },
      gpuMem: { gbps: Results.REF.gpuMem * factor },
      render: { fps: Results.REF.render * factor },
      mlNNInf: { infPerSec: Results.REF.mlNNInf * factor },
      mlConv2D: { convPerSec: Results.REF.mlConv2D * factor },
      mlTensorOps: { gflops: Results.REF.mlTensorOps * factor },
      mlKMeans: { itersPerSec: Results.REF.mlKMeans * factor }
    });

    let prevScore = 0;
    for (let factor = 0; factor <= 2.5; factor += 0.1) {
      const { totalScore } = Results.calculateScore(makeBench(factor));
      expect(totalScore).toBeGreaterThanOrEqual(prevScore);
      prevScore = totalScore;
    }
  });
});
