/**
 * Tests for CpuBenchmarks tier functions and benchmark contracts
 */
const CpuBenchmarks = require('../js/cpuBenchmarks');

describe('CpuBenchmarks', () => {
  // ------------------------------------------------------------------
  // Module structure
  // ------------------------------------------------------------------
  test('exports all expected functions', () => {
    expect(typeof CpuBenchmarks.integerBenchmark).toBe('function');
    expect(typeof CpuBenchmarks.floatBenchmark).toBe('function');
    expect(typeof CpuBenchmarks.multiThreadBenchmark).toBe('function');
    expect(typeof CpuBenchmarks.memoryBandwidthBenchmark).toBe('function');
    expect(typeof CpuBenchmarks.jsonBenchmark).toBe('function');
    expect(typeof CpuBenchmarks.getTierInt).toBe('function');
    expect(typeof CpuBenchmarks.getTierFloat).toBe('function');
    expect(typeof CpuBenchmarks.getTierMulti).toBe('function');
    expect(typeof CpuBenchmarks.getTierMemory).toBe('function');
    expect(typeof CpuBenchmarks.getTierJSON).toBe('function');
  });

  // ------------------------------------------------------------------
  // Integer tier
  // ------------------------------------------------------------------
  describe('getTierInt', () => {
    test.each([
      [3.0, 'S'],
      [2.1, 'S'],
      [1.5, 'A'],
      [1.01, 'A'],
      [0.7, 'B'],
      [0.51, 'B'],
      [0.3, 'C'],
      [0.21, 'C'],
      [0.1, 'D'],
      [0.0, 'D']
    ])('getTierInt(%f) => %s', (gops, expected) => {
      expect(CpuBenchmarks.getTierInt(gops)).toBe(expected);
    });

    test('boundary: exactly 2.0 is A, not S', () => {
      expect(CpuBenchmarks.getTierInt(2.0)).toBe('A');
    });
  });

  // ------------------------------------------------------------------
  // Float tier
  // ------------------------------------------------------------------
  describe('getTierFloat', () => {
    test.each([
      [60, 'S'],
      [51, 'S'],
      [30, 'A'],
      [25.1, 'A'],
      [15, 'B'],
      [10.1, 'B'],
      [5, 'C'],
      [3.1, 'C'],
      [2, 'D'],
      [0, 'D']
    ])('getTierFloat(%f) => %s', (mflops, expected) => {
      expect(CpuBenchmarks.getTierFloat(mflops)).toBe(expected);
    });
  });

  // ------------------------------------------------------------------
  // Multi-thread tier
  // ------------------------------------------------------------------
  describe('getTierMulti', () => {
    test.each([
      [15, 'S'],
      [10.1, 'S'],
      [7, 'A'],
      [5.1, 'A'],
      [3, 'B'],
      [2.1, 'B'],
      [1, 'C'],
      [0.51, 'C'],
      [0.3, 'D'],
      [0, 'D']
    ])('getTierMulti(%f) => %s', (gops, expected) => {
      expect(CpuBenchmarks.getTierMulti(gops)).toBe(expected);
    });
  });

  // ------------------------------------------------------------------
  // Memory tier
  // ------------------------------------------------------------------
  describe('getTierMemory', () => {
    test.each([
      [15, 'S'],
      [10.1, 'S'],
      [7, 'A'],
      [5.1, 'A'],
      [3, 'B'],
      [2.1, 'B'],
      [1, 'C'],
      [0.51, 'C'],
      [0.3, 'D'],
      [0, 'D']
    ])('getTierMemory(%f) => %s', (gbps, expected) => {
      expect(CpuBenchmarks.getTierMemory(gbps)).toBe(expected);
    });
  });

  // ------------------------------------------------------------------
  // JSON tier
  // ------------------------------------------------------------------
  describe('getTierJSON', () => {
    test.each([
      [6000, 'S'],
      [5001, 'S'],
      [3000, 'A'],
      [2001, 'A'],
      [1000, 'B'],
      [801, 'B'],
      [500, 'C'],
      [201, 'C'],
      [100, 'D'],
      [0, 'D']
    ])('getTierJSON(%f) => %s', (ops, expected) => {
      expect(CpuBenchmarks.getTierJSON(ops)).toBe(expected);
    });
  });

  // ------------------------------------------------------------------
  // Tier functions never return invalid values
  // ------------------------------------------------------------------
  describe('tier functions return valid tiers', () => {
    const validTiers = ['S', 'A', 'B', 'C', 'D'];
    const tierFns = [
      { fn: CpuBenchmarks.getTierInt, name: 'getTierInt' },
      { fn: CpuBenchmarks.getTierFloat, name: 'getTierFloat' },
      { fn: CpuBenchmarks.getTierMulti, name: 'getTierMulti' },
      { fn: CpuBenchmarks.getTierMemory, name: 'getTierMemory' },
      { fn: CpuBenchmarks.getTierJSON, name: 'getTierJSON' }
    ];

    tierFns.forEach(({ fn, name }) => {
      test(`${name} always returns S/A/B/C/D`, () => {
        for (let v = 0; v <= 100; v += 0.5) {
          expect(validTiers).toContain(fn(v));
        }
      });

      test(`${name} returns D for negative values`, () => {
        expect(fn(-1)).toBe('D');
        expect(fn(-100)).toBe('D');
      });
    });
  });

  // ------------------------------------------------------------------
  // Integer benchmark returns correct shape (short duration)
  // ------------------------------------------------------------------
  test('integerBenchmark resolves with expected shape', async () => {
    const result = await CpuBenchmarks.integerBenchmark(200);
    expect(result).toHaveProperty('opsPerSec');
    expect(result).toHaveProperty('gops');
    expect(result).toHaveProperty('checksum');
    expect(typeof result.opsPerSec).toBe('number');
    expect(typeof result.gops).toBe('number');
    expect(result.opsPerSec).toBeGreaterThan(0);
    expect(result.gops).toBeGreaterThan(0);
  }, 10000);

  // ------------------------------------------------------------------
  // Float benchmark returns correct shape
  // ------------------------------------------------------------------
  test('floatBenchmark resolves with expected shape', async () => {
    const result = await CpuBenchmarks.floatBenchmark(200);
    expect(result).toHaveProperty('opsPerSec');
    expect(result).toHaveProperty('mflops');
    expect(result).toHaveProperty('checksum');
    expect(typeof result.mflops).toBe('number');
    expect(result.opsPerSec).toBeGreaterThan(0);
  }, 10000);

  // ------------------------------------------------------------------
  // JSON benchmark returns correct shape
  // ------------------------------------------------------------------
  test('jsonBenchmark resolves with expected shape', async () => {
    const result = await CpuBenchmarks.jsonBenchmark(200);
    expect(result).toHaveProperty('opsPerSec');
    expect(result).toHaveProperty('totalOps');
    expect(typeof result.opsPerSec).toBe('number');
    expect(result.totalOps).toBeGreaterThan(0);
  }, 10000);

  // ------------------------------------------------------------------
  // Memory bandwidth benchmark returns correct shape
  // ------------------------------------------------------------------
  test('memoryBandwidthBenchmark resolves with expected shape', async () => {
    const result = await CpuBenchmarks.memoryBandwidthBenchmark();
    expect(result).toHaveProperty('writeBW');
    expect(result).toHaveProperty('readBW');
    expect(result).toHaveProperty('randomBW');
    expect(result.readBW).toBeGreaterThan(0);
    expect(result.writeBW).toBeGreaterThan(0);
  }, 15000);

  // ------------------------------------------------------------------
  // Progress callback is invoked
  // ------------------------------------------------------------------
  test('integerBenchmark calls onProgress', async () => {
    const progressValues = [];
    await CpuBenchmarks.integerBenchmark(200, (p) => progressValues.push(p));
    expect(progressValues.length).toBeGreaterThan(0);
    expect(progressValues[progressValues.length - 1]).toBeCloseTo(1.0, 0);
  }, 10000);
});
