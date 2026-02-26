/**
 * Tests for Capabilities module — browser API detection
 */
const Capabilities = require('../js/capabilities');

describe('Capabilities', () => {
  test('exports render function', () => {
    expect(typeof Capabilities.render).toBe('function');
  });

  test('render does not throw when container is missing', () => {
    // jsdom won't have #capabilities-content by default
    expect(() => Capabilities.render()).not.toThrow();
  });

  test('render populates container when DOM element exists', () => {
    document.body.innerHTML = '<div id="capabilities-content"></div>';
    Capabilities.render();
    const container = document.getElementById('capabilities-content');
    expect(container.innerHTML).not.toBe('');
    expect(container.querySelector('.cap-table')).toBeTruthy();
    expect(container.querySelector('.cap-summary')).toBeTruthy();
  });

  test('render produces rows for all tested APIs', () => {
    document.body.innerHTML = '<div id="capabilities-content"></div>';
    Capabilities.render();
    const rows = document.querySelectorAll('.cap-table tbody tr');
    // At least 10 APIs tested
    expect(rows.length).toBeGreaterThanOrEqual(10);
  });

  test('each row has status badge (cap-yes or cap-no)', () => {
    document.body.innerHTML = '<div id="capabilities-content"></div>';
    Capabilities.render();
    const rows = document.querySelectorAll('.cap-table tbody tr');
    rows.forEach(row => {
      const yes = row.querySelector('.cap-yes');
      const no = row.querySelector('.cap-no');
      expect(yes || no).toBeTruthy();
    });
  });

  test('summary shows correct count format', () => {
    document.body.innerHTML = '<div id="capabilities-content"></div>';
    Capabilities.render();
    const score = document.querySelector('.cap-summary-score');
    expect(score).toBeTruthy();
    // Format: "N / M"
    expect(score.textContent).toMatch(/^\d+ \/ \d+$/);
  });

  test('known jsdom capabilities are detected correctly', () => {
    document.body.innerHTML = '<div id="capabilities-content"></div>';
    Capabilities.render();
    const rows = document.querySelectorAll('.cap-table tbody tr');
    const map = {};
    rows.forEach(row => {
      const name = row.querySelector('td:first-child').textContent.trim();
      const supported = !!row.querySelector('.cap-yes');
      map[name] = supported;
    });

    // jsdom should NOT have WebGPU
    expect(map['WebGPU']).toBe(false);
    // jsdom does have IndexedDB (via fake-indexeddb or built-in)
    // Performance Observer is in Node 16+
    // Just verify the keys exist
    expect('WebGPU' in map).toBe(true);
    expect('Web Workers' in map).toBe(true);
    expect('IndexedDB' in map).toBe(true);
  });
});
