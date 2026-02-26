/**
 * Tests for SystemInfo module
 */
const SystemInfo = require('../js/systemInfo');

describe('SystemInfo', () => {
  test('exports detect and render functions', () => {
    expect(typeof SystemInfo.detect).toBe('function');
    expect(typeof SystemInfo.render).toBe('function');
  });

  test('detect returns info with expected shape', async () => {
    const info = await SystemInfo.detect();
    expect(info).toHaveProperty('cpu');
    expect(info).toHaveProperty('gpu');
    expect(info).toHaveProperty('screen');
    expect(info).toHaveProperty('network');
    expect(info).toHaveProperty('features');
    expect(info).toHaveProperty('refreshRate');
  });

  test('cpu info has cores and architecture', async () => {
    const info = await SystemInfo.detect();
    expect(info.cpu).toHaveProperty('cores');
    expect(typeof info.cpu.cores).toBe('number');
    expect(info.cpu.cores).toBeGreaterThan(0);
    expect(info.cpu).toHaveProperty('architecture');
    expect(typeof info.cpu.architecture).toBe('string');
  });

  test('cpu info has browser and os', async () => {
    const info = await SystemInfo.detect();
    expect(info.cpu).toHaveProperty('browser');
    expect(info.cpu).toHaveProperty('os');
    expect(typeof info.cpu.browser).toBe('string');
    expect(typeof info.cpu.os).toBe('string');
  });

  test('screen info has resolution and colorDepth', async () => {
    const info = await SystemInfo.detect();
    expect(info.screen).toHaveProperty('resolution');
    expect(info.screen).toHaveProperty('viewport');
    expect(info.screen).toHaveProperty('colorDepth');
    expect(info.screen).toHaveProperty('pixelRatio');
    expect(typeof info.screen.colorDepth).toBe('number');
    expect(typeof info.screen.pixelRatio).toBe('number');
  });

  test('features is an array of objects with name and available', async () => {
    const info = await SystemInfo.detect();
    expect(Array.isArray(info.features)).toBe(true);
    expect(info.features.length).toBeGreaterThan(0);
    info.features.forEach(f => {
      expect(f).toHaveProperty('name');
      expect(f).toHaveProperty('available');
      expect(typeof f.name).toBe('string');
      expect(typeof f.available).toBe('boolean');
    });
  });

  test('gpu info has available property', async () => {
    const info = await SystemInfo.detect();
    expect(info.gpu).toHaveProperty('available');
    expect(typeof info.gpu.available).toBe('boolean');
    // In jsdom, WebGPU is not available
    expect(info.gpu.available).toBe(false);
  });

  test('render throws TypeError when DOM element missing', async () => {
    const info = await SystemInfo.detect();
    // render() requires #sysinfo-grid in DOM
    expect(() => SystemInfo.render(info)).toThrow(TypeError);
  });

  test('render populates sysinfo-grid when present', async () => {
    document.body.innerHTML = '<div class="sysinfo-grid" id="sysinfo-grid"></div>';
    const info = await SystemInfo.detect();
    SystemInfo.render(info);
    const grid = document.getElementById('sysinfo-grid');
    expect(grid.innerHTML).not.toBe('');
    // Should have at least 3 cards
    expect(grid.querySelectorAll('.sysinfo-card').length).toBeGreaterThanOrEqual(3);
  });
});
