/**
 * Tests for StressTest module
 */
const StressTest = require('../js/stressTest');

describe('StressTest', () => {
  test('exports expected interface', () => {
    expect(typeof StressTest.render).toBe('function');
    expect(typeof StressTest.start).toBe('function');
    expect(typeof StressTest.stop).toBe('function');
    expect(typeof StressTest.isRunning).toBe('function');
  });

  test('isRunning returns false initially', () => {
    expect(StressTest.isRunning()).toBe(false);
  });

  test('render throws when stress-content element missing', () => {
    // render() requires #stress-content in DOM
    expect(() => StressTest.render()).toThrow(TypeError);
  });

  test('render populates stress-content when present', () => {
    // StressTest.render() needs Chart to be available
    global.Chart = class {
      constructor() { this.data = { labels: [], datasets: [{ data: [] }] }; }
      update() {}
      destroy() {}
    };

    document.body.innerHTML = '<div id="stress-content"></div>';
    StressTest.render();
    const content = document.getElementById('stress-content');
    expect(content.innerHTML).not.toBe('');
    expect(content.querySelector('.stress-controls')).toBeTruthy();

    delete global.Chart;
  });
});
