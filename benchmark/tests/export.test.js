/**
 * Tests for ExportUtils module
 */

// Mock Results and App globals that ExportUtils depends on
global.Results = {
  calculateScore: jest.fn(() => ({
    totalScore: 5000,
    scores: { cpuInt: 50, cpuFloat: 50, cpuMulti: 50, memBW: 50, json: 50, gpuCompute: 50, gpuMem: 50, render: 50 }
  })),
  getTotalTier: jest.fn(() => ({ label: 'WORKSTATION', cls: 'tier-workstation', icon: 'fa-server' }))
};

global.App = {
  toast: jest.fn()
};

const ExportUtils = require('../js/export');

describe('ExportUtils', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('exports all expected functions', () => {
    expect(typeof ExportUtils.copyToClipboard).toBe('function');
    expect(typeof ExportUtils.downloadJSON).toBe('function');
    expect(typeof ExportUtils.shareViaURL).toBe('function');
  });

  // ------------------------------------------------------------------
  // copyToClipboard
  // ------------------------------------------------------------------
  describe('copyToClipboard', () => {
    test('calls clipboard.writeText', async () => {
      Object.assign(navigator, {
        clipboard: { writeText: jest.fn().mockResolvedValue(undefined) }
      });

      const bench = { cpuInt: { gops: 1.5 } };
      await ExportUtils.copyToClipboard(bench, null);

      expect(navigator.clipboard.writeText).toHaveBeenCalledTimes(1);
      const text = navigator.clipboard.writeText.mock.calls[0][0];
      expect(text).toContain('PC BENCHMARK SUITE');
      expect(text).toContain('Overall Score');
    });

    test('clipboard text includes system info when provided', async () => {
      Object.assign(navigator, {
        clipboard: { writeText: jest.fn().mockResolvedValue(undefined) }
      });

      const bench = { cpuInt: { gops: 1.5 } };
      const sysInfo = { cpu: { cores: 8, architecture: 'x86_64', os: 'Linux', browser: 'Chrome', browserVersion: '120', deviceMemory: 16 } };
      await ExportUtils.copyToClipboard(bench, sysInfo);

      const text = navigator.clipboard.writeText.mock.calls[0][0];
      expect(text).toContain('CPU Cores: 8');
      expect(text).toContain('Architecture: x86_64');
    });

    test('shows success toast after copy', async () => {
      Object.assign(navigator, {
        clipboard: { writeText: jest.fn().mockResolvedValue(undefined) }
      });
      await ExportUtils.copyToClipboard({}, null);
      expect(App.toast).toHaveBeenCalledWith(expect.stringContaining('clipboard'), 'success');
    });
  });

  // ------------------------------------------------------------------
  // downloadJSON
  // ------------------------------------------------------------------
  describe('downloadJSON', () => {
    test('creates a download link and clicks it', () => {
      const createObjectURL = jest.fn(() => 'blob:test');
      const revokeObjectURL = jest.fn();
      global.URL.createObjectURL = createObjectURL;
      global.URL.revokeObjectURL = revokeObjectURL;

      const clickSpy = jest.fn();
      jest.spyOn(document, 'createElement').mockImplementation((tag) => {
        if (tag === 'a') {
          return { href: '', download: '', click: clickSpy, set href(v) { this._href = v; } };
        }
        return document.createElement(tag);
      });

      ExportUtils.downloadJSON({ cpuInt: { gops: 1.0 } }, { cpu: { cores: 4 } });

      expect(createObjectURL).toHaveBeenCalledTimes(1);
      expect(clickSpy).toHaveBeenCalledTimes(1);
      expect(revokeObjectURL).toHaveBeenCalledTimes(1);
      expect(App.toast).toHaveBeenCalledWith(expect.stringContaining('JSON'), 'success');

      document.createElement.mockRestore();
    });
  });

  // ------------------------------------------------------------------
  // shareViaURL
  // ------------------------------------------------------------------
  describe('shareViaURL', () => {
    test('encodes results as base64 in URL hash', async () => {
      let capturedUrl = '';
      Object.assign(navigator, {
        clipboard: { writeText: jest.fn((url) => { capturedUrl = url; return Promise.resolve(); }) }
      });

      ExportUtils.shareViaURL({ cpuInt: { gops: 1.0 } });

      // Wait for the promise
      await new Promise(r => setTimeout(r, 10));

      expect(navigator.clipboard.writeText).toHaveBeenCalledTimes(1);
      const url = navigator.clipboard.writeText.mock.calls[0][0];
      expect(url).toContain('#results=');

      // Decode and validate
      const encoded = url.split('#results=')[1];
      const decoded = JSON.parse(atob(encoded));
      expect(decoded).toHaveProperty('s');
      expect(decoded.s).toBe(5000); // From our mock
    });
  });
});
