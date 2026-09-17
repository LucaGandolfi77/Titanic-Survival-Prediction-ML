import { describe, it, expect, vi } from 'vitest';
import { createAppState, createARState } from '../../src/core/state.js';
import { createLifecycleManager } from '../../src/core/lifecycle.js';
import { ALL_FILTERS } from '../../src/core/constants.js';
import { createCameraManager } from '../../src/camera/camera-manager.js';

describe('createAppState', () => {
  it('should return state with default values', () => {
    const { get } = createAppState();
    expect(get('previewing')).toBe(false);
    expect(get('webglSupported')).toBe(true);
    expect(get('facingMode')).toBe('user');
    expect(get('zoom')).toBe(1);
    expect(get('detectionBusy')).toBe(false);
  });

  it('should subscribe and notify listeners', () => {
    const { set, subscribe } = createAppState();
    const callback = vi.fn();
    subscribe('zoom', callback);
    set('zoom', 2);
    expect(callback).toHaveBeenCalledWith({ old: 1, value: 2 });
  });

  it('should unsubscribe correctly', () => {
    const { set, subscribe } = createAppState();
    const callback = vi.fn();
    const unsubscribe = subscribe('zoom', callback);
    unsubscribe();
    set('zoom', 2);
    expect(callback).not.toHaveBeenCalled();
  });
});

describe('createARState', () => {
  it('should create particle systems', () => {
    const state = createARState();
    expect(state.particles.fire).toBeDefined();
    expect(state.particles.sparkles).toBeDefined();
    expect(state.particles.jet).toBeDefined();
  });

  it('particle system should update and remove dead particles', () => {
    const ps = new (createARState().particles.fire.constructor)();
    ps.spawn(10, 10, { life: 0.1 });
    expect(ps.particles.length).toBe(1);
    ps.update(0.2);
    expect(ps.particles.length).toBe(0);
  });
});

describe('createLifecycleManager', () => {
  it('should register and emit hooks', () => {
    const manager = createLifecycleManager({});
    const handler = vi.fn();
    manager.on('shutdown', handler);
    manager.emit('shutdown');
    expect(handler).toHaveBeenCalled();
  });

  it('should not throw on error hooks', () => {
    const manager = createLifecycleManager({});
    manager.on('error', () => {
      throw new Error('test');
    });
    expect(() => manager.emit('error', new Error('test'))).not.toThrow();
  });
});

describe('ALL_FILTERS', () => {
  it('should have at least 28 filters', () => {
    expect(ALL_FILTERS.length).toBeGreaterThanOrEqual(28);
  });

  it('should contain shader and ar categories', () => {
    const categories = ALL_FILTERS.map((f) => f.category);
    expect(categories).toContain('shader');
    expect(categories).toContain('ar');
  });

  it('should have unique ids', () => {
    const ids = ALL_FILTERS.map((f) => f.id);
    expect(new Set(ids).size).toBe(ids.length);
  });
});

describe('createCameraManager', () => {
  it('should expose required methods', () => {
    const state = createAppState();
    const lifecycle = createLifecycleManager({});
    const manager = createCameraManager(state, lifecycle);
    expect(manager.start).toBeInstanceOf(Function);
    expect(manager.stop).toBeInstanceOf(Function);
    expect(manager.flip).toBeInstanceOf(Function);
    expect(manager.toggleTorch).toBeInstanceOf(Function);
    expect(manager.setZoom).toBeInstanceOf(Function);
  });

  it('setZoom should clamp values', async () => {
    const state = createAppState();
    const lifecycle = createLifecycleManager({});
    const manager = createCameraManager(state, lifecycle);
    expect(await manager.setZoom(0.5)).toBe(1);
    expect(await manager.setZoom(5)).toBe(3);
    expect(await manager.setZoom(2)).toBe(2);
  });
});
