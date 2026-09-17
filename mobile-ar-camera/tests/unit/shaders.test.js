import { readFileSync } from 'fs';
import { describe, it, expect } from 'vitest';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const shadersSrc = readFileSync(join(__dirname, '../../shaders.js'), 'utf-8');

function extractFragments(src) {
  const result = {};
  const filterIds = ['normal', 'vintage', 'neon', 'glitch', 'pixelate', 'sketch', 'blur', 'warm-glow', 'cool-bleach', 'film-grain'];
  const fragStart = src.indexOf('const FRAGMENTS');
  let searchFrom = fragStart === -1 ? 0 : fragStart;
  for (const id of filterIds) {
    const funcStart = src.indexOf('commonFragment(', searchFrom);
    if (funcStart === -1) { result[id] = null; continue; }
    const open = src.indexOf('`', funcStart);
    const close = src.indexOf('`', open + 1);
    if (close === -1) { result[id] = null; continue; }
    result[id] = src.slice(open + 1, close);
    searchFrom = close + 1;
  }
  return result;
}

const FRAGMENTS = extractFragments(shadersSrc);

function extractVertex(src) {
  const start = src.indexOf("export const vertexShader = /* glsl */`");
  if (start === -1) return '';
  const bodyStart = start + "export const vertexShader = /* glsl */`".length;
  const end = src.indexOf('`;', bodyStart);
  return end === -1 ? '' : src.slice(bodyStart, end);
}

const vertexShader = extractVertex(shadersSrc);

describe('Shaders (GLSL correctness)', () => {
  it('should have all required fragment shaders', () => {
    const required = ['normal', 'vintage', 'neon', 'glitch', 'pixelate', 'sketch', 'blur', 'warm-glow', 'cool-bleach', 'film-grain'];
    for (const id of required) {
      expect(FRAGMENTS[id]).not.toBeNull();
    }
  });

  it('non-normal shaders should have body content', () => {
    expect(FRAGMENTS.vintage.length).toBeGreaterThan(50);
    expect(FRAGMENTS.neon.length).toBeGreaterThan(50);
    expect(FRAGMENTS.glitch.length).toBeGreaterThan(50);
  });

  it('glitch shader should use sampleX for mirror consistency', () => {
    expect(FRAGMENTS.glitch).toContain('sampleX');
    expect(FRAGMENTS.glitch).not.toMatch(/texture2D\(uTexture,\s*vec2\(1\.0\s*-\s*uv\.x/);
  });

  it('warm-glow should use luma not luminance', () => {
    expect(FRAGMENTS['warm-glow']).not.toContain('luminance(');
    expect(FRAGMENTS['warm-glow']).toContain('luma(');
  });
});

describe('Shader metadata', () => {
  it('should define SHADER_FILTERS', () => {
    expect(shadersSrc).toContain('SHADER_FILTERS');
    expect(shadersSrc).toContain('normal');
    expect(shadersSrc).toContain('vintage');
    expect(shadersSrc).toContain('glitch');
  });
});

describe('Shader utilities', () => {
  it('should define createShaderMaterial', () => {
    expect(shadersSrc).toContain('export function createShaderMaterial');
  });
  it('should define disposeMaterial', () => {
    expect(shadersSrc).toContain('export function disposeMaterial');
  });
  it('should define commonFragment helper', () => {
    expect(shadersSrc).toContain('function commonFragment');
  });
});

describe('vertexShader', () => {
  it('should define vUv varying', () => {
    expect(vertexShader).toContain('varying vec2 vUv');
  });
  it('should set gl_Position', () => {
    expect(vertexShader).toContain('gl_Position');
  });
});

describe('createShaderMaterial structure', () => {
  it('should include uTime uniform', () => {
    expect(shadersSrc).toContain('uTime:');
  });
  it('should include uMirror uniform', () => {
    expect(shadersSrc).toContain('uMirror:');
  });
  it('should include uResolution uniform', () => {
    expect(shadersSrc).toContain('uResolution');
  });
  it('should include uTexture uniform', () => {
    expect(shadersSrc).toContain('uTexture');
  });
});
