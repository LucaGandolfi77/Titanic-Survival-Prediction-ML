/*
 * shaders.js
 * Shader definitions and helpers for full-screen video post-processing filters.
 */
import * as THREE from 'three';

export const SHADER_FILTERS = [
  { id: 'normal', name: 'Normal', category: 'shader' },
  { id: 'vintage', name: 'Vintage', category: 'shader' },
  { id: 'neon', name: 'Neon', category: 'shader' },
  { id: 'glitch', name: 'Glitch', category: 'shader' },
  { id: 'pixelate', name: 'Pixelate', category: 'shader' },
  { id: 'sketch', name: 'Sketch', category: 'shader' },
  { id: 'blur', name: 'Blur', category: 'shader' },
  { id: 'warm-glow', name: 'Warm Glow', category: 'shader' },
  { id: 'cool-bleach', name: 'Cool Bleach', category: 'shader' },
  { id: 'film-grain', name: 'Film Grain', category: 'shader' },
];

export const vertexShader = /* glsl */`
  varying vec2 vUv;
  void main() {
    vUv = uv;
    gl_Position = vec4(position.xy, 0.0, 1.0);
  }
`;

function commonFragment(body) {
  return /* glsl */`
    precision highp float;
    uniform sampler2D uTexture;
    uniform vec2 uResolution;
    uniform float uTime;
    uniform float uMirror;
    varying vec2 vUv;

    float luma(vec3 c){ return dot(c, vec3(0.299, 0.587, 0.114)); }
    float rand(vec2 co){ return fract(sin(dot(co.xy, vec2(12.9898, 78.233))) * 43758.5453); }

    void main() {
      vec2 uv = vUv;
      float sampleX = mix(uv.x, 1.0 - uv.x, uMirror);
      vec4 base = texture2D(uTexture, vec2(sampleX, uv.y));
      vec3 color = base.rgb;
      ${body}
      gl_FragColor = vec4(color, 1.0);
    }
  `;
}

const FRAGMENTS = {
  normal: commonFragment(``),
  vintage: commonFragment(`
    color = vec3(
      dot(color, vec3(0.393, 0.769, 0.189)),
      dot(color, vec3(0.349, 0.686, 0.168)),
      dot(color, vec3(0.272, 0.534, 0.131))
    );
    vec2 centered = uv - 0.5;
    float vignette = smoothstep(0.9, 0.2, length(centered));
    color *= mix(0.72, 1.0, vignette);
  `),
  neon: commonFragment(`
    color = pow(color, vec3(0.9));
    color *= vec3(1.35, 1.15, 1.55);
    float glow = smoothstep(0.3, 1.0, luma(color));
    color += vec3(0.0, 0.1, 0.22) * glow;
  `),
  glitch: commonFragment(`
    float wave = sin(uv.y * 80.0 + uTime * 8.0) * 0.004;
    float noise = rand(vec2(floor(uv.y * 160.0), floor(uTime * 12.0))) * 0.008;
    float shift = wave + noise;
    float r = texture2D(uTexture, vec2(1.0 - uv.x + shift, uv.y)).r;
    float g = texture2D(uTexture, vec2(1.0 - uv.x, uv.y)).g;
    float b = texture2D(uTexture, vec2(1.0 - uv.x - shift, uv.y)).b;
    color = vec3(r, g, b);
  `),
  pixelate: commonFragment(`
    vec2 size = max(vec2(6.0), uResolution / 180.0);
    vec2 px = floor(uv * size) / size;
    color = texture2D(uTexture, vec2(mix(px.x, 1.0 - px.x, uMirror), px.y)).rgb;
  `),
  sketch: commonFragment(`
    vec2 texel = 1.0 / uResolution;
    float tl = luma(texture2D(uTexture, vec2(mix(uv.x - texel.x, 1.0 - (uv.x - texel.x), uMirror), uv.y - texel.y)).rgb);
    float tr = luma(texture2D(uTexture, vec2(mix(uv.x + texel.x, 1.0 - (uv.x + texel.x), uMirror), uv.y - texel.y)).rgb);
    float bl = luma(texture2D(uTexture, vec2(mix(uv.x - texel.x, 1.0 - (uv.x - texel.x), uMirror), uv.y + texel.y)).rgb);
    float br = luma(texture2D(uTexture, vec2(mix(uv.x + texel.x, 1.0 - (uv.x + texel.x), uMirror), uv.y + texel.y)).rgb);
    float edge = abs(tl - br) + abs(tr - bl);
    color = vec3(1.0 - smoothstep(0.08, 0.28, edge));
  `),
  blur: commonFragment(`
    vec2 texel = 1.0 / uResolution;
    vec3 sum = vec3(0.0);
    sum += texture2D(uTexture, vec2(mix(uv.x, 1.0 - uv.x, uMirror), uv.y)).rgb * 0.227027;
    sum += texture2D(uTexture, vec2(mix(uv.x + 1.384615 * texel.x, 1.0 - (uv.x + 1.384615 * texel.x), uMirror), uv.y)).rgb * 0.316216;
    sum += texture2D(uTexture, vec2(mix(uv.x - 1.384615 * texel.x, 1.0 - (uv.x - 1.384615 * texel.x), uMirror), uv.y)).rgb * 0.316216;
    sum += texture2D(uTexture, vec2(mix(uv.x + 3.230769 * texel.x, 1.0 - (uv.x + 3.230769 * texel.x), uMirror), uv.y)).rgb * 0.07027;
    sum += texture2D(uTexture, vec2(mix(uv.x - 3.230769 * texel.x, 1.0 - (uv.x - 3.230769 * texel.x), uMirror), uv.y)).rgb * 0.07027;
    color = sum;
  `),
  'warm-glow': commonFragment(`
    color = color * vec3(1.08, 1.02, 0.9);
    vec3 glow = vec3(0.12, 0.06, 0.02) * smoothstep(0.1, 0.9, luminance(color));
    color += glow;
  `),
  'cool-bleach': commonFragment(`
    color = pow(color, vec3(0.95));
    color = mix(color, vec3(0.85, 0.9, 1.0), 0.12);
    color = color * vec3(0.92, 0.98, 1.05);
  `),
  'film-grain': commonFragment(`
    float g = rand(uv * uResolution.xy + uTime);
    color += (g - 0.5) * 0.03;
    // subtle desat
    float avg = dot(color, vec3(0.3333));
    color = mix(color, vec3(avg), 0.06);
  `),
};

export function createShaderMaterial(filterId, videoTexture, resolution) {
  return new THREE.ShaderMaterial({
    uniforms: {
      uTexture: { value: videoTexture },
      uResolution: { value: new THREE.Vector2(resolution.width, resolution.height) },
      uTime: { value: 0 },
      uMirror: { value: 1 },
    },
    vertexShader,
    fragmentShader: FRAGMENTS[filterId] || FRAGMENTS.normal,
  });
}

export function disposeMaterial(material) {
  if (!material) return;
  material.dispose();
}
