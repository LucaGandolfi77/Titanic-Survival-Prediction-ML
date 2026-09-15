import * as THREE from 'three';

const vertexShader = `
varying vec2 vUv;
uniform float uTime;
uniform vec3 uGravityDir;
uniform float uCupTilt;

void main() {
    vUv = uv;
    
    // Subtle wave animation for living liquid feel
    vec3 pos = position;
    float wave = sin(pos.x * 12.0 + uTime * 4.0) * 0.003 + cos(pos.y * 8.0 + uTime * 2.5) * 0.003;
    pos.z += wave;
    
    gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
}
`;

const fragmentShader = `
varying vec2 vUv;
uniform float uTime;
uniform float uFillLevel;
uniform vec3 uCoffeeColor;
uniform vec3 uCremaColor;

void main() {
    vec2 center = vec2(0.5, 0.5);
    float dist = distance(vUv, center);
    
    if (dist > 0.48) discard;
    
    // Depth-based shading: darker at edges (cup walls), lighter at center
    float depthFactor = smoothstep(0.0, 0.45, dist);
    vec3 baseColor = mix(uCoffeeColor * 0.6, uCoffeeColor, depthFactor);
    
    // Crema swirl
    float swirl = sin(vUv.x * 8.0 + uTime * 1.5) * cos(vUv.y * 6.0 + uTime * 1.2);
    float cremaMix = smoothstep(0.35, 0.5, dist) * 0.7 + swirl * 0.1;
    vec3 color = mix(baseColor, uCremaColor, cremaMix);
    
    // Foam bubbles
    float bubble1 = fract(sin(dot(vUv.xy * 30.0 + uTime * 2.0, vec2(12.9898, 78.233))) * 43758.5453);
    float bubble2 = fract(sin(dot(vUv.xy * 45.0 - uTime * 1.5, vec2(45.1234, 23.4567))) * 12345.6789);
    if(bubble1 > 0.92 && dist < 0.42) {
        color = mix(color, uCremaColor * 1.4, 0.5);
    }
    if(bubble2 > 0.95 && dist < 0.38) {
        color = vec3(1.0, 0.95, 0.85);
    }
    
    // Surface highlight
    float highlight = smoothstep(0.1, 0.3, dist) * (1.0 - smoothstep(0.35, 0.48, dist));
    color += vec3(0.1, 0.07, 0.03) * highlight * (0.7 + sin(uTime * 2.0) * 0.3);
    
    gl_FragColor = vec4(color, 1.0);
}
`;

export function createCoffeeMaterial() {
    return new THREE.ShaderMaterial({
        vertexShader,
        fragmentShader,
        uniforms: {
            uTime: { value: 0 },
            uFillLevel: { value: 0 },
            uGravityDir: { value: new THREE.Vector3(0, -1, 0) },
            uCupTilt: { value: 0 },
            uCoffeeColor: { value: new THREE.Color(0x3d1a00) },
            uCremaColor: { value: new THREE.Color(0xc8a26a) }
        },
        side: THREE.DoubleSide
    });
}
