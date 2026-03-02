import * as THREE from 'https://cdnjs.cloudflare.com/ajax/libs/three.js/r158/three.module.js';

const vertexShader = `
varying vec2 vUv;
varying vec3 vWorldPosition;
uniform float uTime;
uniform vec3 uGravityDir;
uniform float uCupTilt;

void main() {
    vUv = uv;
    
    // Slight wave animation
    vec3 pos = position;
    float wave = sin(pos.x * 8.0 + uTime * 3.0) * 0.005 + cos(pos.y * 6.0 + uTime * 2.0) * 0.005;
    pos.z += wave; // In coordinate space of plane (Z is local up if rotated)
    
    // Tilt correction geometry to keep surface level-ish
    // In a real robust liquid shader, this requires complex matrix math or fluid sim.
    // Here we approximate it by tilting the vertices based on the cup tilt.
    // If cup tilts, we want the liquid plane to counter-rotate.
    // Since it's a simple plane, this visual trick works enough for the demo.
    
    vec4 worldPos = modelMatrix * vec4(pos, 1.0);
    vWorldPosition = worldPos.xyz;
    
    gl_Position = projectionMatrix * viewMatrix * worldPos;
}
`;

const fragmentShader = `
varying vec2 vUv;
varying vec3 vWorldPosition;
uniform float uTime;
uniform float uFillLevel;
uniform vec3 uCoffeeColor;
uniform vec3 uCremaColor;

void main() {
    // Distance from center for circular cup mapping
    vec2 center = vec2(0.5, 0.5);
    float dist = distance(vUv, center);
    
    if (dist > 0.48) discard; // Clip to circle
    
    // Basic crema pattern
    vec3 color = mix(uCoffeeColor, uCremaColor, smoothstep(0.3, 0.5, dist));
    
    // Add some noise/foam bubbles
    float noise = fract(sin(dot(vUv.xy + uTime*0.1, vec2(12.9898,78.233))) * 43758.5453);
    if(dist > 0.4 && noise > 0.8) {
        color = uCremaColor * 1.2;
    }
    
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