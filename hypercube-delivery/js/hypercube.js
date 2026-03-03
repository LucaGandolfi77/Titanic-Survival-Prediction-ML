import * as THREE from 'three';

// Tesseract math, cell topography, and adjacency

export const HYPERCUBE_GRAPH = {
    // Each cell has 6 portals (walls: +x, -x, +y, -y, +z, -z)
    // Connecting to another cell, indicating which wall on the destination cell it maps to.
    
    // Cell 0: The Inner Cube
    0: [
        { dir: '+x', target: 1, targetWall: '-x', axis: new THREE.Vector3(0, 1, 0), angle: -Math.PI / 2 },
        { dir: '-x', target: 2, targetWall: '+x', axis: new THREE.Vector3(0, 1, 0), angle: Math.PI / 2 },
        { dir: '+y', target: 3, targetWall: '-y', axis: new THREE.Vector3(1, 0, 0), angle: Math.PI / 2 },
        { dir: '-y', target: 4, targetWall: '+y', axis: new THREE.Vector3(1, 0, 0), angle: -Math.PI / 2 },
        { dir: '+z', target: 5, targetWall: '-z', axis: new THREE.Vector3(0, 0, 1), angle: 0 }, // Wait, proper 4D fold requires more complex math, but for gameplay:
        { dir: '-z', target: 6, targetWall: '+z', axis: new THREE.Vector3(0, 0, 1), angle: 0 }
    ],
    
    // Cell 1: +X Outer
    1: [
        { dir: '-x', target: 0, targetWall: '+x', axis: new THREE.Vector3(0, 1, 0), angle: Math.PI / 2 },
        { dir: '+x', target: 7, targetWall: '-x', axis: new THREE.Vector3(0, 1, 0), angle: -Math.PI / 2 },
        { dir: '+y', target: 3, targetWall: '+x', axis: new THREE.Vector3(0, 0, 1), angle: Math.PI / 2 },
        { dir: '-y', target: 4, targetWall: '+x', axis: new THREE.Vector3(0, 0, 1), angle: -Math.PI / 2 },
        { dir: '+z', target: 5, targetWall: '+x', axis: new THREE.Vector3(0, 1, 0), angle: Math.PI / 2 },
        { dir: '-z', target: 6, targetWall: '+x', axis: new THREE.Vector3(0, 1, 0), angle: Math.PI / 2 }
    ],
    
    // Cell 2: -X Outer
    2: [
        { dir: '+x', target: 0, targetWall: '-x', axis: new THREE.Vector3(0, 1, 0), angle: -Math.PI / 2 },
        { dir: '-x', target: 7, targetWall: '+x', axis: new THREE.Vector3(0, 1, 0), angle: Math.PI / 2 },
        { dir: '+y', target: 3, targetWall: '-x', axis: new THREE.Vector3(0, 0, 1), angle: -Math.PI / 2 },
        { dir: '-y', target: 4, targetWall: '-x', axis: new THREE.Vector3(0, 0, 1), angle: Math.PI / 2 },
        { dir: '+z', target: 5, targetWall: '-x', axis: new THREE.Vector3(0, 1, 0), angle: -Math.PI / 2 },
        { dir: '-z', target: 6, targetWall: '-x', axis: new THREE.Vector3(0, 1, 0), angle: -Math.PI / 2 }
    ],
    
    // Cell 3: +Y Outer
    3: [
        { dir: '-y', target: 0, targetWall: '+y', axis: new THREE.Vector3(1, 0, 0), angle: -Math.PI / 2 },
        { dir: '+y', target: 7, targetWall: '-y', axis: new THREE.Vector3(1, 0, 0), angle: Math.PI / 2 },
        { dir: '+x', target: 1, targetWall: '+y', axis: new THREE.Vector3(0, 0, 1), angle: -Math.PI / 2 },
        { dir: '-x', target: 2, targetWall: '+y', axis: new THREE.Vector3(0, 0, 1), angle: Math.PI / 2 },
        { dir: '+z', target: 5, targetWall: '+y', axis: new THREE.Vector3(1, 0, 0), angle: Math.PI / 2 },
        { dir: '-z', target: 6, targetWall: '+y', axis: new THREE.Vector3(1, 0, 0), angle: Math.PI / 2 }
    ],
    
    // Cell 4: -Y Outer
    4: [
        { dir: '+y', target: 0, targetWall: '-y', axis: new THREE.Vector3(1, 0, 0), angle: Math.PI / 2 },
        { dir: '-y', target: 7, targetWall: '+y', axis: new THREE.Vector3(1, 0, 0), angle: -Math.PI / 2 },
        { dir: '+x', target: 1, targetWall: '-y', axis: new THREE.Vector3(0, 0, 1), angle: Math.PI / 2 },
        { dir: '-x', target: 2, targetWall: '-y', axis: new THREE.Vector3(0, 0, 1), angle: -Math.PI / 2 },
        { dir: '+z', target: 5, targetWall: '-y', axis: new THREE.Vector3(1, 0, 0), angle: -Math.PI / 2 },
        { dir: '-z', target: 6, targetWall: '-y', axis: new THREE.Vector3(1, 0, 0), angle: -Math.PI / 2 }
    ],
    
    // Cell 5: +Z Outer
    5: [
        { dir: '-z', target: 0, targetWall: '+z', axis: new THREE.Vector3(0, 0, 1), angle: 0 },
        { dir: '+z', target: 7, targetWall: '-z', axis: new THREE.Vector3(0, 0, 1), angle: 0 },
        { dir: '+x', target: 1, targetWall: '+z', axis: new THREE.Vector3(0, 1, 0), angle: -Math.PI / 2 },
        { dir: '-x', target: 2, targetWall: '+z', axis: new THREE.Vector3(0, 1, 0), angle: Math.PI / 2 },
        { dir: '+y', target: 3, targetWall: '+z', axis: new THREE.Vector3(1, 0, 0), angle: -Math.PI / 2 },
        { dir: '-y', target: 4, targetWall: '+z', axis: new THREE.Vector3(1, 0, 0), angle: Math.PI / 2 }
    ],
    
    // Cell 6: -Z Outer
    6: [
        { dir: '+z', target: 0, targetWall: '-z', axis: new THREE.Vector3(0, 0, 1), angle: 0 },
        { dir: '-z', target: 7, targetWall: '+z', axis: new THREE.Vector3(0, 0, 1), angle: 0 },
        { dir: '+x', target: 1, targetWall: '-z', axis: new THREE.Vector3(0, 1, 0), angle: Math.PI / 2 },
        { dir: '-x', target: 2, targetWall: '-z', axis: new THREE.Vector3(0, 1, 0), angle: -Math.PI / 2 },
        { dir: '+y', target: 3, targetWall: '-z', axis: new THREE.Vector3(1, 0, 0), angle: Math.PI / 2 },
        { dir: '-y', target: 4, targetWall: '-z', axis: new THREE.Vector3(1, 0, 0), angle: -Math.PI / 2 }
    ],
    
    // Cell 7: The Outer Cube
    7: [
        { dir: '-x', target: 1, targetWall: '+x', axis: new THREE.Vector3(0, 1, 0), angle: Math.PI / 2 },
        { dir: '+x', target: 2, targetWall: '-x', axis: new THREE.Vector3(0, 1, 0), angle: -Math.PI / 2 },
        { dir: '-y', target: 3, targetWall: '+y', axis: new THREE.Vector3(1, 0, 0), angle: -Math.PI / 2 },
        { dir: '+y', target: 4, targetWall: '-y', axis: new THREE.Vector3(1, 0, 0), angle: Math.PI / 2 },
        { dir: '-z', target: 5, targetWall: '+z', axis: new THREE.Vector3(0, 0, 1), angle: 0 },
        { dir: '+z', target: 6, targetWall: '-z', axis: new THREE.Vector3(0, 0, 1), angle: 0 }
    ]
};

// Simplified adjacency for gameplay: when going through a portal, 
// the worldContainer rotates around 'axis' by 'angle' radians.

export const CELL_THEMES = {
    0: { name: "Central Terminal", type: "urban",     bg: "#1a237e", fog: "#1a237e", accent: "#4fc3f7" },
    1: { name: "Scrap Rooftop",    type: "rooftop",   bg: "#ff6f00", fog: "#331600", accent: "#ff8f00" },
    2: { name: "Neon Tunnels",     type: "tunnel",    bg: "#1a0033", fog: "#1a0033", accent: "#e040fb" },
    3: { name: "Floating Isles",   type: "floating",  bg: "#0d0d0d", fog: "#0d0d0d", accent: "#00e676" },
    4: { name: "Deep Storage",     type: "warehouse", bg: "#2d2a13", fog: "#2d2a13", accent: "#ffeb3b" },
    5: { name: "Zenith Park",      type: "park",      bg: "#87ceeb", fog: "#87ceeb", accent: "#81c784" },
    6: { name: "Crimson Bazaar",   type: "market",    bg: "#330000", fog: "#330000", accent: "#ff5252" },
    7: { name: "The Void",         type: "void",      bg: "#000000", fog: "#000000", accent: "#ffffff" }
};

export const PORTAL_POSITIONS = {
    '+x': { pos: new THREE.Vector3( 49.5, 5, 0),   rot: new THREE.Euler(0, -Math.PI/2, 0) },
    '-x': { pos: new THREE.Vector3(-49.5, 5, 0),   rot: new THREE.Euler(0, Math.PI/2, 0) },
    '+y': { pos: new THREE.Vector3(0,  49.5, 0),   rot: new THREE.Euler(Math.PI/2, 0, 0) },
    '-y': { pos: new THREE.Vector3(0, -49.5, 0),   rot: new THREE.Euler(-Math.PI/2, 0, 0) },
    '+z': { pos: new THREE.Vector3(0, 5,  49.5),   rot: new THREE.Euler(0, Math.PI, 0) },
    '-z': { pos: new THREE.Vector3(0, 5, -49.5),   rot: new THREE.Euler(0, 0, 0) }
};
