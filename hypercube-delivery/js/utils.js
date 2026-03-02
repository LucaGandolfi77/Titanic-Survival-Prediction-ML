// Mathematics, 4D Utilities, and Helpers

/**
 * Projects a 4D point (x,y,z,w) into 3D (x,y,z) using perspective projection
 */
export function project4Dto3D(x, y, z, w, wDistance = 2.5) {
    // Prevent divide by zero if w approaches wDistance
    const perspective = 1 / (wDistance - w);
    return new THREE.Vector3(
        x * perspective,
        y * perspective,
        z * perspective
    );
}

/**
 * Breadth-First Search to find shortest path between two nodes in an adjacency graph
 */
export function findPath(startNode, endNode, graph) {
    if (startNode === endNode) return [startNode];
    
    const queue = [[startNode]];
    const visited = new Set();
    visited.add(startNode);
    
    while (queue.length > 0) {
        const path = queue.shift();
        const node = path[path.length - 1];
        
        const neighbors = graph[node];
        for (const neighbor of neighbors) {
            if (!visited.has(neighbor.target)) {
                visited.add(neighbor.target);
                const newPath = [...path, neighbor.target];
                if (neighbor.target === endNode) {
                    return newPath;
                }
                queue.push(newPath);
            }
        }
    }
    return [];
}

/**
 * Generates the 16 vertices and 32 edges of a tesseract
 * Returns { vertices: Array(16), edges: Array(32 of pairs) }
 */
export function generateTesseract() {
    const vertices = [];
    for (let i = 0; i < 16; i++) {
        vertices.push([
            (i & 1) ? 1 : -1,
            (i & 2) ? 1 : -1,
            (i & 4) ? 1 : -1,
            (i & 8) ? 1 : -1
        ]);
    }
    
    const edges = [];
    for (let i = 0; i < 16; i++) {
        for (let j = i + 1; j < 16; j++) {
            // Edges connect vertices that differ by exactly 1 bit
            let diffs = 0;
            if (vertices[i][0] !== vertices[j][0]) diffs++;
            if (vertices[i][1] !== vertices[j][1]) diffs++;
            if (vertices[i][2] !== vertices[j][2]) diffs++;
            if (vertices[i][3] !== vertices[j][3]) diffs++;
            if (diffs === 1) {
                edges.push([i, j]);
            }
        }
    }
    
    return { vertices, edges };
}

/**
 * Rotates a 4D point (multiplies by two rotation matrices, XW and YZ)
 */
export function rotate4D(point, angleXW, angleYZ) {
    const cxw = Math.cos(angleXW);
    const sxw = Math.sin(angleXW);
    const cyz = Math.cos(angleYZ);
    const syz = Math.sin(angleYZ);
    
    // Rotate in XW plane
    let x = point[0] * cxw - point[3] * sxw;
    let y = point[1];
    let z = point[2];
    let w = point[0] * sxw + point[3] * cxw;
    
    // Rotate in YZ plane
    let newY = y * cyz - z * syz;
    let newZ = y * syz + z * cyz;
    
    return [x, newY, newZ, w];
}

/**
 * Spherical linear interpolation between two quaternions
 */
export function slerpQuaternion(q1, q2, t) {
    const q1c = new THREE.Quaternion().copy(q1);
    const q2c = new THREE.Quaternion().copy(q2);
    return q1c.slerp(q2c, t);
}

/**
 * Easing function for smooth animations
 */
export function easeInOutCubic(x) {
    return x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2;
}

/**
 * Helper to get a nicely formatted time mm:ss
 */
export function formatTime(seconds) {
    const min = Math.floor(seconds / 60);
    const sec = Math.floor(seconds % 60);
    return `${min.toString().padStart(2, '0')}:${sec.toString().padStart(2, '0')}`;
}