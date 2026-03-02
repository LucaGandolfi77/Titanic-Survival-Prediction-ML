import * as THREE from 'https://cdnjs.cloudflare.com/ajax/libs/three.js/r158/three.module.js';

export const MathUtils = {
    lerp: (a, b, t) => a + (b - a) * t,
    clamp: (val, min, max) => Math.min(Math.max(val, min), max),
    randRange: (min, max) => Math.random() * (max - min) + min,
    randInt: (min, max) => Math.floor(Math.random() * (max - min + 1)) + min,
    randItem: (arr) => arr[Math.floor(Math.random() * arr.length)],
    
    // Smooth angle interpolation handling wrap-around
    lerpAngle: (a, b, t) => {
        let diff = b - a;
        while (diff > Math.PI) diff -= Math.PI * 2;
        while (diff < -Math.PI) diff += Math.PI * 2;
        return a + diff * t;
    }
};

export const dom = {
    get: (id) => document.getElementById(id),
    show: (id) => document.getElementById(id).classList.remove('hidden'),
    hide: (id) => document.getElementById(id).classList.add('hidden'),
    setTxt: (id, txt) => { const el = document.getElementById(id); if(el) el.innerText = txt; }
};

// Simple AABB Box vs Point check
export function checkPointInAABB(point, boxCenter, boxSize) {
    return (
        point.x >= boxCenter.x - boxSize.x/2 && point.x <= boxCenter.x + boxSize.x/2 &&
        point.y >= boxCenter.y - boxSize.y/2 && point.y <= boxCenter.y + boxSize.y/2 &&
        point.z >= boxCenter.z - boxSize.z/2 && point.z <= boxCenter.z + boxSize.z/2
    );
}

export function createScreenPopup(text, position3D, camera, color) {
    const vector = position3D.clone();
    vector.project(camera);
    
    const x = (vector.x * .5 + .5) * window.innerWidth;
    const y = (vector.y * -.5 + .5) * window.innerHeight;
    
    const div = document.createElement('div');
    div.className = 'event-popup';
    div.innerText = text;
    div.style.left = `${x}px`;
    div.style.top = `${y}px`;
    div.style.color = color || 'var(--text-primary)';
    
    document.getElementById('event-popups').appendChild(div);
    setTimeout(() => div.remove(), 1200);
}