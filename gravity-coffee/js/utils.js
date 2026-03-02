export const MathUtils = {
    lerp: (a, b, t) => a + (b - a) * t,
    clamp: (val, min, max) => Math.min(Math.max(val, min), max),
    randomRange: (min, max) => Math.random() * (max - min) + min,
    randomInt: (min, max) => Math.floor(Math.random() * (max - min + 1)) + min,
    randomChoice: (arr) => arr[Math.floor(Math.random() * arr.length)],
    
    // Smoothstep interpolation
    smoothstep: (min, max, value) => {
        let x = Math.max(0, Math.min(1, (value - min) / (max - min)));
        return x * x * (3 - 2 * x);
    },

    // Get angle between two vectors
    angleBetween: (v1, v2) => {
        return v1.angleTo(v2);
    }
};

export const domUtils = {
    show: (id) => document.getElementById(id).classList.remove('hidden'),
    hide: (id) => document.getElementById(id).classList.add('hidden'),
    get: (id) => document.getElementById(id),
    setText: (id, text) => {
        const el = document.getElementById(id);
        if (el) el.innerText = text;
    }
};
