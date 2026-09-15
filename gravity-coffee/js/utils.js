export const CONFIG = {
    CUP_POUR_THRESHOLD: Math.PI / 3,
    POUR_DISTANCE: 0.3,
    POUR_FILL_RATE: 0.1,
    SPILL_TILT_THRESHOLD: Math.PI / 3,
    SPILL_PROBABILITY: 0.2,
    MAX_SPILLS: 5,
    GRAVITY_SHIFT_WARNING: 3,
    CUP_RESPAWN_DELAY: 2.0,
    CUSTOMER_SPAWN_INITIAL: 5,
    CUSTOMER_SPAWN_MIN: 15,
    CUSTOMER_SPAWN_MAX: 30,
    MAX_CUSTOMERS: 2,
    CUSTOMER_EXPIRE: 45,
    CUSTOMER_DELIVERY_DIST: 1.5,
    CUSTOMER_FILL_TOLERANCE: 0.15,
    CUSTOMER_PERFECT_BONUS: 50,
    SPOON_GRAB_DIST: 2.0,
    GRAVITY_TRANSITION_DURATION: 2.0,
    CHAOS_SHIFT_INTERVAL: 3.0,
    PHYSICS_MAX_DT: 0.05,
    PHYSICS_BROADPHASE: 9,
    DUST_COUNT: 200,
    DUST_ROOM: { w: 12, h: 8, d: 10 },
    BAR_COUNTER_W: 8,
    BAR_COUNTER_D: 1.5,
    NEON_SIGN_W: 4,
    NEON_SIGN_H: 1,
    NUM_STOOLS: 4,
    NUM_SHELVES: 3,
    EYE_HEIGHT: 1.7,
    SHADOW_MAP_SIZE: 1024,
};

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
    },

    // Seasonal colors based on date
    getSeasonalTheme() {
        const month = new Date().getMonth();
        const day = new Date().getDate();
        if((month === 10 && day >= 20) || (month === 10)) {
            return { // Halloween (Oct 20-Nov 7)
                primary: '#ff6600', secondary: '#9933ff', accent: '#ffcc00',
                dustColor: 0xff6600, alienColors: [0xff3300, 0x9933ff, 0xffcc00],
                name: 'Halloween'
            };
        } else if((month === 11 && day >= 15) || (month === 0 && day <= 5)) {
            return { // Christmas (Dec 15-Jan 5)
                primary: '#cc0000', secondary: '#006600', accent: '#ffdd00',
                dustColor: 0xffdd00, alienColors: [0xcc0000, 0x006600, 0xffdd00],
                name: 'Christmas'
            };
        }
        return null;
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
