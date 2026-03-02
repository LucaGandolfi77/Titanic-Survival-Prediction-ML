// User input handling (Keyboard, Touch, On-Screen UI)
export class Controls {
    constructor() {
        this.keys = {
            w: false, a: false, s: false, d: false,
            space: false,
            up: false, left: false, down: false, right: false
        };
        
        this.touch = { active: false, x: 0, y: 0, accel: false };
        
        this.accel = 0;
        this.reverse = 0;
        this.steer = 0;
        this.brake = false;
        
        this.setupKeyboard();
        this.setupOnScreen();
    }
    
    setupKeyboard() {
        const setKey = (e, state) => {
            const key = e.key.toLowerCase();
            if (key === 'w' || key === 'arrowup') this.keys.w = state;
            if (key === 's' || key === 'arrowdown') this.keys.s = state;
            if (key === 'a' || key === 'arrowleft') this.keys.a = state;
            if (key === 'd' || key === 'arrowright') this.keys.d = state;
            if (key === ' ') this.keys.space = state;
        };
        
        window.addEventListener('keydown', e => setKey(e, true));
        window.addEventListener('keyup', e => setKey(e, false));
    }
    
    setupOnScreen() {
        // Touch events for the left joystick
        const stickArea = document.getElementById('left-joystick-area');
        const knob = document.getElementById('left-stick-knob');
        const brakeBtn = document.getElementById('btn-brake');
        
        if (!stickArea) return;
        
        let origin = {x:0, y:0};
        
        stickArea.addEventListener('touchstart', e => {
            const touch = e.changedTouches[0];
            const rect = stickArea.getBoundingClientRect();
            // Center of stick area
            origin = {
                x: rect.left + rect.width / 2,
                y: rect.top + rect.height / 2
            };
            this.touch.active = true;
            this.updateStick(touch, origin, knob);
        });

        stickArea.addEventListener('touchmove', e => {
            e.preventDefault();
            if(!this.touch.active) return;
            this.updateStick(e.changedTouches[0], origin, knob);
        });

        const resetStick = () => {
            this.touch.active = false;
            this.touch.x = 0;
            this.touch.y = 0;
            knob.style.transform = `translate(0px, 0px)`;
        };

        stickArea.addEventListener('touchend', resetStick);
        stickArea.addEventListener('touchcancel', resetStick);
        
        // Right side screen accelerating (fallback) or gas pedal logic
        // We tie accelerating to up on joypad for simplicity currently
        
        // Brake button
        brakeBtn.addEventListener('touchstart', e => { e.preventDefault(); this.keys.space = true; });
        brakeBtn.addEventListener('touchend', e => { e.preventDefault(); this.keys.space = false; });
    }
    
    updateStick(touch, origin, knob) {
        let dx = touch.clientX - origin.x;
        let dy = touch.clientY - origin.y;
        
        const dist = Math.sqrt(dx*dx + dy*dy);
        const maxRadius = 35;
        
        if (dist > maxRadius) {
            dx = (dx / dist) * maxRadius;
            dy = (dy / dist) * maxRadius;
        }
        
        knob.style.transform = `translate(${dx}px, ${dy}px)`;
        
        this.touch.x = dx / maxRadius;
        this.touch.y = dy / maxRadius; // positive is down
    }

    update() {
        this.accel = 0;
        this.reverse = 0;
        this.steer = 0;
        this.brake = this.keys.space;
        
        // Keyboard mapping
        if (this.keys.w) this.accel = 1;
        if (this.keys.s) this.reverse = 1;
        if (this.keys.a) this.steer = 1;  // Right-handed sys: positive steer is left turn
        if (this.keys.d) this.steer = -1;
        
        // Touch mapping overrides keyboard if active
        if (this.touch.active) {
            this.steer = -this.touch.x; // invert for natural feel
            if (this.touch.y < -0.2) this.accel = Math.min(1, Math.abs(this.touch.y) * 1.5);
            if (this.touch.y > 0.2) this.reverse = Math.min(1, this.touch.y * 1.5);
        }
    }
}