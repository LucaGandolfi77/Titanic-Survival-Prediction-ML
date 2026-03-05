class DialogueSystem {
    constructor() {
        this.queue = [];
        this.active = null;
        this.charIndex = 0;
        this.lastTick = 0;
        this.speed = 30; // chars per second
    }

    enqueue(text, onComplete) {
        this.queue.push({ text, onComplete });
        if (!this.active) this._startNext();
    }

    _startNext() {
        if (this.queue.length === 0) {
            this.active = null;
            return;
        }
        this.active = this.queue.shift();
        this.charIndex = 0;
        this.lastTick = performance.now();
    }

    update(now) {
        if (!this.active) return;
        const elapsed = (now - this.lastTick) / 1000;
        const advance = Math.floor(elapsed * this.speed);
        if (advance > 0) {
            this.charIndex += advance;
            this.lastTick = now;
            if (this.charIndex >= this.active.text.length) {
                this.charIndex = this.active.text.length;
                const cb = this.active.onComplete;
                const finished = this.active.text;
                this._startNext();
                if (cb) cb(finished);
            }
        }
    }

    get visibleText() {
        if (!this.active) return '';
        return this.active.text.slice(0, this.charIndex);
    }
}

export default DialogueSystem;
