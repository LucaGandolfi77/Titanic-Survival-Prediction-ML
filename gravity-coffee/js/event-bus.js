export class EventBus {
    constructor() {
        this._listeners = {};
    }

    on(event, callback) {
        if (!this._listeners[event]) this._listeners[event] = [];
        this._listeners[event].push(callback);
        return () => this.off(event, callback);
    }

    off(event, callback) {
        if (!this._listeners[event]) return;
        const idx = this._listeners[event].indexOf(callback);
        if (idx >= 0) this._listeners[event].splice(idx, 1);
    }

    emit(event, data) {
        if (!this._listeners[event]) return;
        for (const cb of this._listeners[event]) {
            try { cb(data); } catch(e) { console.error(`EventBus error [${event}]:`, e); }
        }
    }

    once(event, callback) {
        const unsubscribe = this.on(event, (data) => {
            unsubscribe();
            callback(data);
        });
        return unsubscribe;
    }

    clear() {
        this._listeners = {};
    }
}

export const bus = new EventBus();
