export class Analytics {
    constructor() {
        this.sessionId = this._generateId();
        this.startTime = Date.now();
        this.eventQueue = [];
        this.enabled = this._checkConsent();
        this.flushInterval = null;

        if (this.enabled) {
            this._startFlush();
            this.track('session_start', { sessionId: this.sessionId });
            window.addEventListener('beforeunload', () => this.flush(true));
            document.addEventListener('visibilitychange', () => {
                if (document.visibilityState === 'hidden') this.flush(true);
            });
        }
    }

    _generateId() {
        return 's_' + Math.random().toString(36).substring(2, 11);
    }

    _checkConsent() {
        try {
            const consent = localStorage.getItem('analytics_consent');
            if (consent === 'granted') return true;
            if (consent === 'denied') return false;
            return false;
        } catch {
            return false;
        }
    }

    grantConsent() {
        this.enabled = true;
        try { localStorage.setItem('analytics_consent', 'granted'); } catch {}
        this._startFlush();
        this.track('consent_granted');
    }

    denyConsent() {
        this.enabled = false;
        try { localStorage.setItem('analytics_consent', 'denied'); } catch {}
        if (this.flushInterval) clearInterval(this.flushInterval);
    }

    _startFlush() {
        this.flushInterval = setInterval(() => this.flush(), 30000);
    }

    track(type, data = {}) {
        if (!this.enabled) return;
        this.eventQueue.push({
            type,
            timestamp: Date.now(),
            sessionId: this.sessionId,
            data
        });
    }

    flush(force = false) {
        if (!this.enabled || this.eventQueue.length === 0) return;
        const events = [...this.eventQueue];
        this.eventQueue = [];

        const payload = JSON.stringify({ events, sentAt: Date.now() });

        if (force && navigator.sendBeacon) {
            const blob = new Blob([payload], { type: 'application/json' });
            navigator.sendBeacon('/api/analytics', blob);
            return;
        }

        if (navigator.sendBeacon) {
            const blob = new Blob([payload], { type: 'application/json' });
            navigator.sendBeacon('/api/analytics', blob);
        } else {
            this._storePending(events);
        }
    }

    _storePending(events) {
        try {
            const pending = JSON.parse(localStorage.getItem('analytics_pending') || '[]');
            pending.push(...events);
            localStorage.setItem('analytics_pending', JSON.stringify(pending));
        } catch {}
    }

    getSessionDuration() {
        return Date.now() - this.startTime;
    }
}