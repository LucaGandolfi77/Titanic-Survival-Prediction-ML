export class PushManager {
    constructor() {
        this.subscription = null;
        this.permission = 'default';
        this.lastDaily = null;
        this.notifications = [];
    }

    async requestPermission() {
        if (!('Notification' in window)) return false;
        const result = await Notification.requestPermission();
        this.permission = result;
        return result === 'granted';
    }

    async subscribe() {
        if (this.permission !== 'granted') {
            const ok = await this.requestPermission();
            if (!ok) return false;
        }

        if ('serviceWorker' in navigator) {
            try {
                const reg = await navigator.serviceWorker.ready;
                this.subscription = await reg.pushManager.subscribe({
                    userVisibleOnly: true,
                    applicationServerKey: this._generateVapidKey()
                });
                return true;
            } catch (e) {
                console.warn('Push subscription failed:', e);
                return false;
            }
        }
        return false;
    }

    _generateVapidKey() {
        const bytes = new Uint8Array(65);
        crypto.getRandomValues(bytes);
        return bytes;
    }

    async sendLocal(title, body, options = {}) {
        if (this.permission !== 'granted') return false;

        try {
            const notification = new Notification(title, {
                icon: 'data:image/svg+xml,%3Csvg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 192 192%22%3E%3Crect width=%22192%22 height=%22192%22 fill=%22%234a7c59%22 rx=%2224%22/%3E%3Ctext x=%2296%22 y=%22120%22 font-size=%2290%22 text-anchor=%22middle%22%3E%F0%9F%A6%8E%3C/text%3E%3C/svg%3E',
                badge: 'data:image/svg+xml,%3Csvg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 96 96%22%3E%3Crect width=%2296%22 height=%2296%22 fill=%22%234a7c59%22 rx=%2216%22/%3E%3Ctext x=%2248%22 y=%2260%22 font-size=%2240%22 text-anchor=%22middle%22%3E%F0%9F%A6%8E%3C/text%3E%3C/svg%3E',
                tag: options.tag || 'lazykaiju',
                renotify: true,
                body,
                ...options
            });

            this.notifications.push({
                title, body, time: Date.now(), tag: options.tag
            });

            notification.onclick = () => {
                window.focus();
                notification.close();
            };

            return true;
        } catch (e) {
            return false;
        }
    }

    async sendDailyChallenge(level) {
        const today = new Date().toDateString();
        if (this.lastDaily === today) return false;
        this.lastDaily = today;

        try {
            localStorage.setItem('lazykaiju_lastdaily', today);
            const challenges = [
                `Sfida quotidiana: Raccogli 30 rifiuti! 🗑️`,
                `Sfida quotidiana: Non distruggere nessun eco-building! 🌿`,
                `Sfida quotidiana: Raggiungi 200 punti! ⭐`,
                `Sfida quotidiana: Completa il livello in meno di 90s! ⏱️`,
            ];
            const challenge = challenges[Math.floor(Math.random() * challenges.length)];
            return await this.sendLocal('🦎 Lazy Kaiju', challenge, { tag: 'daily' });
        } catch {
            return false;
        }
    }

    isDailyExpired() {
        const today = new Date().toDateString();
        return localStorage.getItem('lazykaiju_lastdaily') !== today;
    }

    getPermission() {
        return this.permission;
    }
}