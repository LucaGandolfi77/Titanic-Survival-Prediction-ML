export class PluginManager {
    constructor() {
        this.plugins = [];
        this.workers = [];
        this.hooks = new Map();
        this._registered = false;
    }

    registerHook(hookName, callback) {
        if (!this.hooks.has(hookName)) {
            this.hooks.set(hookName, []);
        }
        this.hooks.get(hookName).push(callback);
    }

    async loadPlugin(pluginUrl) {
        try {
            const module = await import(/* @vite-ignore */ pluginUrl);
            const plugin = module.default || module;

            if (typeof plugin.install !== 'function') {
                console.warn(`Plugin ${pluginUrl} missing install()`);
                return false;
            }

            const sandbox = this._createSandbox();
            const result = plugin.install(sandbox);

            this.plugins.push({
                name: plugin.name || pluginUrl,
                version: plugin.version || '0.0.1',
                instance: result,
                url: pluginUrl,
                loadedAt: Date.now()
            });

            this._fireHook('pluginLoaded', this.plugins[this.plugins.length - 1]);
            return true;
        } catch (e) {
            console.error(`Failed to load plugin ${pluginUrl}:`, e);
            return false;
        }
    }

    _createSandbox() {
        const self = this;
        return {
            log: (msg) => console.log(`[Plugin] ${msg}`),
            error: (msg) => console.error(`[Plugin] ${msg}`),
            on: (event, cb) => self.registerHook(event, cb),
            off: (event, cb) => {
                const hooks = self.hooks.get(event);
                if (hooks) {
                    const idx = hooks.indexOf(cb);
                    if (idx >= 0) hooks.splice(idx, 1);
                }
            },
            emit: (event, data) => self._fireHook(event, data),
            api: {
                getGameState: () => self._getGameState?.(),
                getScore: () => self._getScore?.(),
                setScore: (val) => self._setScore?.(val),
                showNotification: (text) => self._showNotification?.(text),
            },
            fetch: (url) => fetch(url),
            setTimeout: (fn, ms) => setTimeout(fn, ms),
            clearTimeout: (id) => clearTimeout(id),
        };
    }

    _fireHook(hookName, data) {
        const hooks = this.hooks.get(hookName);
        if (hooks) {
            hooks.forEach(cb => {
                try { cb(data); } catch (e) { console.error(`Hook ${hookName} error:`, e); }
            });
        }
    }

    _registerGameRefs(getState, getScore, setScore, showNotification) {
        this._getGameState = getState;
        this._getScore = getScore;
        this._setScore = setScore;
        this._showNotification = showNotification;
        this._registered = true;
    }

    getAll() {
        return this.plugins;
    }

    unloadAll() {
        this.workers.forEach(w => w.terminate());
        this.workers = [];
        this.plugins.forEach(p => {
            if (p.instance?.unload) p.instance.unload();
        });
        this.plugins = [];
        this.hooks.clear();
    }

    isRegistered() {
        return this._registered;
    }
}