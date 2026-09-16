export class ShareManager {
    constructor(gameController) {
        this.game = gameController;
        this.deepLinkData = null;
    }

    async canShare() {
        return !!(navigator.share && navigator.canShare);
    }

    async shareScore() {
        const state = this.game.multiplayer?.isActive
            ? this.game.multiplayer.getGameState()
            : { players: [{ name: 'Solo', score: this.game.score }] };

        const text = state.players.map(p => `${p.name}: ${p.score} punti`).join(' | ');

        if (await this.canShare()) {
            try {
                await navigator.share({
                    title: 'Lazy Kaiju City Cleaner',
                    text: `Ho fatto ${text}! 🦎 Sfidaami!`,
                    url: window.location.href + '?challenge=' + this._encodeChallenge()
                });
                if (this.game.analytics) this.game.analytics.track('shared', { mode: this.game.gameMode });
            } catch (e) {
                this._fallbackCopy(text);
            }
        } else {
            this._fallbackCopy(text);
        }
    }

    async shareChallenge(levelConfig) {
        const encoded = btoa(unescape(encodeURIComponent(JSON.stringify(levelConfig))));
        const url = window.location.href + '?level=' + encoded;

        if (await this.canShare()) {
            try {
                await navigator.share({
                    title: 'Lazy Kaiju — Custom Level',
                    text: 'Gioca a questo livello custom! 🦎',
                    url: url
                });
                if (this.game.analytics) this.game.analytics.track('level_shared');
            } catch (e) {
                this._copyToClipboard(url);
            }
        } else {
            this._copyToClipboard(url);
        }
    }

    _encodeChallenge() {
        const data = {
            score: this.game.score,
            level: this.game.currentLevel,
            mode: this.game.gameMode
        };
        return btoa(unescape(encodeURIComponent(JSON.stringify(data))));
    }

    async handleDeepLink() {
        const url = new URL(window.location.href);
        const levelParam = url.searchParams.get('level');
        const challengeParam = url.searchParams.get('challenge');

        if (levelParam) {
            try {
                const decoded = decodeURIComponent(escape(atob(levelParam)));
                this.deepLinkData = JSON.parse(decoded);
                if (this.game.analytics) this.game.analytics.track('deep_link', { type: 'level' });
                return { type: 'level', data: this.deepLinkData };
            } catch {
                return null;
            }
        }

        if (challengeParam) {
            try {
                const decoded = decodeURIComponent(escape(atob(challengeParam)));
                this.deepLinkData = JSON.parse(decoded);
                if (this.game.analytics) this.game.analytics.track('deep_link', { type: 'challenge', score: this.deepLinkData.score });
                return { type: 'challenge', data: this.deepLinkData };
            } catch {
                return null;
            }
        }

        return null;
    }

    _fallbackCopy(text) {
        navigator.clipboard.writeText(text + ' ' + window.location.href).then(() => {
            if (this.game.analytics) this.game.analytics.track('link_copied');
        }).catch(() => {});
    }

    _copyToClipboard(url) {
        navigator.clipboard.writeText(url).then(() => {
            if (this.game?.analytics) this.game.analytics.track('link_copied');
        }).catch(() => {});
    }
}