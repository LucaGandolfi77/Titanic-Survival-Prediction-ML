export class MultiplayerManager {
    constructor(sceneManager, cityGenerator, onGameEnd) {
        this.sceneMgr = sceneManager;
        this.city = cityGenerator;
        this.onGameEnd = onGameEnd;
        this.isActive = false;

        this.players = [];
        this.kaijuInstances = [];
        this.cameraInstances = [];
        this.rendererInstances = [];

        this.winningScore = 50;
    }

    activate(playerNames) {
        this.isActive = true;
        this._createPlayers(playerNames);
    }

    deactivate() {
        this.isActive = false;
    }

    _createPlayers(names) {
        names.forEach((name, index) => {
            const viewport = index === 0
                ? { x: 0, y: 0, width: 0.5, height: 1.0 }
                : { x: 0.5, y: 0, width: 0.5, height: 1.0 };

            const player = {
                name,
                index,
                score: 0,
                karma: 100,
                isGameOver: false,
                winner: false,
                viewport,
                keyPrefix: index === 0 ? 'P1_' : 'P2_'
            };

            this.players.push(player);
        });
    }

    getInput(index) {
        const prefix = this.players[index]?.keyPrefix || '';
        return {
            W: window.keys[`${prefix}W`] || false,
            A: window.keys[`${prefix}A`] || false,
            S: window.keys[`${prefix}S`] || false,
            D: window.keys[`${prefix}D`] || false,
            SPACE: window.keys[`${prefix}SPACE`] || false
        };
    }

    addScore(playerIndex, points) {
        const player = this.players[playerIndex];
        if (!player || player.isGameOver) return;
        player.score += points;

        if (player.score >= this.winningScore) {
            player.winner = true;
            player.isGameOver = true;
            this.announceWinner(player);
        }
    }

    addKarmaPenalty(playerIndex, amount) {
        const player = this.players[playerIndex];
        if (!player || player.isGameOver) return;
        player.karma = Math.max(0, player.karma + amount);
        if (player.karma <= 0) {
            player.isGameOver = true;
            const other = this.players[1 - playerIndex];
            if (other && !other.isGameOver) {
                other.winner = true;
                this.announceWinner(other);
            }
        }
    }

    announceWinner(winner) {
        if (this.onGameEnd) {
            this.onGameEnd(winner, this.players);
        }
    }

    getGameState() {
        return {
            isActive: this.isActive,
            players: this.players.map(p => ({
                name: p.name,
                score: p.score,
                karma: p.karma,
                isGameOver: p.isGameOver,
                winner: p.winner
            })),
            winningScore: this.winningScore
        };
    }
}
