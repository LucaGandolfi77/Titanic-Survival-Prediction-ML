// Screen management, menus, transitions
export class UIManager {
    constructor(gameCore) {
        this.game = gameCore;
        this.screens = {
            menu: document.getElementById('screen-menu'),
            howto: document.getElementById('screen-howto'),
            pause: document.getElementById('screen-pause'),
            levelComplete: document.getElementById('screen-level-complete'),
            gameover: document.getElementById('screen-gameover'),
            leaderboard: document.getElementById('screen-leaderboard')
        };
        
        this.hud = document.getElementById('hud');
        this.toastContainer = document.getElementById('toast-container');
        
        this.bindEvents();
        this.initMenuTesseract();
    }
    
    bindEvents() {
        // Menu
        const btnPlay = document.getElementById('btn-play');
        if (btnPlay) btnPlay.addEventListener('click', () => {
            this.showScreen(null);
            if (this.hud) this.hud.classList.remove('hidden');
            this.game.startLevel(1);
        });

        const btnHowto = document.getElementById('btn-howto');
        if (btnHowto) btnHowto.addEventListener('click', () => this.showScreen('howto'));

        const btnHowtoBack = document.getElementById('btn-howto-back');
        if (btnHowtoBack) btnHowtoBack.addEventListener('click', () => this.showScreen('menu'));

        const btnLeaderboard = document.getElementById('btn-leaderboard');
        if (btnLeaderboard) btnLeaderboard.addEventListener('click', () => {
            this.renderLeaderboard();
            this.showScreen('leaderboard');
        });

        const btnLbBack = document.getElementById('btn-lb-back');
        if (btnLbBack) btnLbBack.addEventListener('click', () => this.showScreen('menu'));

        // Pause
        const btnResume = document.getElementById('btn-resume');
        if (btnResume) btnResume.addEventListener('click', () => {
            this.showScreen(null);
            this.game.togglePause();
        });

        const btnRestart = document.getElementById('btn-restart');
        if (btnRestart) btnRestart.addEventListener('click', () => {
            this.showScreen(null);
            this.game.startLevel(this.game.currentLevel);
        });

        const btnToMenu = document.getElementById('btn-to-menu');
        if (btnToMenu) btnToMenu.addEventListener('click', () => this.goToMenu());

        // Level Complete
        const btnNextLevel = document.getElementById('btn-next-level');
        if (btnNextLevel) btnNextLevel.addEventListener('click', () => {
            this.showScreen(null);
            this.game.startLevel(this.game.currentLevel + 1);
        });

        const btnReplay = document.getElementById('btn-replay');
        if (btnReplay) btnReplay.addEventListener('click', () => {
            this.showScreen(null);
            this.game.startLevel(this.game.currentLevel);
        });

        // Game Over
        const btnTryAgain = document.getElementById('btn-tryagain');
        if (btnTryAgain) btnTryAgain.addEventListener('click', () => {
            this.showScreen(null);
            this.game.startLevel(this.game.currentLevel);
        });

        const btnGameoverMenu = document.getElementById('btn-gameover-menu');
        if (btnGameoverMenu) btnGameoverMenu.addEventListener('click', () => this.goToMenu());

        const btnSaveScore = document.getElementById('btn-save-score');
        if (btnSaveScore) btnSaveScore.addEventListener('click', () => this.saveScore());

        // Escape for pause
        window.addEventListener('keydown', e => {
            if (e.key === 'Escape' && this.game.isRunning) {
                this.game.togglePause();
                if (this.game.isPaused) {
                    const pi = document.getElementById('pause-info');
                    if (pi) pi.innerHTML = `Score: ${this.game.score}<br>Level: ${this.game.currentLevel}`;
                    this.showScreen('pause');
                } else {
                    this.showScreen(null);
                }
            }
        });
    }
    
    goToMenu() {
        this.game.isRunning = false;
        this.hud.classList.add('hidden');
        this.showScreen('menu');
    }
    
    showScreen(screenId) {
        Object.values(this.screens).forEach(s => {
            if (s) {
                s.classList.add('hidden');
                s.classList.remove('active');
            }
        });
        if (screenId && this.screens[screenId]) {
            this.screens[screenId].classList.remove('hidden');
            this.screens[screenId].classList.add('active');
        }
    }
    
    showToast(message) {
        if (!this.toastContainer) return;
        const t = document.createElement('div');
        t.className = 'toast';
        t.innerText = message;
        this.toastContainer.appendChild(t);
        setTimeout(() => {
            if (t.parentNode) t.parentNode.removeChild(t);
        }, 3000);
    }
    
    showLevelComplete(stats) {
        this.showScreen('level-complete');
        const title = document.getElementById('level-complete-title');
        if (title) title.innerText = `LEVEL ${stats.level} COMPLETE!`;
        const sb = document.getElementById('score-breakdown');
        if (sb) sb.innerHTML = `Deliveries: ${stats.deliveries}<br>Score: ${stats.score}`;
    }
    
    showGameOver(stats) {
        this.showScreen('gameover');
        const go = document.getElementById('gameover-stats');
        if (go) go.innerHTML = `Reached Level: ${stats.level}<br>Final Score: ${stats.score}`;
    }
    
    saveScore() {
        const initialsEl = document.getElementById('initials-input');
        const initials = (initialsEl && initialsEl.value) ? initialsEl.value : 'AAA';
        const scores = JSON.parse(localStorage.getItem('hds_scores') || '[]');
        scores.push({
            name: initials.toUpperCase(),
            score: this.game.score,
            level: this.game.currentLevel
        });
        scores.sort((a,b) => b.score - a.score);
        localStorage.setItem('hds_scores', JSON.stringify(scores.slice(0, 10)));
        this.showToast('Score Saved!');
        this.renderLeaderboard();
        this.showScreen('leaderboard');
    }
    
    renderLeaderboard() {
        const scores = JSON.parse(localStorage.getItem('hds_scores') || '[]');
        const table = document.getElementById('scores-table');
        if (!table) return;
        table.innerHTML = `<tr><th>Rank</th><th>Name</th><th>Score</th><th>Level</th></tr>`;
        scores.forEach((s, i) => {
            table.innerHTML += `<tr><td>${i+1}</td><td>${s.name}</td><td>${s.score}</td><td>${s.level}</td></tr>`;
        });
    }

    // Mini Tesseract for main menu background
    initMenuTesseract() {
        const canvas = document.getElementById('menu-canvas');
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
        
        let angle = 0;
        
        const draw = () => {
            if (!this.screens.menu || !this.screens.menu.classList.contains('active')) {
                requestAnimationFrame(draw);
                return;
            }
            
            ctx.fillStyle = '#030308';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            
            ctx.save();
            ctx.translate(canvas.width/2, canvas.height/2);
            
            // Generate 16 vertices of tesseract
            const verts = [];
            for(let i=0; i<16; i++) {
                verts.push([
                    (i&1 ? 1 : -1),
                    (i&2 ? 1 : -1),
                    (i&4 ? 1 : -1),
                    (i&8 ? 1 : -1)
                ]);
            }
            
            // Rotate and project
            const cx = Math.cos(angle); const sx = Math.sin(angle);
            const cy = Math.cos(angle*0.5); const sy = Math.sin(angle*0.5);
            
            const p2d = verts.map(v => {
                // Rot XW
                let x1 = v[0]*cx - v[3]*sx;
                let w1 = v[0]*sx + v[3]*cx;
                // Rot YZ
                let y1 = v[1]*cy - v[2]*sy;
                let z1 = v[1]*sy + v[2]*cy;
                
                // project 4d to 3d
                const wDist = 3;
                const p = 1 / (wDist - w1);
                const x2 = x1 * p;
                const y2 = y1 * p;
                const z2 = z1 * p;
                
                // project 3d to 2d
                const zDist = 4;
                const p2 = 1 / (zDist - z2);
                
                const scale = Math.min(canvas.width, canvas.height) * 0.25;
                return { x: x2 * p2 * scale, y: y2 * p2 * scale };
            });
            
            ctx.strokeStyle = '#00e5ff';
            ctx.lineWidth = 1;
            ctx.globalAlpha = 0.5;
            
            // Draw edges
            ctx.beginPath();
            for(let i=0; i<16; i++) {
                for(let j=i+1; j<16; j++) {
                    let diff = 0;
                    if(verts[i][0]!==verts[j][0]) diff++;
                    if(verts[i][1]!==verts[j][1]) diff++;
                    if(verts[i][2]!==verts[j][2]) diff++;
                    if(verts[i][3]!==verts[j][3]) diff++;
                    if(diff === 1) {
                        ctx.moveTo(p2d[i].x, p2d[i].y);
                        ctx.lineTo(p2d[j].x, p2d[j].y);
                    }
                }
            }
            ctx.stroke();
            
            ctx.restore();
            angle += 0.005;
            requestAnimationFrame(draw);
        };
        draw();
    }
}