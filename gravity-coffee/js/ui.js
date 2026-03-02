import { domUtils } from './utils.js';

export class UI {
    constructor(engine) {
        this.engine = engine;
        
        // Screens
        this.menu = domUtils.get('screen-menu');
        this.howto = domUtils.get('screen-howto');
        this.pause = domUtils.get('screen-pause');
        this.gameover = domUtils.get('screen-gameover');
        this.settings = domUtils.get('screen-settings');
        this.hud = domUtils.get('hud');
        
        this.bindEvents();
    }
    
    bindEvents() {
        // Main Menu
        domUtils.get('btn-play').addEventListener('click', () => {
            this.hideAll();
            this.engine.startGame();
        });
        
        domUtils.get('btn-howto').addEventListener('click', () => {
            this.hideAll();
            domUtils.show('screen-howto');
        });
        
        domUtils.get('btn-settings').addEventListener('click', () => {
            this.hideAll();
            domUtils.show('screen-settings');
        });
        
        // How to
        domUtils.get('btn-howto-back').addEventListener('click', () => {
            this.hideAll();
            domUtils.show('screen-menu');
        });
        
        // Settings
        domUtils.get('btn-settings-back').addEventListener('click', () => {
            this.hideAll();
            domUtils.show('screen-menu');
        });
        
        // Pause
        domUtils.get('btn-resume').addEventListener('click', () => {
            this.engine.resumeGame();
        });
        
        domUtils.get('btn-restart').addEventListener('click', () => {
            this.hideAll();
            this.engine.startGame();
        });
        
        domUtils.get('btn-pause-menu').addEventListener('click', () => {
            this.hideAll();
            domUtils.show('screen-menu');
            this.engine.endGame();
        });
        
        // Game Over
        domUtils.get('btn-playagain').addEventListener('click', () => {
            this.hideAll();
            this.engine.startGame();
        });
        
        domUtils.get('btn-go-menu').addEventListener('click', () => {
            this.hideAll();
            domUtils.show('screen-menu');
            this.engine.endGame();
        });
        
        // ESC key
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && this.engine.state === 'playing') {
                this.engine.pauseGame();
            }
        });
    }
    
    hideAll() {
        document.querySelectorAll('.screen').forEach(s => s.classList.add('hidden'));
        this.hud.classList.add('hidden');
    }
    
    showHUD() {
        this.hideAll();
        this.hud.classList.remove('hidden');
    }
    
    showPause(score) {
        this.hideAll();
        domUtils.show('screen-pause');
        domUtils.setText('pause-score', `Score: ${score}`);
    }
    
    showGameOver(score, spills) {
        this.hideAll();
        domUtils.show('screen-gameover');
        domUtils.get('gameover-stats').innerHTML = `
            <p>Final Score: ${score}</p>
            <p>Total Spills: ${spills}</p>
        `;
    }
}