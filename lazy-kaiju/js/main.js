// Main Entry Point
import * as THREE from 'three';
import { MathUtils } from './utils.js';
import { SceneManager } from './scene.js';
import { CityGenerator } from './city.js';
import { Kaiju } from './kaiju.js';
import { Tail } from './tail.js';
import { TrashManager } from './items.js';
import { GameAudio } from './audio.js';
import { Analytics } from './analytics.js';
import { LevelEditor } from './editor.js';
import { MultiplayerManager } from './multiplayer.js';
import { ActivistNPC } from './npc.js';
import { ProceduralGenerator } from './procedural.js';
import { MusicEngine } from './music.js';
import { GestureRecognition } from './gesture.js';
import { ShareManager } from './share.js';
import { AchievementSystem } from './gamification.js';
import { PushManager } from './push.js';
import { WebRTCAudio } from './webrtc.js';
import { ARMode } from './ar.js';
import { IoTConnector } from './iot.js';
import { PluginManager } from './plugins.js';

class GameController {
    constructor() {
        this.gameState = 'start';
        this.gameMode = 'single';
        this.score = 0;
        this.karma = 100;
        this.stamina = 100;
        this.maxStamina = 100;

        this.sceneMgr = new SceneManager();
        this.city = new CityGenerator(this.sceneMgr);
        this.city.generate({ ecoCount: 8, maxHeight: 22 });

        this.kaiju = new Kaiju(this.sceneMgr.scene);
        this.tail = new Tail(this.sceneMgr.scene, this.kaiju, 12, 1.5);
        this.trashMgr = new TrashManager(this.sceneMgr.scene, this.city);

        window.gameAudio = new GameAudio();

        this.keys = { W:false, A:false, S:false, D:false, SPACE:false };
        this.inputVec = new THREE.Vector3();

        this.sweepCooldown = 0;
        this.isSweeping = false;
        this._sweepTimeout = null;

        this.analytics = new Analytics();
        this.multiplayer = new MultiplayerManager(this.sceneMgr, this.city, (winner, players) => {
            this._onMultiplayerEnd(winner, players);
        });
        this.levelEditor = null;

        this.npcs = [];
        this.procedural = new ProceduralGenerator();
        this.music = new MusicEngine();
        this.gesture = new GestureRecognition();
        this.share = new ShareManager(this);
        this.achievements = new AchievementSystem();
        this.push = new PushManager();
        this.webrtc = new WebRTCAudio();
        this.ar = new ARMode(this.sceneMgr);
        this.iot = new IoTConnector();
        this.plugins = new PluginManager();
        this.currentLevel = 1;
        this.isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent) || (window.innerWidth < 768);
        this.sessionStart = Date.now();
        this._lastAmbientCheck = 0;

        this.lastTime = performance.now();
        this.showConsentBanner();
        this.setupBindings();
        this.music.init();

        this.ar.checkSupport().then(support => {
            if (support.ar || support.deviceOrientation) {
                this._arSupported = true;
            }
        });

        this._handleDeepLink();
        this._initGameRefs();

        requestAnimationFrame(t => this.loop(t));
    }

    showConsentBanner() {
        if (this.analytics.enabled) return;
        const banner = document.getElementById('consent-banner');
        if (!banner) return;
        banner.classList.remove('hidden');
        banner.style.display = 'flex';

        const acceptBtn = document.getElementById('consent-accept');
        const declineBtn = document.getElementById('consent-decline');

        if (acceptBtn) {
            acceptBtn.addEventListener('click', () => {
                this.analytics.grantConsent();
                banner.classList.add('hidden');
                banner.style.display = 'none';
            });
        }
        if (declineBtn) {
            declineBtn.addEventListener('click', () => {
                this.analytics.denyConsent();
                banner.classList.add('hidden');
                banner.style.display = 'none';
            });
        }
    }

    setupBindings() {
        window.addEventListener('keydown', e => {
            const key = e.key.toUpperCase();
            if(this.keys.hasOwnProperty(key)) this.keys[key] = true;
            if(e.code === 'Space') this.keys.SPACE = true;
            if(this.gameMode === 'multi') {
                if(e.code === 'KeyI') this.keys['P2_I'] = true;
                if(e.code === 'KeyK') this.keys['P2_K'] = true;
                if(e.code === 'KeyJ') this.keys['P2_J'] = true;
                if(e.code === 'KeyL') this.keys['P2_L'] = true;
                if(e.code === 'Enter') this.keys['P2_SPACE'] = true;
            }
        });
        window.addEventListener('keyup', e => {
            const key = e.key.toUpperCase();
            if(this.keys.hasOwnProperty(key)) this.keys[key] = false;
            if(e.code === 'Space') this.keys.SPACE = false;
            if(this.gameMode === 'multi') {
                if(e.code === 'KeyI') this.keys['P2_I'] = false;
                if(e.code === 'KeyK') this.keys['P2_K'] = false;
                if(e.code === 'KeyJ') this.keys['P2_J'] = false;
                if(e.code === 'KeyL') this.keys['P2_L'] = false;
                if(e.code === 'Enter') this.keys['P2_SPACE'] = false;
            }
        });

        const startBtn = document.getElementById('start-btn');
        if (startBtn) startBtn.addEventListener('click', () => {
            const ss = document.getElementById('start-screen'); if (ss) ss.classList.add('hidden');
            if (window.gameAudio && typeof window.gameAudio.resume === 'function') window.gameAudio.resume();
            this.gameState = 'playing';
            this.resetStats();
        });

        const btnPlay = document.getElementById('btn-play');
        if (btnPlay) btnPlay.addEventListener('click', () => {
            this.gameMode = 'single';
            this.startSinglePlayer();
        });

        const btn2P = document.getElementById('btn-2p');
        if (btn2P) btn2P.addEventListener('click', () => {
            const mp = document.getElementById('screen-multiplayer');
            if (mp) mp.classList.remove('hidden');
        });

        const btnEditor = document.getElementById('btn-editor');
        if (btnEditor) btnEditor.addEventListener('click', () => {
            this.openEditor();
        });

        const btnMpStart = document.getElementById('btn-mp-start');
        if (btnMpStart) btnMpStart.addEventListener('click', () => {
            this.startMultiplayer();
        });
        const btnMpBack = document.getElementById('btn-mp-back');
        if (btnMpBack) btnMpBack.addEventListener('click', () => {
            const mp = document.getElementById('screen-multiplayer');
            if (mp) mp.classList.add('hidden');
        });

        const btnHowTo = document.getElementById('btn-howto');
        if (btnHowTo) btnHowTo.addEventListener('click', () => {
            const menu = document.getElementById('screen-menu'); if (menu) menu.classList.add('hidden');
            const how = document.getElementById('screen-howto'); if (how) how.classList.remove('hidden');
        });
        const btnHowToBack = document.getElementById('btn-howto-back');
        if (btnHowToBack) btnHowToBack.addEventListener('click', () => {
            const how = document.getElementById('screen-howto'); if (how) how.classList.add('hidden');
            const menu = document.getElementById('screen-menu'); if (menu) menu.classList.remove('hidden');
        });

        const btnHiscores = document.getElementById('btn-hiscores');
        if (btnHiscores) btnHiscores.addEventListener('click', () => {
            const menu = document.getElementById('screen-menu'); if (menu) menu.classList.add('hidden');
            const hs = document.getElementById('screen-hiscores'); if (hs) hs.classList.remove('hidden');
        });
        const btnHsBack = document.getElementById('btn-hs-back');
        if (btnHsBack) btnHsBack.addEventListener('click', () => {
            const hs = document.getElementById('screen-hiscores'); if (hs) hs.classList.add('hidden');
            const menu = document.getElementById('screen-menu'); if (menu) menu.classList.remove('hidden');
        });

        const btnSettings = document.getElementById('btn-settings');
        if (btnSettings) btnSettings.addEventListener('click', () => {
            const menu = document.getElementById('screen-menu'); if (menu) menu.classList.add('hidden');
            const s = document.getElementById('screen-settings'); if (s) s.classList.remove('hidden');
        });
        const btnSetBack = document.getElementById('btn-set-back');
        if (btnSetBack) btnSetBack.addEventListener('click', () => {
            const s = document.getElementById('screen-settings'); if (s) s.classList.add('hidden');
            const menu = document.getElementById('screen-menu'); if (menu) menu.classList.remove('hidden');
        });

        const btnMusic = document.getElementById('btn-music');
        if (btnMusic) btnMusic.addEventListener('click', () => {
            this.music.toggle();
            if (btnMusic) {
                btnMusic.classList.toggle('active');
                btnMusic.innerText = this.music.isPlaying ? 'ON' : 'OFF';
            }
        });

        const btnGesture = document.getElementById('btn-gesture');
        if (btnGesture) btnGesture.addEventListener('click', () => {
            if (this.gesture.enabled) {
                this.gesture.deactivate();
                if (btnGesture) { btnGesture.classList.remove('active'); btnGesture.innerText = 'OFF'; }
            } else {
                this.gesture.activate().then(ok => {
                    if (ok && btnGesture) { btnGesture.classList.add('active'); btnGesture.innerText = 'ON'; }
                });
            }
        });

        const btnNextLevel = document.getElementById('btn-next-level');
        if (btnNextLevel) btnNextLevel.addEventListener('click', () => {
            this._nextLevel();
        });

        const btnLsBack = document.getElementById('btn-ls-back');
        if (btnLsBack) btnLsBack.addEventListener('click', () => {
            const ls = document.getElementById('screen-level-select'); if (ls) ls.classList.add('hidden');
            const menu = document.getElementById('screen-menu'); if (menu) menu.classList.remove('hidden');
        });

        const btnEditorBack = document.getElementById('btn-editor-back');
        if (btnEditorBack) btnEditorBack.addEventListener('click', () => {
            this.closeEditor();
        });

        const btnShare = document.getElementById('btn-share');
        if (btnShare) btnShare.addEventListener('click', () => {
            this.share.shareScore();
        });

        const btnAR = document.getElementById('btn-ar');
        if (btnAR) btnAR.addEventListener('click', () => {
            if (this.ar.isSupported()) {
                this.ar.setKaijuRef(this.kaiju);
                this.ar.activate();
                if (btnAR) { btnAR.classList.toggle('active'); }
            }
        });

        const btnPush = document.getElementById('btn-push');
        if (btnPush) btnPush.addEventListener('click', async () => {
            const ok = await this.push.requestPermission();
            if (ok) {
                await this.push.subscribe();
                if (btnPush) { btnPush.classList.add('active'); btnPush.innerText = 'ON'; }
            }
        });

        const btnWebrtc = document.getElementById('btn-webrtc');
        if (btnWebrtc) btnWebrtc.addEventListener('click', async () => {
            if (!this.webrtc.isReady) {
                const ok = await this.webrtc.init();
                if (ok) {
                    this.webrtc.onRemoteTrack = (stream) => {
                        if (window.gameAudio) window.gameAudio.playClick();
                    };
                    if (btnWebrtc) { btnWebrtc.classList.add('active'); btnWebrtc.innerText = '🎤 ON'; }
                }
            } else {
                this.webrtc.toggleMute();
                if (btnWebrtc) {
                    btnWebrtc.innerText = this.webrtc.isMuted() ? '🎤 OFF' : '🎤 ON';
                }
            }
        });

        const btnBadges = document.getElementById('btn-badges');
        if (btnBadges) btnBadges.addEventListener('click', () => {
            this._showBadgesScreen();
        });

        const btnBadgesBack = document.getElementById('btn-badges-back');
        if (btnBadgesBack) btnBadgesBack.addEventListener('click', () => {
            const bs = document.getElementById('screen-badges'); if (bs) bs.classList.add('hidden');
            const menu = document.getElementById('screen-menu'); if (menu) menu.classList.remove('hidden');
        });

        const btnIoT = document.getElementById('btn-iot');
        if (btnIoT) btnIoT.addEventListener('click', () => {
            if (!this.iot.isConnected()) {
                this.iot.connect('wss://lazykaiju.example.com/ws');
                if (btnIoT) { btnIoT.classList.add('active'); btnIoT.innerText = 'IoT ON'; }
            }
        });

        if (this.plugins.isRegistered()) {
            this.plugins.registerHook('pluginLoaded', (p) => {
                console.log(`[Plugin] ${p.name} v${p.version} loaded`);
            });
        }

        this._onTrashCleared = (e) => {
            this.score++;
            const tc = document.getElementById('trash-count'); if (tc) tc.innerText = this.score;
            if (window.gameAudio && typeof window.gameAudio.playClick === 'function') window.gameAudio.playClick();
            this.modKarma(2);
            if (navigator.vibrate) navigator.vibrate(20);
            if (this.analytics) this.analytics.track('trash_cleared', { score: this.score });
        };
        this._onKaijuPopup = (e) => {
            const hud = document.getElementById('hud-center');
            if (hud) {
                hud.innerText = e.detail.text;
                hud.style.opacity = 1;
                setTimeout(() => { if (hud) hud.style.opacity = 0; }, 2000);
            }
        };
        window.addEventListener('trashCleared', this._onTrashCleared);
        window.addEventListener('kaijuPopup', this._onKaijuPopup);
    }

    _handleDeepLink() {
        if (!this.share) return;
        const link = this.share.handleDeepLink();
        if (link) {
            if (link.type === 'level') {
                this.loadCustomLevel(link.data);
            } else if (link.type === 'challenge') {
                this.startSinglePlayer();
                if (this.analytics) this.analytics.track('challenge_accepted', { score: link.data.score });
            }
        }
    }

    _initGameRefs() {
        if (this.plugins.isRegistered()) return;
        this.plugins._registerGameRefs(
            () => this.gameState,
            () => this.score,
            (v) => { this.score = Math.max(0, v); },
            (text) => { window.dispatchEvent(new CustomEvent('kaijuPopup', {detail: {text}})); }
        );
    }

    openEditor() {
        if (!this.levelEditor) {
            this.levelEditor = new LevelEditor(this.sceneMgr, this.city, (config) => {
                this.loadCustomLevel(config);
            });
        }
        const menu = document.getElementById('screen-menu'); if (menu) menu.classList.add('hidden');
        const editor = document.getElementById('screen-editor'); if (editor) editor.classList.remove('hidden');
        this.levelEditor.activate();
    }

    closeEditor() {
        if (this.levelEditor) this.levelEditor.deactivate();
        const editor = document.getElementById('screen-editor'); if (editor) editor.classList.add('hidden');
        const menu = document.getElementById('screen-menu'); if (menu) menu.classList.remove('hidden');
    }

    startSinglePlayer(config = null) {
        if (!config) this.currentLevel = 1;
        const menu = document.getElementById('screen-menu'); if (menu) menu.classList.add('hidden');
        const hud = document.getElementById('hud'); if (hud) hud.classList.remove('hidden');
        const mobile = document.getElementById('mobile-controls'); if (mobile) mobile.classList.remove('hidden');
        this.gameMode = 'single';
        this.gameState = 'playing';
        this.sessionStart = Date.now();

        const levelConfig = config || this.procedural.generateConfig(this.currentLevel);
        const finalConfig = this.procedural.adjustForPerformance(levelConfig, this.isMobile);

        this.city.generate(finalConfig);
        this.resetStats();
        if (window.gameAudio && typeof window.gameAudio.resume === 'function') window.gameAudio.resume();
        if (this.trashMgr && typeof this.trashMgr.initialSpawn === 'function') this.trashMgr.initialSpawn(finalConfig.trashCount);

        this._spawnNPCs(finalConfig.difficulty);
        this.music.setLevel(this.currentLevel);
        this.music.start();
        this.music.setState('calm');

        if (this.gesture && !this.isMobile) {
            this.gesture.onSlam = () => { if (this.gameState === 'playing') this.performSweep(); };
            this.gesture.onYawn = () => { this.kaiju.yawn(); };
            this.gesture.activate();
        }

        this.push.sendDailyChallenge(this.currentLevel);

        if (this.analytics) this.analytics.track('game_start', { mode: 'single', level: this.currentLevel });
    }

    startMultiplayer() {
        const p1Name = document.getElementById('mp-p1-name')?.value || 'Player 1';
        const p2Name = document.getElementById('mp-p2-name')?.value || 'Player 2';

        this.multiplayer.activate([p1Name, p2Name]);
        this.gameMode = 'multi';

        const menu = document.getElementById('screen-menu'); if (menu) menu.classList.add('hidden');
        const mp = document.getElementById('screen-multiplayer'); if (mp) mp.classList.add('hidden');
        const hud = document.getElementById('hud'); if (hud) hud.classList.remove('hidden');
        const mobile = document.getElementById('mobile-controls'); if (mobile) mobile.classList.remove('hidden');

        this.gameState = 'playing';
        this.resetStats();
        if (window.gameAudio && typeof window.gameAudio.resume === 'function') window.gameAudio.resume();
        if (this.trashMgr && typeof this.trashMgr.initialSpawn === 'function') this.trashMgr.initialSpawn(80);

        this._spawnNPCs(3);
        this.music.start();
        this.music.setState('tense');

        if (this.analytics) this.analytics.track('game_start', { mode: 'multi', p1: p1Name, p2: p2Name });
    }

    _onMultiplayerEnd(winner, players) {
        this.gameState = 'gameover';
        this.music.setState('gameover');
        this.music.stop();
        this._cleanupNPCs();
        if (this.analytics) {
            this.analytics.track('game_over', {
                mode: 'multi',
                winner: winner.name,
                scores: players.map(p => ({ name: p.name, score: p.score }))
            });
            this.analytics.flush(true);
        }
    }

    loadCustomLevel(config) {
        this.closeEditor();
        this.city.generate(config);
        this.resetStats();
        this.gameState = 'playing';
        if (window.gameAudio && typeof window.gameAudio.resume === 'function') window.gameAudio.resume();
        if (this.trashMgr && typeof this.trashMgr.initialSpawn === 'function') {
            this.trashMgr.initialSpawn(config.trashCount || 40);
        }
        this.currentLevel = Math.max(1, config.difficulty || 1);
        this._spawnNPCs(Math.min(3, this.currentLevel));
        this.music.setLevel(this.currentLevel);
        this.music.start();
        this.music.setState('calm');
        if (this.analytics) this.analytics.track('custom_level_loaded', { buildings: (config.customBuildings || []).length });
    }

    startPlaying(config) {
        const menu = document.getElementById('screen-menu'); if (menu) menu.classList.add('hidden');
        const hud = document.getElementById('hud'); if (hud) hud.classList.remove('hidden');
        const mobile = document.getElementById('mobile-controls'); if (mobile) mobile.classList.remove('hidden');
        this.gameState = 'playing';
        if (window.gameAudio && typeof window.gameAudio.resume === 'function') window.gameAudio.resume();
        if (config && config.trashCount && this.trashMgr && typeof this.trashMgr.initialSpawn === 'function') {
            this.trashMgr.initialSpawn(config.trashCount);
        }
    }

    resetStats() {
        this.destroy();
        this.score = 0;
        this.karma = 100;
        this.stamina = 100;
        this.isSweeping = false;
        this.sweepCooldown = 0;
        this._sweepTimeout = null;
        this._cleanupNPCs();
        const trashCountEl = document.getElementById('trash-count');
        if (trashCountEl) trashCountEl.innerText = "0";
        this.updateKarmaUI();
    }

    updateKarmaUI() {
        const barFill = document.querySelector('.bar-fill');
        if (barFill) {
            barFill.style.width = `${this.karma}%`;
            if(this.karma > 60) barFill.style.background = 'var(--eco-green)';
            else if(this.karma > 30) barFill.style.background = 'var(--karma-gold)';
            else barFill.style.background = 'var(--danger-red)';
        }

        if(this.karma <= 0) {
            this.destroy();
            this.gameState = 'gameover';
            this.music.setState('gameover');
            this.music.stop();
            this._cleanupNPCs();
            this._checkAchievementsOnGameOver();
            if (this.analytics) {
                this.analytics.track('game_over', { mode: this.gameMode, score: this.score, level: this.currentLevel });
                this.analytics.flush(true);
            }
            const go = document.getElementById('game-over-screen'); if (go) go.classList.remove('hidden');
        }
    }

    _checkAchievementsOnGameOver() {
        const duration = (Date.now() - this.sessionStart) / 1000;
        const stats = {
            gamesPlayed: 1,
            trashCleared: this.score,
            bestScore: Math.max(this.score, parseInt(localStorage.getItem('lazykaiju_bestscore') || '0')),
            finishedKarmaMax: this.karma >= 100,
            sessionDuration: Math.round(duration),
            level: this.currentLevel,
            playedAtNight: (() => { const h = new Date().getHours(); return h >= 22 || h < 5; })(),
            mobileGames: this.isMobile ? 1 : 0,
        };
        try {
            const prev = parseInt(localStorage.getItem('lazykaiju_bestscore') || '0');
            if (this.score > prev) localStorage.setItem('lazykaiju_bestscore', String(this.score));
        } catch {}

        this.achievements.updateSession(stats);
        this._showBadgePopup();
    }

    _showBadgePopup() {
        const progress = this.achievements.getProgress();
        if (progress.totalBadges > 0) {
            window.dispatchEvent(new CustomEvent('kaijuPopup', {
                detail: { text: `🏆 ${progress.totalBadges} badge! Lv.${progress.level}` }
            }));
        }
    }

    _showBadgesScreen() {
        const progress = this.achievements.getProgress();
        const unlocked = this.achievements.unlocked;
        const xa = document.getElementById('xp-value');
        const xn = document.getElementById('xp-next');
        const lv = document.getElementById('level-value');
        if (xa) xa.innerText = progress.xp;
        if (xn) xn.innerText = progress.nextLevelXP;
        if (lv) lv.innerText = progress.level;

        const grid = document.getElementById('badges-grid');
        if (grid) {
            grid.innerHTML = AchievementSystem.ACHIEVEMENTS.map(ach => {
                const has = unlocked.includes(ach.id);
                return `<div style="background:var(--bg-card);border:1px solid ${has?'var(--eco-green)':'var(--text-muted)'};border-radius:6px;padding:8px;text-align:center;opacity:${has?1:0.4}">
                    <div style="font-size:1.5rem">${ach.icon}</div>
                    <div style="font-size:0.8rem;color:var(--text-primary);margin-top:4px">${has?ach.name:'???'}</div>
                    <div style="font-size:0.7rem;color:var(--text-muted)">${ach.desc}</div>
                </div>`;
            }).join('');
        }

        const menu = document.getElementById('screen-menu'); if (menu) menu.classList.add('hidden');
        const bs = document.getElementById('screen-badges'); if (bs) bs.classList.remove('hidden');
    }

    modKarma(amount) {
        if (this.gameMode === 'multi') {
            this.multiplayer.addKarmaPenalty(0, amount);
        }
        this.karma = MathUtils.clamp(this.karma + amount, 0, 100);
        this.updateKarmaUI();

        if(amount < 0) {
            const flash = document.getElementById('karma-flash');
            if (flash) {
                flash.style.animation = 'none';
                void flash.offsetWidth;
                flash.style.animation = 'flashFade 0.5s ease-out';
            }
            if (this.karma < 40) {
                this.music.setState('danger');
            } else {
                this.music.setState('tense');
            }
        } else if (this.karma > 60) {
            this.music.setState('calm');
        }
    }

    _spawnNPCs(difficulty) {
        this._cleanupNPCs();
        const count = Math.min(Math.floor(difficulty / 2) + 1, 5);
        for (let i = 0; i < count; i++) {
            const angle = (i / count) * Math.PI * 2;
            const dist = 30 + Math.random() * 20;
            const startPos = new THREE.Vector3(
                Math.cos(angle) * dist,
                0,
                Math.sin(angle) * dist
            );
            const npc = new ActivistNPC(this.sceneMgr.scene, startPos);
            this.npcs.push(npc);
        }
    }

    _updateNPCs(safeDt) {
        const kaijuPos = this.kaiju.getPosition();
        for (let i = this.npcs.length - 1; i >= 0; i--) {
            const npc = this.npcs[i];
            const result = npc.update(safeDt, kaijuPos, this.sceneMgr.scene.children, this.city.buildings);
            if (result.caught) {
                this.modKarma(-8);
                if (navigator.vibrate) navigator.vibrate([100, 50, 100]);
                npc.dispose();
                this.npcs.splice(i, 1);
                this._spawnNPCs(Math.min(this.currentLevel, 5));
                window.dispatchEvent(new CustomEvent('kaijuPopup', {detail: {text: "⚠️ Activist caught you!"}}));
            }
        }
    }

    _cleanupNPCs() {
        for (const npc of this.npcs) {
            npc.dispose();
        }
        this.npcs = [];
    }

    _nextLevel() {
        this.currentLevel++;
        this.music.stop();
        this._cleanupNPCs();
        this.startSinglePlayer();
    }

    destroy() {
        if (this._sweepTimeout) { clearTimeout(this._sweepTimeout); this._sweepTimeout = null; }
        window.removeEventListener('trashCleared', this._onTrashCleared);
        window.removeEventListener('kaijuPopup', this._onKaijuPopup);
        this._onTrashCleared = null;
        this._onKaijuPopup = null;
        this._cleanupNPCs();
        if (this.gesture) this.gesture.deactivate();
        if (this.music) this.music.stop();
        if (this.ar) this.ar.deactivate();
        if (this.webrtc) this.webrtc.dispose();
    }

    handleInput() {
        if (this.gameMode === 'multi') {
            this._handleMultiplayerInput();
            return;
        }
        this.inputVec.set(0, 0, 0);

        if(this.keys.W) this.inputVec.z -= 1;
        if(this.keys.S) this.inputVec.z += 1;
        if(this.keys.A) this.inputVec.x -= 1;
        if(this.keys.D) this.inputVec.x += 1;

        if(this.inputVec.lengthSq() > 0) this.inputVec.normalize();

        if(this.keys.SPACE && this.sweepCooldown <= 0 && this.stamina >= 15) {
            this.performSweep();
        }
    }

    _handleMultiplayerInput() {
        const p1Input = {
            W: this.keys.W || false,
            A: this.keys.A || false,
            S: this.keys.S || false,
            D: this.keys.D || false,
            SPACE: this.keys.SPACE || false
        };
        const p2Input = {
            W: this.keys.P2_I || false,
            A: this.keys.P2_J || false,
            S: this.keys.P2_K || false,
            D: this.keys.P2_L || false,
            SPACE: this.keys.P2_SPACE || false
        };

        if (this.kaiju && typeof this.kaiju.update === 'function') {
            this.kaiju.update(0.016, p1Input);
        }
    }

    performSweep() {
        this.sweepCooldown = 1.5;
        this.stamina -= 15;
        this.isSweeping = true;

        const cooldownInner = document.querySelector('.cooldown-inner');
        if (cooldownInner) cooldownInner.style.transform = 'scale(1)';
        if (this._sweepTimeout) clearTimeout(this._sweepTimeout);
        this._sweepTimeout = setTimeout(() => {
            if (cooldownInner) cooldownInner.style.transform = 'scale(0)';
            this.isSweeping = false;
            this._sweepTimeout = null;
        }, 300);

        let fwd = new THREE.Vector3(0,0,1).applyAxisAngle(new THREE.Vector3(0,1,0), this.kaiju.targetYaw);
        let right = new THREE.Vector3(fwd.z, 0, -fwd.x).multiplyScalar(50);
        this.tail.applySweepImpulse(right);

        window.gameAudio.playSweep();
        if (navigator.vibrate) navigator.vibrate([30, 20, 60]);

        this.music.setState('tense');
    }

    checkDestruction() {
        const spheres = this.tail.getCollisionSpheres();
        let blocks = this.city.buildings;

        for(let i = blocks.length - 1; i >= 0; i--) {
            let b = blocks[i];
            if(!b || !b.userData || b.userData.destroyed) continue;

            let pos = b.position;
            for(let s of spheres) {
                let dx = pos.x - s.x;
                let dz = pos.z - s.z;
                if(dx*dx + dz*dz < (s.r + 5)*(s.r + 5)) {
                    b.userData.destroyed = true;
                    this.sceneMgr.scene.remove(b);
                    blocks.splice(i, 1);

                    window.gameAudio.playCrunch();

                    if(b.userData.isEco) {
                        this.modKarma(-20);
                        if (navigator.vibrate) navigator.vibrate([50, 30, 50]);
                        window.dispatchEvent(new CustomEvent('kaijuPopup', {detail: {text: "-20 KARMA! Eco-Building Destroyed!"}}));
                    } else {
                        this.modKarma(-5);
                        if (navigator.vibrate) navigator.vibrate(15);
                    }

                    this.trashMgr.spawnExplosion(pos, 0x555555, 10);
                    if (this.analytics) this.analytics.track('building_destroyed', { type: b.type, isEco: b.userData.isEco });
                    break;
                }
            }
        }
    }

    loop(time) {
        requestAnimationFrame(t => this.loop(t));

        const dt = (time - this.lastTime) / 1000;
        this.lastTime = time;
        const safeDt = Math.min(dt, 0.1);

        this.kaiju.update(safeDt, this.inputVec);
        this.tail.update(safeDt);

        if (this.gameState === 'playing') {
            this.stamina = MathUtils.clamp(this.stamina + 5 * safeDt, 0, this.maxStamina);

            if(this.sweepCooldown > 0) {
                this.sweepCooldown -= safeDt;
            }

            this.handleInput();
            if (this.trashMgr && typeof this.trashMgr.update === 'function') {
                this.trashMgr.update(safeDt, this.tail.getCollisionSpheres());
            }

            if(this.isSweeping) {
                this.checkDestruction();
            }

            this._updateNPCs(safeDt);
            this.music.update(safeDt, this.gameState === 'playing' ? (this.karma < 40 ? 'danger' : this.karma < 60 ? 'tense' : 'calm') : this.gameState);

            const now = Date.now();
            if (this.gameState === 'playing' && now - this._lastAmbientCheck > 5000) {
                this._lastAmbientCheck = now;
                const hour = new Date().getHours();
                const motion = this.kaiju.isMoving;
                this.iot.checkAmbientTriggers(hour, motion, 100);
            }

            if (this.multiplayer && this.multiplayer.isActive) {
                const state = this.multiplayer.getGameState();
                state.players.forEach((p, i) => {
                    if (p.score > 0) {
                        this.multiplayer.addScore(i, 0);
                    }
                });
            }
        }

        const kPos = this.kaiju.getPosition();
        let fwd = new THREE.Vector3(0,0,1).applyAxisAngle(new THREE.Vector3(0,1,0), this.kaiju.targetYaw);
        const offset = new THREE.Vector3(
           -fwd.x * 20,
           15,
           -fwd.z * 20
        );

        if(!this.kaiju.isMoving) {
            offset.set(0, 20, -25);
        }

        const targetCamPos = new THREE.Vector3(
            kPos.x + offset.x,
            kPos.y + offset.y,
            kPos.z + offset.z
        );

        this.sceneMgr.camera.position.lerp(targetCamPos, 3 * safeDt);
        this.sceneMgr.camera.lookAt(kPos.x, kPos.y + 5, kPos.z);

        this.sceneMgr.render();
    }
}

window.onload = () => {
    const game = new GameController();
};
