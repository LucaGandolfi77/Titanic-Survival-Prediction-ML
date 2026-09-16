import { Renderer } from './renderer.js';
import { Controls } from './controls.js';
import { Match, MatchState } from './match.js';
import { UI } from './ui.js';
import { audio } from './audio.js';
import { Vector2 } from './utils.js';
import { storage } from './services/storage.js';
import { GamificationEngine } from './game/gamification.js';
import { TournamentManager, TOURNAMENT_STATUS } from './game/tournament.js';
import { AIBrain, AI_BACKEND } from './game/ai-brain.js';
import { CoachAI } from './game/coach.js';
import { CommentaryEngine } from './game/commentary.js';
import { ArenaManager, SURFACE_TYPES, ENVIRONMENT_TYPES } from './game/arena.js';

const GAME = {
  renderer: null,
  controls: null,
  match: null,
  lastTime: 0,
  settings: {
    difficulty: 'medium',
    duration: 3,
    teamName: 'HOME',
    teamColor: '#3b82f6',
    sound: true,
    autoSwitch: true
  },
  animationFrameId: null,
  gamification: null,
  tournament: null,
  aiBrain: null,
  coach: null,
  commentary: null,
  arena: null,
  tournamentMode: false,
  tournamentState: null,
  tournamentMatch: null,
  controlledPlayer: null,
  shootChargeTime: 0,
  lastShootState: false
};

function loop(timestamp) {
  if (!GAME.lastTime) GAME.lastTime = timestamp;
  const dt = Math.min((timestamp - GAME.lastTime) / 1000, 0.05);
  GAME.lastTime = timestamp;

  update(dt);
  render();

  GAME.animationFrameId = requestAnimationFrame(loop);
}

function update(dt) {
  if (!GAME.match) return;

  const m = GAME.match;
  const input = GAME.controls;

  if (m.state === MatchState.PLAYING) {
    let bestPlayer = null;
    let minDist = Infinity;
    const myPlayers = m.players.filter(p => p.team === 0);

    if (input.buttons.switch && !input.wasSwitchPressed) {
      input.wasSwitchPressed = true;
      let idx = myPlayers.indexOf(GAME.controlledPlayer);
      if (idx >= 0) {
        idx = (idx + 1) % myPlayers.length;
        GAME.controlledPlayer = myPlayers[idx];
      }
    } else if (!input.buttons.switch) {
      input.wasSwitchPressed = false;
      if (document.getElementById('btn-autoswitch')?.classList.contains('active')) {
        myPlayers.forEach(p => {
          const d = p.pos.distanceTo(m.ball.pos);
          if (d < minDist) { minDist = d; bestPlayer = p; }
        });
        if (bestPlayer && bestPlayer !== GAME.controlledPlayer) {
          const currentDist = GAME.controlledPlayer?.pos.distanceTo(m.ball.pos) || Infinity;
          if (minDist < currentDist - 40) { GAME.controlledPlayer = bestPlayer; }
        }
        if (!GAME.controlledPlayer) GAME.controlledPlayer = bestPlayer;
      }
    }

    if (GAME.controlledPlayer) {
      GAME.controlledPlayer.isControlled = true;
    }

    if (input.buttons.pass) {
      if (GAME.controlledPlayer?.hasBall && !input.wasPassPressed) {
        let target = null;
        let bestScore = -Infinity;
        const joyHeading = input.getOutput().mag() > 0.1 ? input.getOutput().angle() : GAME.controlledPlayer?.facing;

        myPlayers.forEach(p => {
          if (p === GAME.controlledPlayer) return;
          const toP = p.pos.clone().sub(GAME.controlledPlayer.pos);
          const dist = toP.mag();
          const angle = toP.angle();
          const angleDiff = Math.abs(angle - (joyHeading || 0));
          if (angleDiff < 1.0) {
            const score = 1000 - dist;
            if (score > bestScore) { bestScore = score; target = p; }
          }
        });

        if (target) { m.ball.pass(target.pos); audio.playPass(); GAME.commentary?.commentPass(); }
        else {
          const shootDir = new Vector2(Math.cos(joyHeading || 0), Math.sin(joyHeading || 0));
          m.ball.vel = shootDir.multiplyScalar(400);
          m.ball.owner = null;
          audio.playKick(0.5);
        }
        input.wasPassPressed = true;
      }
    } else { input.wasPassPressed = false; }

    if (input.buttons.shoot) {
      if (!GAME.lastShootState && GAME.shootChargeTime === 0) {
        GAME.shootChargeTime = performance.now();
      }
    } else {
      if (GAME.lastShootState && GAME.shootChargeTime > 0 && GAME.controlledPlayer?.hasBall) {
        const power = Math.min((performance.now() - GAME.shootChargeTime) / 800, 1.0);
        let dir = input.getOutput().mag() > 0.1 ? input.getOutput().clone() : new Vector2(1, 0);
        if (input.getOutput().mag() < 0.1) {
          dir = new Vector2(800 - GAME.controlledPlayer.pos.x, 250 - GAME.controlledPlayer.pos.y).normalize();
        }
        m.ball.shoot(dir, power);
        audio.playKick(power);
      }
      GAME.shootChargeTime = 0;
    }
    GAME.lastShootState = input.buttons.shoot;

    if (input.buttons.tackle && !input.wasTacklePressed) {
      GAME.controlledPlayer?.startTackle();
      audio.playTackle();
      if (navigator.vibrate) navigator.vibrate(30);
      GAME.commentary?.commentTackle();
      input.wasTacklePressed = true;
    } else if (!input.buttons.tackle) { input.wasTacklePressed = false; }

    if (m.ai && m.ai.update) {
      m.ai.update(dt, { ball: m.ball, players: m.players, myTeamIndex: 0 });
    }

    if (m.aiBrain && m.aiBrain.initialized) {
      const predictions = m.aiBrain._heuristicDecide({ ball: m.ball, players: m.players, myTeamIndex: 0 });
      predictions.forEach(pred => {
        if (pred.player && pred.action === 'tackle' && Math.random() < pred.confidence) {
          if (!pred.player.isTackling) {
            pred.player.startTackle();
          }
        }
      });
    }
  }

  m.update(dt, {
    joyVec: input.getOutput(),
    buttons: input.buttons,
    controlledPlayer: GAME.controlledPlayer,
    isCharging: input.isCharging
  });

  UI.updateTime(m.duration - m.currentTime, m.half);
  UI.updateScore(m.scores[0], m.scores[1]);
}

function render() {
  if (!GAME.renderer || !GAME.match) return;
  const arenaColors = GAME.arena ? GAME.arena.getFieldColors() : null;
  GAME.renderer.render({
    players: GAME.match.players,
    ball: GAME.match.ball,
    matchTime: GAME.match.currentTime,
    scores: GAME.match.scores,
    isGoal: GAME.match.state === MatchState.GOAL_SCORED,
    controlledPlayer: GAME.controlledPlayer,
    arenaColors: arenaColors
  });
}

async function initApp() {
  await storage.ready();

  GAME.aiBrain = new AIBrain(GAME.settings.difficulty);
  await GAME.aiBrain._init();

  GAME.coach = new CoachAI();

  GAME.commentary = new CommentaryEngine();
  GAME.commentary.init();

  GAME.arena = new ArenaManager(storage);
  await GAME.arena.init();

  GAME.gamification = new GamificationEngine(storage);
  await GAME.gamification.init();

  GAME.tournament = new TournamentManager(storage);
  await GAME.tournament.init();

  const prog = GAME.gamification.progression;
  const levelEl = document.getElementById('menu-player-level');
  if (levelEl) {
    const info = GAME.gamification.getLevelInfo();
    levelEl.textContent = `Level ${info.level} • ${prog.totalMatches} matches • ${prog.totalGoals} goals`;
  }

  const arenaName = GAME.arena.getCurrentArena()?.name || 'Stadio Standard';
  if (levelEl) {
    levelEl.textContent += ` • ${arenaName}`;
  }

  registerServiceWorker();
}

function registerServiceWorker() {
  if ('serviceWorker' in navigator) {
    navigator.serviceWorker.register('./sw.js').then(() => {
      console.log('SW registered');
    }).catch((err) => {
      console.warn('SW registration failed:', err);
    });
  }
}

window.onload = () => {
  initApp().then(() => {
    const canvas = document.getElementById('game-canvas');
    GAME.renderer = new Renderer(canvas);
    GAME.controls = new Controls();

    UI.setupListeners({
      onSelectDifficulty: (diff) => {
        GAME.settings.difficulty = diff;
        storage.setSetting('difficulty', diff);
        if (GAME.aiBrain) GAME.aiBrain.difficulty = diff;
      },
      onSelectColor: (col) => {
        GAME.settings.teamColor = col;
        storage.setSetting('teamColor', col);
      },
      onStartMatch: (name) => {
        GAME.settings.teamName = name;
        storage.setSetting('teamName', name);
        document.getElementById('home-name').textContent = name;
        audio.init();
        if (navigator.vibrate) navigator.vibrate(50);
        GAME.commentary?.commentKickoff();
        startMatch();
      },
      onResume: () => {
        UI.showHUD();
        UI.showScreen(null);
        if (GAME.match) GAME.match.state = MatchState.PLAYING;
      },
      onRestart: () => { startMatch(); },
      onQuit: quitMatch,
      onNextHalf: nextHalf,
      onSetDuration: (min) => {
        GAME.settings.duration = min;
        storage.setSetting('duration', min);
      }
    });

    setupMenuListeners();
    setupProfileListeners();
    setupTournamentListeners();
    setupCoachListeners();
    setupArenaListeners();
    setupCommentaryListeners();

    restoreSettings();
    requestAnimationFrame(loop);
  });
};

function setupMenuListeners() {
  document.getElementById('btn-play').addEventListener('click', () => { UI.showScreen('difficulty'); });
  document.getElementById('btn-tournament').addEventListener('click', () => {
    GAME.tournamentMode = true;
    showTournamentSetup();
  });
  document.getElementById('btn-settings-menu').addEventListener('click', () => { UI.showScreen('settings'); });
  document.getElementById('btn-records').addEventListener('click', () => {
    UI.renderRecords();
    UI.showScreen('records');
  });
  document.getElementById('btn-records-back')?.addEventListener('click', () => { UI.showScreen('menu'); });
  document.getElementById('btn-autoswitch').addEventListener('click', function() {
    const on = !this.classList.contains('active');
    this.classList.toggle('active', on);
    this.textContent = on ? '🔄 ON' : '🔄 OFF';
    storage.setSetting('autoSwitch', on);
  });
  document.getElementById('btn-gamification').addEventListener('click', () => { showProfile(); });
  document.getElementById('btn-arena').addEventListener('click', () => { showArena(); });
  document.getElementById('btn-coach').addEventListener('click', () => { showCoachAnalysis(GAME.lastAnalysis); });
  document.getElementById('btn-sound').addEventListener('click', function() {
    const on = !this.classList.contains('active');
    this.classList.toggle('active', on);
    this.textContent = on ? 'ON' : 'OFF';
    audio.setEnabled(on);
    storage.setSetting('sound', on);
  });
  document.getElementById('btn-vibration').addEventListener('click', function() {
    const on = !this.classList.contains('active');
    this.classList.toggle('active', on);
    this.textContent = on ? 'ON' : 'OFF';
    storage.setSetting('vibration', on);
    if (on && navigator.vibrate) navigator.vibrate(50);
  });
}

function showArena() {
  const arena = GAME.arena?.getCurrentArena();
  if (!arena) return;
  document.querySelectorAll('.surface-btn').forEach(btn => {
    btn.classList.toggle('active', btn.getAttribute('data-surface') === arena.surface);
  });
  document.querySelectorAll('.env-btn').forEach(btn => {
    btn.classList.toggle('active', btn.getAttribute('data-env') === arena.environment);
  });
  const preview = document.getElementById('arena-preview');
  if (preview) {
    const surf = SURFACE_TYPES[arena.surface] || SURFACE_TYPES.GRASS;
    const env = ENVIRONMENT_TYPES[arena.environment] || ENVIRONMENT_TYPES.CLEAR;
    preview.innerHTML = `
      <div class="arena-preview" style="background:${surf.color};height:100px;border-radius:10px;margin-top:15px;opacity:${env.ambient}">
        <div style="display:flex;align-items:center;justify-content:center;height:100%;font-size:1.5rem;color:white;text-shadow:1px 1px 3px rgba(0,0,0,0.5)">
          ${surf.name} • ${env.name}
        </div>
      </div>
    `;
  }
  UI.showScreen('arena');
}

function setupProfileListeners() {
  document.getElementById('btn-profile-back')?.addEventListener('click', () => { UI.showScreen('menu'); });
}

function setupTournamentListeners() {
  document.getElementById('btn-tournament-back')?.addEventListener('click', () => { UI.showScreen('menu'); });
  document.getElementById('btn-bracket-back')?.addEventListener('click', () => { UI.showScreen('menu'); });
  document.getElementById('btn-tournament-start')?.addEventListener('click', () => { startTournament(); });
  document.getElementById('btn-tournament-next')?.addEventListener('click', () => { playNextTournamentMatch(); });
}

function setupCoachListeners() {
  document.getElementById('btn-coach-back')?.addEventListener('click', () => { UI.showScreen('menu'); });
  document.getElementById('btn-coach-replay')?.addEventListener('click', () => {
    if (GAME.coach && GAME.lastAnalysis) {
      showCoachAnalysis(GAME.lastAnalysis);
    }
  });
}

function setupArenaListeners() {
  document.getElementById('btn-arena-back')?.addEventListener('click', () => { UI.showScreen('settings'); });
  document.querySelectorAll('.surface-btn').forEach(btn => {
    btn.addEventListener('click', async () => {
      const surface = btn.getAttribute('data-surface');
      if (GAME.arena) {
        const arena = GAME.arena.getCurrentArena();
        arena.surface = surface;
        await GAME.arena.saveCustomArena(arena);
        GAME.arena.currentArena = arena;
        UI.showToast(`Campo: ${SURFACE_TYPES[surface]?.name || surface}`);
      }
    });
  });
  document.querySelectorAll('.env-btn').forEach(btn => {
    btn.addEventListener('click', async () => {
      const env = btn.getAttribute('data-env');
      if (GAME.arena) {
        const arena = GAME.arena.getCurrentArena();
        arena.environment = env;
        await GAME.arena.saveCustomArena(arena);
        GAME.arena.currentArena = arena;
        UI.showToast(`Atmosfera: ${ENVIRONMENT_TYPES[env]?.name || env}`);
      }
    });
  });
}

function setupCommentaryListeners() {
  document.getElementById('btn-commentary-on')?.addEventListener('click', () => {
    GAME.commentary?.unmute();
    document.getElementById('btn-commentary-on').classList.add('active');
    document.getElementById('btn-commentary-off')?.classList.remove('active');
  });
  document.getElementById('btn-commentary-off')?.addEventListener('click', () => {
    GAME.commentary?.mute();
    document.getElementById('btn-commentary-off').classList.add('active');
    document.getElementById('btn-commentary-on')?.classList.remove('active');
  });
}

async function restoreSettings() {
  const diff = await storage.getSetting('difficulty');
  const color = await storage.getSetting('teamColor');
  const name = await storage.getSetting('teamName');
  const dur = await storage.getSetting('duration');
  const autoSwitch = await storage.getSetting('autoSwitch');
  if (diff) {
    GAME.settings.difficulty = diff;
    if (GAME.aiBrain) GAME.aiBrain.difficulty = diff;
  }
  if (color) GAME.settings.teamColor = color;
  if (name) {
    GAME.settings.teamName = name;
    document.getElementById('home-name').textContent = name;
    const input = document.getElementById('team-name-input');
    if (input) input.value = name;
  }
  if (dur) GAME.settings.duration = parseInt(dur);
  if (autoSwitch !== null) {
    GAME.settings.autoSwitch = autoSwitch;
    const btn = document.getElementById('btn-autoswitch');
    if (btn) {
      btn.classList.toggle('active', autoSwitch);
      btn.textContent = autoSwitch ? '🔄 ON' : '🔄 OFF';
    }
  }

  document.querySelectorAll('.diff-card').forEach(card => {
    card.classList.toggle('active', card.getAttribute('data-diff') === GAME.settings.difficulty);
  });
  document.querySelectorAll('.color-swatch').forEach(sw => {
    sw.classList.toggle('active', sw.getAttribute('data-color') === GAME.settings.teamColor);
  });
  document.querySelectorAll('.duration-group .opt-btn').forEach(btn => {
    btn.classList.toggle('active', parseInt(btn.getAttribute('data-min')) === GAME.settings.duration);
  });

  const currentArena = GAME.arena?.getCurrentArena();
  if (currentArena) {
    document.querySelectorAll('.surface-btn').forEach(btn => {
      btn.classList.toggle('active', btn.getAttribute('data-surface') === currentArena.surface);
    });
    document.querySelectorAll('.env-btn').forEach(btn => {
      btn.classList.toggle('active', btn.getAttribute('data-env') === currentArena.environment);
    });
  }
}

function showProfile() {
  const prog = GAME.gamification.progression;
  const stats = GAME.gamification.getStats();
  const badges = GAME.gamification.getActiveBadges();
  const levelInfo = GAME.gamification.getLevelInfo();
  UI.updateProfile(prog, stats, badges, levelInfo);
  UI.showScreen('profile');
}

async function showTournamentSetup() {
  const container = document.getElementById('tournament-players-list');
  if (container) {
    const saved = await GAME.tournament.getAllTournaments();
    if (saved.length > 0) {
      container.innerHTML = saved.map(t =>
        `<div class="stat-row">
          <span>${t.name}</span>
          <button class="menu-btn small" onclick="loadTournament('${t.id}')">Load</button>
        </div>`
      ).join('');
    } else {
      container.innerHTML = '<p style="color:var(--text-muted)">No tournaments saved</p>';
    }
  }
  UI.showScreen('tournament-setup');
}

window.loadTournament = async function(id) {
  const tournament = await GAME.tournament.getTournament(id);
  if (tournament) {
    GAME.tournamentState = tournament;
    showTournamentBracket(tournament);
  }
};

async function startTournament() {
  const name = document.getElementById('tournament-name-input')?.value || 'Tournament';
  const formatEl = document.querySelector('.btn-group[data-format] .opt-btn.active');
  const format = formatEl?.getAttribute('data-format') || 'round_robin';
  const playersEl = document.querySelector('.btn-group[data-players] .opt-btn.active');
  const numPlayers = parseInt(playersEl?.getAttribute('data-players') || '4');

  const tournament = await GAME.tournament.createTournament({ name, format });

  tournament.players.push({ name: GAME.settings.teamName, isHuman: true, totalScore: 0, matchesPlayed: 0, wins: 0 });
  for (let i = 1; i < numPlayers; i++) {
    await GAME.tournament.addPlayer(tournament.id, `Player ${i}`, false);
  }

  if (format === 'round_robin') {
    const players = tournament.players;
    for (let i = 0; i < players.length; i++) {
      for (let j = i + 1; j < players.length; j++) {
        tournament.matches.push({
          homePlayerIdx: i,
          awayPlayerIdx: j,
          homeScore: -1,
          awayScore: -1,
          winner: -1,
          played: false
        });
      }
    }
  }

  await GAME.tournament.saveTournament(tournament);
  GAME.tournamentState = tournament;
  tournament.status = TOURNAMENT_STATUS.IN_PROGRESS;
  await GAME.tournament.saveTournament(tournament);

  showTournamentBracket(tournament);
}

function showTournamentBracket(tournament) {
  const bracketEl = document.getElementById('tournament-bracket');
  const statusEl = document.getElementById('tournament-status');

  if (bracketEl) {
    bracketEl.innerHTML = tournament.matches.map((m, idx) => {
      const home = tournament.players[m.homePlayerIdx]?.name || '?';
      const away = tournament.players[m.awayPlayerIdx]?.name || '?';
      if (m.played) {
        return `<div class="stat-row">
          <span>${home} ${m.homeScore}</span>
          <span>vs</span>
          <span>${away} ${m.awayScore}</span>
        </div>`;
      }
      return `<div class="stat-row" style="opacity:0.5">
        <span>${home}</span>
        <span>vs</span>
        <span>${away}</span>
      </div>`;
    }).join('');
  }

  if (statusEl) {
    const nextMatch = tournament.matches.find(m => !m.played);
    if (nextMatch) {
      const home = tournament.players[nextMatch.homePlayerIdx];
      const away = tournament.players[nextMatch.awayPlayerIdx];
      statusEl.innerHTML = `<p>Next: ${home.name} vs ${away.name}</p>`;
    } else {
      statusEl.innerHTML = '<p>All matches played!</p>';
    }
  }

  const nextBtn = document.getElementById('btn-tournament-next');
  if (nextBtn) {
    nextBtn.style.display = tournament.matches.some(m => !m.played) ? '' : 'none';
  }

  UI.showScreen('tournament-bracket');
}

async function playNextTournamentMatch() {
  if (!GAME.tournamentState) return;
  const tournament = GAME.tournamentState;
  const nextMatch = tournament.matches.find(m => !m.played);
  if (!nextMatch) return;

  document.getElementById('home-name').textContent = tournament.players[nextMatch.homePlayerIdx].name;
  GAME.tournamentMatch = nextMatch;
  startMatch(true);
}

async function endTournamentMatch(homeScore, awayScore) {
  if (!GAME.tournamentMatch) return;
  const t = GAME.tournamentMatch;
  const tournament = GAME.tournamentState;
  if (!tournament) return;

  t.homeScore = homeScore;
  t.awayScore = awayScore;
  t.played = true;
  t.winner = homeScore > awayScore ? t.homePlayerIdx : (awayScore > homeScore ? t.awayPlayerIdx : -1);

  await GAME.tournament.recordMatchResult(tournament.id, tournament.matches.indexOf(t), t.homePlayerIdx, t.awayPlayerIdx, homeScore, awayScore);

  const isTournamentOver = tournament.matches.every(m => m.played);
  if (isTournamentOver) {
    tournament.status = TOURNAMENT_STATUS.FINISHED;
    await GAME.tournament.saveTournament(tournament);
    const winner = await GAME.tournament.getWinner(tournament.id);
    if (winner) {
      UI.showToast(`🏆 ${winner.name} wins the tournament!`);
      if (navigator.vibrate) navigator.vibrate([100, 50, 100, 50, 200]);
      GAME.commentary?.commentFullTime('win');
    }
  }
}

function startMatch(isTournamentMatch = false) {
  GAME.tournamentMode = isTournamentMatch;
  if (!isTournamentMatch) {
    GAME.tournamentMatch = null;
  }

  GAME.match = new Match(
    GAME.settings.difficulty,
    GAME.settings.duration,
    (team) => {
      UI.showEvent('GOAL!');
      if (navigator.vibrate) navigator.vibrate([50, 30, 50]);
      GAME.gamification?.checkBadges({});
      GAME.commentary?.commentGoal(team === 0 ? 'home' : 'away');
    },
    () => {
      UI.showScreen('halftime');
      GAME.commentary?.commentHalfTime();
    },
    () => {
      const home = GAME.match.scores[0];
      const away = GAME.match.scores[1];
      const res = home > away ? 'YOU WIN!' : (home < away ? 'YOU LOSE' : 'DRAW');
      document.getElementById('fulltime-result').textContent = res;
      UI.showScreen('fulltime');

      const won = home > away;
      const result = GAME.gamification.recordMatch({
        goalsScored: home,
        goalsConceded: away,
        won,
        passes: Math.floor(Math.random() * 10) + 5,
        tackles: Math.floor(Math.random() * 8) + 2
      });

      GAME.lastAnalysis = GAME.coach?.analyzeMatch({
        players: GAME.match.players,
        ball: GAME.match.ball,
        scores: GAME.match.scores,
        duration: GAME.match.duration,
        currentTime: GAME.match.currentTime
      });

      if (result.leveledUp) {
        UI.showToast(`⬆️ Level Up! Now Level ${result.newLevel}`);
        if (navigator.vibrate) navigator.vibrate([100, 50, 100, 50, 200]);
      } else if (won) {
        if (navigator.vibrate) navigator.vibrate(100);
      }

      GAME.commentary?.commentFullTime(won ? 'win' : (home < away ? 'lose' : 'draw'));

      document.getElementById('fulltime-xp').innerHTML = `
        <div style="color:var(--goal-yellow);font-family:var(--font-mono);font-size:1.2rem">
          +${result.xpGained} XP (Level ${result.newLevel})
        </div>
      `;

      if (GAME.tournamentMode && GAME.tournamentMatch) {
        endTournamentMatch(home, away);
      }

      const recs = JSON.parse(localStorage.getItem('pf_records') || '[]');
      recs.unshift({ date: new Date().toLocaleDateString(), result: res, score: `${home}-${away}` });
      localStorage.setItem('pf_records', JSON.stringify(recs.slice(0, 10)));
    }
  );

  GAME.controlledPlayer = GAME.match.players[3];
  UI.showHUD();
  UI.showScreen(null);
  UI.showEvent('KICK OFF!', 1000);
  audio.playWhistle();
  GAME.match.state = MatchState.PLAYING;
}

function nextHalf() {
  UI.showHUD();
  UI.showScreen(null);
  GAME.match.state = MatchState.PLAYING;
  GAME.match.half = 2;
  GAME.match.resetPositions();
  audio.playWhistle();
}

function quitMatch() {
  if (GAME.match) {
    GAME.match = null;
  }
  GAME.controlledPlayer = null;
  GAME.tournamentMatch = null;
  UI.showScreen('menu');
  UI.hideHUD();

  if (GAME.gamification) {
    const prog = GAME.gamification.progression;
    const levelEl = document.getElementById('menu-player-level');
    if (levelEl) {
      const info = GAME.gamification.getLevelInfo();
      const arenaName = GAME.arena?.getCurrentArena()?.name || 'Stadio Standard';
      levelEl.textContent = `Level ${info.level} • ${prog.totalMatches} matches • ${prog.totalGoals} goals • ${arenaName}`;
    }
  }
}

async function showCoachAnalysis(analysis) {
  if (!analysis) {
    UI.showToast('Nessuna partita analizzata ancora');
    return;
  }
  const container = document.getElementById('coach-content');
  if (!container) return;

  const ratingColor = analysis.rating >= 7 ? 'var(--btn-pass)' : analysis.rating >= 4 ? 'var(--btn-shoot)' : 'var(--btn-tackle)';

  let html = `
    <div class="coach-rating" style="background:${ratingColor}">
      <span style="font-size:2.5rem">${analysis.rating}</span><span>/10</span>
    </div>
    <h3 style="margin:15px 0;text-align:center">⚽ Analisi Partita</h3>
    <div class="stat-row"><span>Risultato</span><span>${analysis.scores.home} - ${analysis.scores.away}</span></div>
    <div class="stat-row"><span>Durata</span><span>${Math.floor(analysis.playedTime)}s</span></div>
    <div class="stat-row"><span>Velocità Media</span><span>${analysis.stats.avgSpeed.toFixed(1)}</span></div>
  `;

  if (analysis.suggestions.length > 0) {
    html += `<h3 style="margin:15px 0;text-align:center">💡 Suggerimenti</h3>`;
    analysis.suggestions.forEach(s => {
      const priorityIcon = s.priority === 'high' ? '🔴' : s.priority === 'medium' ? '🟡' : '🟢';
      html += `
        <div class="suggestion-item">
          <div class="suggestion-header">${priorityIcon} ${s.icon} ${s.title}</div>
          <div class="suggestion-message">${s.message}</div>
        </div>
      `;
    });
  }

  container.innerHTML = html;
  UI.showScreen('coach');
}
