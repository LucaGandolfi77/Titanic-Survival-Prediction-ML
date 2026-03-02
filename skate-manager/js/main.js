/* ===== Main entry point — game loop, events, routing ===== */
import { GameState, saveGame, loadGame, hasSave, resetState } from './state.js';
import { createSkater, generateStartingSquad, generateMarketSkaters, weeklyStatFluctuation,
         injureSkater, getSquadAvgOverall, getSquadAvgMorale, recalcSkater } from './skaters.js';
import { generateCalendar, enterCompetition, canEnterCompetition,
         getThisWeekCompetition, calculatePlacements, generateRivals } from './competitions.js';
import { processWeeklySponsors, negotiateSponsor, getTotalSponsorIncome, getFameBonus, getWinPointsBonus } from './sponsors.js';
import { promoteToActive, demoteToReserve, releaseSkater, buySkater,
         listForSale, cancelListing, aiMarketActivity, refreshMarket,
         scoutMarket, trainTeam, getWeeklyWages } from './squad.js';
import { MiniGame } from './minigame.js';
import { MusicEngine } from './music.js';
import { randInt, pick, clamp, formatMoneyFull, formatMoney } from './utils.js';
import {
  showToast, showModal, hideModal, confirmModal,
  showSkaterDetail, showSellModal, showBuyModal, showCompEntryModal,
  showScreen, switchTab, updateHeader, renderOverview, renderSquad,
  renderMarket, renderCalendar, renderSponsors, renderStandings,
  renderResults, renderSeasonEnd, refreshAllPanels
} from './ui.js';

// ===== Global instances =====
const miniGame = new MiniGame();
const bgMusic = new MusicEngine();

// ===== Event handlers =====
function handleCardAction(action, skaterId) {
  bgMusic.playClick();
  switch (action) {
    case 'promote': {
      const ok = promoteToActive(skaterId);
      if (ok) {
        showToast('Skater promoted to active squad', 'success');
        refreshUI();
      } else {
        showToast('Active squad full (max 16)', 'warning');
      }
      break;
    }
    case 'demote': {
      const ok = demoteToReserve(skaterId);
      if (ok) {
        showToast('Skater moved to reserves', 'info');
        refreshUI();
      } else {
        showToast('Reserve bench full (max 8)', 'warning');
      }
      break;
    }
    case 'sell': {
      const skater = findSkater(skaterId);
      if (!skater) return;
      showSellModal(skater, (price) => {
        const ok = listForSale(skaterId, price);
        if (ok) {
          showToast(`${skater.name} listed for ${formatMoneyFull(price)}`, 'gold');
          GameState.eventLog.push(`🏷️ Listed ${skater.name} for ${formatMoney(price)}`);
          refreshUI();
        }
      });
      break;
    }
    case 'release': {
      const skater = findSkater(skaterId);
      if (!skater) return;
      confirmModal('Release Skater', `Release ${skater.name}? This cannot be undone.`, () => {
        releaseSkater(skaterId);
        showToast(`${skater.name} released`, 'danger');
        GameState.eventLog.push(`✖ Released ${skater.name}`);
        refreshUI();
      });
      break;
    }
    case 'buy': {
      const skater = findSkater(skaterId);
      if (!skater) return;
      showBuyModal(skater, () => {
        const result = buySkater(skaterId);
        if (result.ok) {
          showToast(result.msg, 'success');
          GameState.eventLog.push(`💰 ${result.msg}`);
          refreshUI();
        } else {
          showToast(result.msg, 'warning');
        }
      });
      break;
    }
    case 'cancel-listing': {
      const ok = cancelListing(skaterId);
      if (ok) {
        showToast('Listing cancelled', 'info');
        refreshUI();
      }
      break;
    }
  }
}

function handleNegotiate(sponsorId) {
  bgMusic.playClick();
  const result = negotiateSponsor(sponsorId);
  if (result.ok) {
    showToast(result.msg, 'gold');
    GameState.eventLog.push(`🤝 ${result.msg}`);
    refreshUI();
  } else {
    showToast(result.msg, 'warning');
  }
}

function findSkater(id) {
  return GameState.activeSquad.find(s => s.id === id) ||
         GameState.reserveBench.find(s => s.id === id) ||
         GameState.marketSkaters.find(s => s.id === id) ||
         GameState.listedSkaters.find(s => s.id === id);
}

function refreshUI() {
  refreshAllPanels(handleCardAction, handleNegotiate);
  autoSave();
}

function autoSave() {
  if (GameState.autosave) saveGame();
}

// ===== Random events (called during advanceWeek) =====
function triggerRandomEvents() {
  const events = [];
  const r = Math.random();

  // 1. Injury (10% chance)
  if (r < 0.10 && GameState.activeSquad.length > 0) {
    const idx = randInt(0, GameState.activeSquad.length - 1);
    const sk = GameState.activeSquad[idx];
    if (sk.injuryWeeks === 0) {
      const weeks = randInt(1, 3);
      injureSkater(sk, weeks);
      events.push(`🏥 ${sk.name} injured for ${weeks} week(s)!`);
    }
  }

  // 2. Morale boost (12% chance)
  if (Math.random() < 0.12 && GameState.activeSquad.length > 0) {
    const idx = randInt(0, GameState.activeSquad.length - 1);
    const sk = GameState.activeSquad[idx];
    sk.morale = clamp(sk.morale + randInt(10, 20), 0, 100);
    events.push(`🎉 ${sk.name} is feeling great! Morale boosted.`);
  }

  // 3. Form spike (10% chance)
  if (Math.random() < 0.10 && GameState.activeSquad.length > 0) {
    const idx = randInt(0, GameState.activeSquad.length - 1);
    const sk = GameState.activeSquad[idx];
    sk.form = clamp(sk.form + randInt(10, 25), 0, 100);
    events.push(`📈 ${sk.name} is on fire! Form spiked to ${sk.form}%.`);
  }

  // 4. Fan donation (8% chance)
  if (Math.random() < 0.08) {
    const amount = randInt(1, 5) * 1000;
    GameState.money += amount;
    events.push(`🎁 Fan donation: +${formatMoneyFull(amount)}!`);
  }

  // 5. Youth prospect appears (6% chance)
  if (Math.random() < 0.06 && GameState.reserveBench.length < 8) {
    const youth = createSkater(1);
    youth.status = 'reserve';
    GameState.reserveBench.push(youth);
    events.push(`🌱 Youth prospect ${youth.name} (${youth.nationality.flag}) joins the reserves!`);
  }

  // 6. Rival scandal (5% chance) — opponent loses points
  if (Math.random() < 0.05 && GameState.rivals.length > 0) {
    const rival = pick(GameState.rivals);
    const loss = randInt(10, 30);
    rival.points = Math.max(0, rival.points - loss);
    events.push(`📰 Scandal! ${rival.name} loses ${loss} points.`);
  }

  // 7. Equipment upgrade (7% chance) — team technique boost
  if (Math.random() < 0.07 && GameState.activeSquad.length > 0) {
    for (const sk of GameState.activeSquad) {
      sk.stats.technique = clamp(sk.stats.technique + 1, 1, 100);
      recalcSkater(sk);
    }
    events.push(`🔧 New equipment! All skaters gained +1 technique.`);
  }

  // 8. Fame event (6% chance)
  if (Math.random() < 0.06) {
    const fameGain = randInt(2, 8);
    GameState.fame += fameGain;
    events.push(`🌟 Media spotlight! +${fameGain} fame.`);
  }

  return events;
}

// ===== Advance Week =====
async function advanceWeek() {
  bgMusic.playClick();

  // Check if we need to play a competition first
  const compInfo = getThisWeekCompetition();
  if (compInfo && compInfo.entered && !compInfo.comp.competition) {
    showToast('You must compete first!', 'warning');
    return;
  }

  // Deduct wages
  const wages = getWeeklyWages();
  GameState.money -= wages;
  GameState.eventLog.push(`💵 Wages paid: -${formatMoneyFull(wages)}`);

  // Process sponsors
  const sponsorResult = processWeeklySponsors();
  if (sponsorResult.totalIncome > 0) {
    GameState.eventLog.push(`💼 Sponsor income: +${formatMoneyFull(sponsorResult.totalIncome)}`);
  }
  for (const msg of sponsorResult.messages) {
    GameState.eventLog.push(msg);
  }

  // Weekly stat fluctuation
  for (const sk of [...GameState.activeSquad, ...GameState.reserveBench]) {
    weeklyStatFluctuation(sk);
    recalcSkater(sk);
  }

  // AI market activity
  const marketMsgs = aiMarketActivity();
  for (const msg of marketMsgs) {
    GameState.eventLog.push(`🤖 ${msg}`);
  }

  // Market refresh every 2 weeks
  if (GameState.week % 2 === 0) {
    refreshMarket();
    GameState.eventLog.push('🛒 Market refreshed with new skaters.');
  }

  // Random events
  const events = triggerRandomEvents();
  for (const ev of events) {
    GameState.eventLog.push(ev);
    showToast(ev, 'info', 4000);
  }

  // Check bankruptcy
  if (GameState.money < -20000) {
    showToast('⚠ Warning: Severe debt! Sell players or cut costs.', 'danger', 5000);
  }

  // Advance week
  GameState.week++;
  GameState.scoutedThisWeek = false;

  // Check season end
  if (GameState.week > GameState.maxWeeks) {
    endSeason();
    return;
  }

  refreshUI();
  showToast(`Week ${GameState.week} begins`, 'success');
}

// ===== Competition flow =====
function startCompetition() {
  bgMusic.playClick();
  const compInfo = getThisWeekCompetition();
  if (!compInfo || !compInfo.entered) {
    showToast('No competition to play', 'warning');
    return;
  }
  if (compInfo.comp.competition) {
    showToast('Already competed this week', 'warning');
    return;
  }

  // Transition to mini-game screen
  showScreen('screen-minigame');
  bgMusic.stopMusic();

  miniGame.init(compInfo.comp);
  miniGame.onFinish = (routineResult) => {
    // Calculate placements
    const result = calculatePlacements(routineResult.score, compInfo.weekIndex);

    // Apply fame bonus from sponsors
    const fameBonus = getFameBonus();
    if (fameBonus > 0) {
      GameState.fame += fameBonus;
      result.fameAwarded += fameBonus;
    }

    // Apply win points bonus from sponsors
    if (result.placement === 1) {
      const wpBonus = getWinPointsBonus();
      if (wpBonus > 0) {
        GameState.points += wpBonus;
        result.pointsAwarded += wpBonus;
      }
    }

    // Log result
    const placeStr = result.placement <= 3 ? ['','🥇','🥈','🥉'][result.placement] : `#${result.placement}`;
    GameState.eventLog.push(`🏆 ${result.competition}: ${placeStr} (${result.score.toLocaleString()} pts)`);

    if (result.placement <= 3) {
      bgMusic.init();
      bgMusic.playWinFanfare();
    } else {
      bgMusic.init();
      bgMusic.playLoseSad();
    }

    // Switch to results screen
    setTimeout(() => {
      showScreen('screen-results');
      renderResults(result);
    }, 1500);
  };

  // Start the mini-game after a brief delay
  setTimeout(() => miniGame.start(), 500);
}

function enterComp() {
  bgMusic.playClick();
  const compInfo = getThisWeekCompetition();
  if (!compInfo || !compInfo.comp.name) return;

  showCompEntryModal(compInfo.comp, () => {
    const result = enterCompetition(compInfo.weekIndex);
    if (result.ok) {
      showToast(result.msg, 'gold');
      GameState.eventLog.push(`📝 ${result.msg}`);
      refreshUI();
    } else {
      showToast(result.msg, 'warning');
    }
  });
}

// ===== Season end =====
function endSeason() {
  // Save season history
  const entries = [
    { name: GameState.teamName, points: GameState.points, fame: GameState.fame, isPlayer: true },
    ...GameState.rivals.map(r => ({ name: r.name, points: r.points, fame: r.fame, isPlayer: false }))
  ];
  entries.sort((a, b) => b.points - a.points);
  const rank = entries.findIndex(e => e.isPlayer) + 1;

  GameState.seasonHistory.push({
    season: GameState.season,
    rank,
    points: GameState.points,
    fame: GameState.fame,
    money: GameState.money
  });

  if (rank === 1) {
    bgMusic.init();
    bgMusic.playWinFanfare();
  } else {
    bgMusic.init();
    bgMusic.playLoseSad();
  }

  showScreen('screen-season-end');
  renderSeasonEnd();
  autoSave();
}

function startNewSeason() {
  bgMusic.playClick();
  GameState.season++;
  GameState.week = 1;
  GameState.points = 0;
  GameState.enteredCompetitions = {};
  GameState.scoutedThisWeek = false;

  // Rivals reset for new season but keep some fame
  for (const rival of GameState.rivals) {
    rival.points = 0;
    rival.wins = 0;
    rival.strength = clamp(rival.strength + randInt(-5, 10), 30, 90);
  }

  // Age skaters
  for (const sk of [...GameState.activeSquad, ...GameState.reserveBench]) {
    sk.age++;
    // Retirement check for very old skaters
    if (sk.age >= 35 && Math.random() < 0.3) {
      GameState.eventLog.push(`👋 ${sk.name} (age ${sk.age}) has retired.`);
      releaseSkater(sk.id);
    } else {
      recalcSkater(sk);
    }
  }

  // New calendar
  GameState.calendar = generateCalendar(GameState.season);
  GameState.competitionResults = GameState.competitionResults.filter(r => r.season !== GameState.season);

  // Refresh market
  refreshMarket();

  showScreen('screen-game');
  refreshUI();
  showToast(`Season ${GameState.season} begins!`, 'gold', 4000);
}

// ===== New game =====
function startNewGame() {
  bgMusic.playClick();
  const teamName = document.getElementById('team-name').value.trim() || 'Ice Stars';
  const teamColor = document.getElementById('team-color').value || '#7dd3fc';
  const difficulty = document.getElementById('difficulty').value || 'semi-pro';

  resetState();
  GameState.teamName = teamName;
  GameState.teamColor = teamColor;
  GameState.difficulty = difficulty;

  // Generate squad
  const { squad, reserve } = generateStartingSquad(difficulty);
  GameState.activeSquad = squad;
  GameState.reserveBench = reserve;

  // Generate calendar
  GameState.calendar = generateCalendar(1);

  // Generate rivals
  GameState.rivals = generateRivals();

  // Generate market
  GameState.marketSkaters = generateMarketSkaters();

  GameState.eventLog.push(`⛸️ ${teamName} established! Ready for Season 1.`);

  showScreen('screen-game');
  refreshUI();
  showToast(`Welcome, Coach! ${teamName} is ready to skate!`, 'gold', 5000);
}

// ===== Load game =====
function loadSavedGame() {
  bgMusic.playClick();
  if (!hasSave()) {
    showToast('No saved game found', 'warning');
    return;
  }
  const ok = loadGame();
  if (ok) {
    showScreen('screen-game');
    refreshUI();
    showToast(`Welcome back, Coach!`, 'success');
  } else {
    showToast('Failed to load save', 'danger');
  }
}

// ===== Train team =====
function doTrainTeam() {
  bgMusic.playClick();
  const result = trainTeam();
  if (result.ok) {
    showToast(result.msg, 'success');
    GameState.eventLog.push(`🎯 ${result.msg}`);
    refreshUI();
  } else {
    showToast(result.msg, 'warning');
  }
}

// ===== Scout market =====
function doScout() {
  bgMusic.playClick();
  const result = scoutMarket();
  if (result.ok) {
    showToast(result.msg, result.msg.includes('STAR') ? 'gold' : 'success');
    GameState.eventLog.push(`🔍 ${result.msg}`);
    refreshUI();
  } else {
    showToast(result.msg, 'warning');
  }
}

// ===== Wire up all event listeners =====
function setupEventListeners() {
  // Menu buttons
  document.getElementById('btn-new-game').addEventListener('click', () => {
    bgMusic.playClick();
    showScreen('screen-setup');
  });
  document.getElementById('btn-load-game').addEventListener('click', loadSavedGame);
  document.getElementById('btn-settings').addEventListener('click', () => {
    bgMusic.playClick();
    showScreen('screen-settings');
  });

  // Setup screen
  document.getElementById('btn-start-game').addEventListener('click', startNewGame);
  document.getElementById('btn-setup-back').addEventListener('click', () => {
    bgMusic.playClick();
    showScreen('screen-menu');
  });

  // Settings screen
  document.getElementById('btn-settings-back').addEventListener('click', () => {
    bgMusic.playClick();
    showScreen('screen-menu');
  });
  document.getElementById('setting-volume').addEventListener('input', (e) => {
    const v = parseInt(e.target.value) / 100;
    GameState.volume = v;
    bgMusic.setVolume(v);
    miniGame.music.setVolume(v);
  });
  document.getElementById('setting-sfx').addEventListener('change', (e) => {
    GameState.sfxEnabled = e.target.checked;
    bgMusic.sfxEnabled = e.target.checked;
    miniGame.music.sfxEnabled = e.target.checked;
  });
  document.getElementById('setting-autosave').addEventListener('change', (e) => {
    GameState.autosave = e.target.checked;
  });

  // Tab navigation
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      bgMusic.playClick();
      const tab = btn.dataset.tab;
      switchTab(tab);

      // Re-render the panel that was switched to
      switch (tab) {
        case 'overview': renderOverview(); break;
        case 'squad': renderSquad(handleCardAction); break;
        case 'market': renderMarket(handleCardAction); break;
        case 'calendar': renderCalendar(); break;
        case 'sponsors': renderSponsors(handleNegotiate); break;
        case 'standings': renderStandings(); break;
      }
    });
  });

  // Game actions
  document.getElementById('btn-advance-week').addEventListener('click', advanceWeek);
  document.getElementById('btn-compete').addEventListener('click', startCompetition);
  document.getElementById('btn-train-team').addEventListener('click', doTrainTeam);
  document.getElementById('btn-scout').addEventListener('click', doScout);

  // Enter competition from overview (the "not entered" state)
  document.getElementById('overview-next-comp').addEventListener('click', (e) => {
    const badge = e.target.closest('.badge-not-entered');
    if (badge) enterComp();
  });

  // Results continue
  document.getElementById('btn-results-continue').addEventListener('click', () => {
    bgMusic.playClick();
    showScreen('screen-game');
    refreshUI();
  });

  // Season end buttons
  document.getElementById('btn-new-season').addEventListener('click', startNewSeason);
  document.getElementById('btn-season-menu').addEventListener('click', () => {
    bgMusic.playClick();
    showScreen('screen-menu');
  });

  // Audio context resume on first user gesture
  document.addEventListener('click', () => {
    bgMusic.init();
    bgMusic.resume();
  }, { once: true });
}

// ===== Initialization =====
function init() {
  setupEventListeners();

  // Show/hide continue button based on save existence
  const loadBtn = document.getElementById('btn-load-game');
  loadBtn.style.opacity = hasSave() ? '1' : '0.4';
  loadBtn.style.pointerEvents = hasSave() ? 'auto' : 'none';

  // Populate settings from defaults
  document.getElementById('setting-volume').value = Math.round(GameState.volume * 100);
  document.getElementById('setting-sfx').checked = GameState.sfxEnabled;
  document.getElementById('setting-autosave').checked = GameState.autosave;
}

// Boot
init();
