/* ===== Main entry point — game loop, events, routing ===== */
import { GameState, saveGame, loadGame, hasAnySave, getSaveInfo,
         resetState, serializeState, migrateState, isValidSave } from './state.js';
import { createSkater, generateStartingSquad, generateMarketSkaters, weeklyStatFluctuation,
         injureSkater, recalcSkater } from './skaters.js';
import { generateCalendar, enterCompetition, withdrawCompetition, simulateCompetition,
         getThisWeekCompetition, calculatePlacements, generateRivals } from './competitions.js';
import { processWeeklySponsors, negotiateSponsor, getFameBonus, getWinPointsBonus } from './sponsors.js';
import { promoteToActive, demoteToReserve, releaseSkater, buySkater,
         listForSale, cancelListing, aiMarketActivity, refreshMarket,
         scoutMarket, trainTeam, getWeeklyWages, findSkater, renewContract, hireStaff } from './squad.js';
import { MiniGame } from './minigame.js';
import { MusicEngine } from './music.js';
import { applyI18n, setLang } from './i18n.js';
import { randInt, pick, clamp, formatMoneyFull, formatMoney } from './utils.js';
import {
  STARTING_MONEY,
  ACTIVE_SQUAD_SIZE,
  RESERVE_SIZE,
  RETIREMENT_AGE,
  RETIREMENT_CHANCE,
  CONTRACT_EXPIRY_WARN_WEEKS
} from './config.js';
import {
  showToast, showModal, hideModal, confirmModal,
  showSkaterDetail, showSellModal, showBuyModal, showCompEntryModal,
  showScreen, switchTab, renderOverview, renderSquad,
  renderMarket, renderCalendar, renderSponsors, renderStandings,
  renderResults, renderSeasonEnd, renderStats, refreshAllPanels
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
        showToast(`Active squad full (max ${ACTIVE_SQUAD_SIZE})`, 'warning');
      }
      break;
    }
    case 'demote': {
      const ok = demoteToReserve(skaterId);
      if (ok) {
        showToast('Skater moved to reserves', 'info');
        refreshUI();
      } else {
        showToast(`Reserve bench full (max ${RESERVE_SIZE})`, 'warning');
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
    case 'renew': {
      const skater = findSkater(skaterId);
      if (!skater) return;
      const result = renewContract(skaterId);
      if (result.ok) {
        showToast(result.msg, 'gold');
        GameState.eventLog.push(`📝 ${result.msg}`);
        refreshUI();
      } else {
        showToast(result.msg, 'warning');
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

function refreshUI() {
  refreshAllPanels();
  autoSave();
}

function autoSave() {
  if (GameState.autosave) saveGame();
}

// ===== Random events (called during advanceWeek) =====

/** Track team records after each competition result. */
function updateRecords(result) {
  const rec = GameState.records;
  rec.totalPrize += result.prizeMoney || 0;
  if (result.placement === 1) rec.totalWins++;
  if (result.placement <= 3) rec.totalPodiums++;
  if (result.score > rec.bestScore) rec.bestScore = result.score;
  if (result.placement < rec.bestPlacement) rec.bestPlacement = result.placement;
}

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

  // 9. Blizzard training (4% chance) — tough conditions build stamina
  if (Math.random() < 0.04 && GameState.activeSquad.length > 0) {
    for (const sk of GameState.activeSquad) {
      sk.stats.stamina = clamp(sk.stats.stamina + 2, 1, 100);
      recalcSkater(sk);
    }
    events.push('🌨️ Blizzard training camp! All skaters gained +2 stamina.');
  }

  // 10. Sellout crowd (5% chance) — gate receipts
  if (Math.random() < 0.05) {
    const gate = randInt(3, 6) * 1000;
    GameState.money += gate;
    events.push(`🎫 Sellout crowd! Gate receipts: +${formatMoneyFull(gate)}.`);
  }

  // 11. Mentorship (4% chance) — a veteran boosts a young skater
  if (Math.random() < 0.04 && GameState.activeSquad.length > 0) {
    const veterans = GameState.activeSquad.filter(sk => sk.age >= 26);
    const youths = [...GameState.activeSquad, ...GameState.reserveBench].filter(sk => sk.age < 21);
    if (veterans.length > 0 && youths.length > 0) {
      const mentor = pick(veterans);
      const youth = pick(youths);
      const stat = pick(['technique', 'stamina', 'rhythm', 'sync', 'charisma']);
      youth.stats[stat] = clamp(youth.stats[stat] + 3, 1, 100);
      recalcSkater(youth);
      events.push(`🤝 ${mentor.name} mentored ${youth.name}: +3 ${stat}.`);
    }
  }

  // 12. Rivalry tension (4% chance) — pressure hurts morale
  if (Math.random() < 0.04 && GameState.activeSquad.length > 0) {
    for (const sk of GameState.activeSquad) {
      sk.morale = clamp(sk.morale - 3, 0, 100);
    }
    events.push('😤 Rivalry tension is building — team morale dropped.');
  }

  return events;
}

// ===== Advance Week =====
async function advanceWeek() {
  bgMusic.playClick();

  // Block advancing while an entered competition hasn't been played or withdrawn
  const compInfo = getThisWeekCompetition();
  if (compInfo && compInfo.entered && !compInfo.comp.competition) {
    showToast("You've entered this week's competition — compete or withdraw first!", 'warning');
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

  // Coaching staff passive effects
  if (GameState.staff.includes('fitness') || GameState.staff.includes('psychologist')) {
    for (const sk of [...GameState.activeSquad, ...GameState.reserveBench]) {
      if (GameState.staff.includes('fitness')) sk.form = clamp(sk.form + 2, 0, 100);
      if (GameState.staff.includes('psychologist')) sk.morale = clamp(sk.morale + 3, 0, 100);
    }
    GameState.eventLog.push('👔 Coaching staff worked with the team this week.');
  }

  // Warn about contracts expiring soon
  for (const sk of [...GameState.activeSquad, ...GameState.reserveBench]) {
    if (sk.contract.weeksRemaining === CONTRACT_EXPIRY_WARN_WEEKS) {
      GameState.eventLog.push(`⚠️ ${sk.name}'s contract expires in ${CONTRACT_EXPIRY_WARN_WEEKS} weeks.`);
    }
  }

  // Contract expiries — skaters with 0 weeks remaining leave the team
  const departed = [];
  GameState.activeSquad = GameState.activeSquad.filter(sk => {
    if (sk.contract.weeksRemaining <= 0) { departed.push(sk); return false; }
    return true;
  });
  GameState.reserveBench = GameState.reserveBench.filter(sk => {
    if (sk.contract.weeksRemaining <= 0) { departed.push(sk); return false; }
    return true;
  });
  for (const sk of departed) {
    GameState.eventLog.push(`📝 ${sk.name}'s contract expired — they left the team.`);
    showToast(`📝 ${sk.name}'s contract expired!`, 'warning', 4000);
  }
  // Auto-promote the best available reserve to keep the active squad full
  while (GameState.activeSquad.length < ACTIVE_SQUAD_SIZE && GameState.reserveBench.length > 0) {
    const best = GameState.reserveBench.reduce((a, b) => (b.overall > a.overall ? b : a));
    promoteToActive(best.id);
    GameState.eventLog.push(`⬆ ${best.name} promoted from reserves to fill the squad.`);
  }

  // AI market activity
  const marketMsgs = aiMarketActivity();
  for (const msg of marketMsgs) {
    GameState.eventLog.push(`🤖 ${msg}`);
  }

  // Market refresh cadence (driven by marketRefreshWeek)
  if (GameState.week >= GameState.marketRefreshWeek) {
    refreshMarket();
    GameState.marketRefreshWeek = GameState.week + 2;
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
  miniGame.onFinish = (routineResult) => finishCompetition(routineResult.score, compInfo);

  // Start the mini-game after a brief delay
  setTimeout(() => miniGame.start(), 500);
}

/** Shared post-competition flow: placements, sponsor bonuses, log, results screen. */
function finishCompetition(score, compInfo) {
  const result = calculatePlacements(score, compInfo.weekIndex);

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

  // Track records & appearances
  updateRecords(result);
  for (const sk of GameState.activeSquad) sk.appearances = (sk.appearances || 0) + 1;

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
}

/** Quick-sim the routine from squad stats instead of playing the mini-game. */
function quickSim() {
  bgMusic.playClick();
  const compInfo = getThisWeekCompetition();
  if (!compInfo || !compInfo.entered) {
    showToast('No competition to simulate', 'warning');
    return;
  }
  if (compInfo.comp.competition) {
    showToast('Already competed this week', 'warning');
    return;
  }

  GameState.eventLog.push('⚡ Quick-simulated the routine.');
  finishCompetition(simulateCompetition(), compInfo);
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

function withdrawComp() {
  bgMusic.playClick();
  const compInfo = getThisWeekCompetition();
  if (!compInfo || !compInfo.entered || compInfo.comp.competition) return;

  confirmModal('Withdraw from Competition',
    `Withdraw from ${compInfo.comp.name}? The entry fee will be lost.`, () => {
      const result = withdrawCompetition(compInfo.weekIndex);
      if (result.ok) {
        showToast(result.msg, 'info');
        GameState.eventLog.push(`↩ ${result.msg}`);
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
    // Retirement check for very old skaters — legends enter the Hall of Fame
    if (sk.age >= RETIREMENT_AGE && Math.random() < RETIREMENT_CHANCE) {
      GameState.eventLog.push(`👋 ${sk.name} (age ${sk.age}) has retired.`);
      if (sk.overall >= 75) {
        GameState.hallOfFame.push({ name: sk.name, overall: sk.overall, age: sk.age });
        GameState.eventLog.push(`🏅 ${sk.name} enters the Hall of Fame!`);
      }
      releaseSkater(sk.id);
    } else {
      recalcSkater(sk);
    }
  }

  // Youth intake draft — new prospects arrive at the start of the season
  const draftCount = Math.min(2, RESERVE_SIZE - GameState.reserveBench.length);
  if (draftCount > 0) {
    const drafted = [];
    for (let i = 0; i < draftCount; i++) {
      const youth = createSkater(1);
      youth.status = 'reserve';
      GameState.reserveBench.push(youth);
      drafted.push(youth.name);
    }
    GameState.eventLog.push(`📜 Youth draft: ${drafted.join(', ')} join the reserves.`);
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
  GameState.money = STARTING_MONEY[difficulty] || STARTING_MONEY['semi-pro'];

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

// ===== Save slots (load & save) =====
const SLOT_COUNT = 3;

function showLoadSlots() {
  bgMusic.playClick();
  const gameInProgress = GameState.activeSquad.length > 0;
  const rows = [];
  for (let slot = 1; slot <= SLOT_COUNT; slot++) {
    const info = getSaveInfo(slot);
    rows.push(`
      <div class="slot-row">
        <span class="slot-label">Slot ${slot}: ${info ? `${info.teamName} — S${info.season} W${info.week}` : '<em>empty</em>'}</span>
        <div class="modal-buttons" style="margin:0">
          ${info ? `<button class="modal-btn confirm" data-slot="${slot}" data-mode="load">📂 Load</button>` : ''}
          ${gameInProgress ? `<button class="modal-btn cancel" data-slot="${slot}" data-mode="save">💾 Save here</button>` : ''}
        </div>
      </div>
    `);
  }
  showModal(`
    <h3 class="modal-title">Save Slots</h3>
    ${rows.join('')}
    <div class="modal-buttons"><button class="modal-btn cancel" id="modal-slots-close">Close</button></div>
  `);
  document.querySelectorAll('#modal-content [data-slot]').forEach(btn => {
    btn.addEventListener('click', () => {
      const slot = parseInt(btn.dataset.slot);
      const mode = btn.dataset.mode;
      hideModal();
      if (mode === 'load') loadFromSlot(slot);
      else saveToSlot(slot);
    });
  });
  document.getElementById('modal-slots-close').addEventListener('click', hideModal);
}

function loadFromSlot(slot) {
  const ok = loadGame(slot);
  if (ok) {
    showScreen('screen-game');
    refreshUI();
    showToast('Welcome back, Coach!', 'success');
  } else {
    showToast('Failed to load save', 'danger');
  }
}

function saveToSlot(slot) {
  const ok = saveGame(slot);
  if (ok) {
    GameState.eventLog.push(`💾 Game saved to slot ${slot}.`);
    showToast(`Saved to slot ${slot}`, 'success');
  } else {
    showToast('Save failed', 'danger');
  }
}

// ===== Export / import save (file) =====
function exportSave() {
  bgMusic.playClick();
  try {
    const blob = new Blob([JSON.stringify(serializeState(), null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `skate-manager-save-s${GameState.season}-w${GameState.week}.json`;
    a.click();
    URL.revokeObjectURL(url);
    showToast('Save exported', 'success');
  } catch {
    showToast('Export failed', 'danger');
  }
}

function importSaveFile(file) {
  const reader = new FileReader();
  reader.onload = () => {
    try {
      const parsed = JSON.parse(reader.result);
      if (!isValidSave(parsed)) {
        showToast('Not a valid Skate Manager save', 'danger');
        return;
      }
      Object.assign(GameState, migrateState(parsed));
      GameState.minigameActive = false;
      GameState.currentCompetition = null;
      showScreen('screen-game');
      refreshUI();
      showToast(`Imported save: ${GameState.teamName}`, 'success');
      autoSave();
    } catch {
      showToast('Import failed — invalid file', 'danger');
    }
  };
  reader.readAsText(file);
}

// ===== Train team =====
function doTrainTeam() {
  bgMusic.playClick();
  const focusSelect = document.getElementById('train-focus');
  const focus = focusSelect ? focusSelect.value : 'balanced';
  const result = trainTeam(focus);
  if (result.ok) {
    showToast(result.msg, 'success');
    GameState.eventLog.push(`🎯 ${result.msg}`);
    refreshUI();
  } else {
    showToast(result.msg, 'warning');
  }
}

// ===== Hire coaching staff =====
function handleStaffHire(staffId) {
  bgMusic.playClick();
  const result = hireStaff(staffId);
  if (result.ok) {
    showToast(result.msg, 'gold');
    GameState.eventLog.push(`👔 ${result.msg}`);
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
  document.getElementById('btn-load-game').addEventListener('click', showLoadSlots);
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
  document.getElementById('setting-language').addEventListener('change', (e) => {
    GameState.language = e.target.value;
    setLang(e.target.value);
    applyI18n();
    refreshUI();
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
        case 'squad': renderSquad(); break;
        case 'market': renderMarket(); break;
        case 'calendar': renderCalendar(); break;
        case 'sponsors': renderSponsors(); break;
        case 'standings': renderStandings(); break;
        case 'stats': renderStats(); break;
      }
    });
  });

  // Game actions
  document.getElementById('btn-advance-week').addEventListener('click', advanceWeek);
  document.getElementById('btn-compete').addEventListener('click', startCompetition);
  document.getElementById('btn-quick-sim').addEventListener('click', quickSim);
  document.getElementById('btn-train-team').addEventListener('click', doTrainTeam);
  document.getElementById('btn-scout').addEventListener('click', doScout);

  // Delegated card actions + skater detail — one listener covers all panels.
  // Replaces re-wiring every button on each render.
  document.getElementById('game-content').addEventListener('click', (e) => {
    const btn = e.target.closest('.card-btn');
    if (btn) {
      e.stopPropagation();
      if (btn.classList.contains('negotiate')) {
        handleNegotiate(btn.dataset.sponsor);
        return;
      }
      if (btn.classList.contains('hire-staff')) {
        handleStaffHire(btn.dataset.staff);
        return;
      }
      const action = btn.classList.contains('promote') ? 'promote' :
                     btn.classList.contains('demote') ? 'demote' :
                     btn.classList.contains('sell') ? 'sell' :
                     btn.classList.contains('release') ? 'release' :
                     btn.classList.contains('buy') ? 'buy' :
                     btn.classList.contains('cancel-listing') ? 'cancel-listing' : null;
      if (action && btn.dataset.id) handleCardAction(action, btn.dataset.id);
      return;
    }
    // Click on card (not buttons) = show detail (squad skaters get renew action)
    const card = e.target.closest('.skater-card');
    if (card && card.dataset.id) {
      const skater = findSkater(card.dataset.id);
      if (skater) {
        const inSquad = GameState.activeSquad.includes(skater) || GameState.reserveBench.includes(skater);
        const actions = inSquad
          ? [{ id: 'renew', label: '📝 Renew Contract (+12 wks)', class: 'confirm' }]
          : [];
        showSkaterDetail(skater, actions);
      }
    }
  });

  // Enter / withdraw competition from overview
  document.getElementById('overview-next-comp').addEventListener('click', (e) => {
    if (e.target.closest('.badge-not-entered')) enterComp();
    if (e.target.closest('.badge-withdraw')) withdrawComp();
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

  // Save I/O (settings screen)
  document.getElementById('btn-export-save').addEventListener('click', exportSave);
  document.getElementById('btn-import-save').addEventListener('click', () => {
    document.getElementById('import-file').click();
  });
  document.getElementById('import-file').addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) importSaveFile(file);
    e.target.value = ''; // allow re-importing the same file
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
  setLang(GameState.language);
  applyI18n();

  // Register service worker for offline play (requires http/https)
  if ('serviceWorker' in navigator && location.protocol.startsWith('http')) {
    navigator.serviceWorker.register('sw.js').catch(() => {
      // offline support unavailable — game still works normally
    });
  }

  // Show/hide continue button based on save existence
  const loadBtn = document.getElementById('btn-load-game');
  loadBtn.style.opacity = hasAnySave() ? '1' : '0.4';
  loadBtn.style.pointerEvents = hasAnySave() ? 'auto' : 'none';

  // Populate settings from defaults
  document.getElementById('setting-volume').value = Math.round(GameState.volume * 100);
  document.getElementById('setting-sfx').checked = GameState.sfxEnabled;
  document.getElementById('setting-autosave').checked = GameState.autosave;
  const langSelect = document.getElementById('setting-language');
  if (langSelect) langSelect.value = GameState.language;
}

// Boot
init();
