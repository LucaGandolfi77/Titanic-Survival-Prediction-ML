/* ===== GAME LOOP, INITIALIZATION, STATE MACHINE ===== */
import { $, showScreen, showModal, hideModal, showPanel, hidePanel, saveGame, loadGame, saveRecords, loadRecords, wait } from './utils.js';
import { getState, setState, createNewState, resetDayStats, getTeamHappiness, getCharLoveLevel } from './state.js';
import { CHARACTERS, getDateableCharacters, getUnlockedCharacters } from './characters.js';
import { TASKS, getAvailableTasks, calculateTaskFame, calculateTaskMoney, getTaskMinigame } from './tasks.js';
import { TEAM_MEMBERS, payWages, applyDailyHappinessDecay, boostTeamHappiness, assignMemberToTask, releaseMember, releaseAll, getAvailableMembers } from './team.js';
import { initDaySlots, getCurrentSlot, advanceSlot, isLunchSlot, isEveningSlot, isDayOver, setSlotTask, setSlotResult } from './planner.js';
import { applyFood, applyDrink, applyTeamTreat, inviteCharacterToLunch } from './lunch.js';
import { canDateCharacter, getDateScene, applyDateChoice, checkGiftUnlock, getRandomEncounter, meetCharacter, getHighestLoveCharacter, getEndingForCharacter } from './dates.js';
import { startMinigame } from './minigames.js';
import { buyEquipment, addFame, addMoney, spendMoney, checkMilestones, getDayEndBonus, getEndingTier, getRomanceEnding } from './economy.js';
import { initAudio, playSound, startMusic, stopMusic, setMasterVolume, setMusicVolume, setSfxEnabled } from './audio.js';
import {
  updateHUD, renderDayPlanner, renderTeamStatus, renderTaskArea,
  showAssignModal, showLunchModal, showDateModal,
  renderTeamPanel, renderCharactersPanel, renderShopPanel,
  renderDayStart, renderEveningSummary, renderEnding, renderRecords,
  renderHowToPlay, showToast, showFamePopup, showEveningOptions
} from './ui.js';

/* ------ GAME PHASES ------
  menu → setup → dayStart → playing → (lunch|date|minigame) → evening → ending
*/

let gamePhase = 'menu';
let sfxOn = true;

/* ======================================================== */
/* INITIALIZATION                                           */
/* ======================================================== */
document.addEventListener('DOMContentLoaded', () => {
  showScreen('screen-menu');
  wireMenuEvents();
  wireSetupEvents();
  wireGameEvents();
  wireSettingsEvents();
  wireRecordsEvents();
  wireHowToPlayEvents();
  wireEndingEvents();
  wireDayStartEvents();
  wireEveningEvents();
});

/* ======================================================== */
/* MENU SCREEN                                              */
/* ======================================================== */
function wireMenuEvents() {
  $('#btn-new-game').onclick = () => {
    playSound('btn');
    gamePhase = 'setup';
    showScreen('screen-setup');
  };
  $('#btn-continue').onclick = () => {
    playSound('btn');
    const loaded = loadGame();
    if (loaded) {
      setState(loaded);
      gamePhase = 'playing';
      enterGameScreen();
    } else {
      showToast('No saved game found!', 'warning');
    }
  };
  $('#btn-records').onclick = () => {
    playSound('btn');
    const records = loadRecords();
    renderRecords(records);
    showScreen('screen-records');
  };
  $('#btn-how-to-play').onclick = () => {
    playSound('btn');
    renderHowToPlay();
    showScreen('screen-howto');
  };
}

/* ======================================================== */
/* SETUP SCREEN                                             */
/* ======================================================== */
function wireSetupEvents() {
  $('#btn-start').onclick = () => {
    playSound('btn');
    const name = $('#player-name').value.trim() || 'Captain';
    const teamName = $('#team-name').value.trim() || 'Aurora Creative';
    const diffEl = $('#setup-difficulty');
    const difficulty = diffEl ? diffEl.value : 'Normal';

    const s = createNewState();
    s.playerName = name;
    s.teamName = teamName;
    s.difficulty = difficulty;
    setState(s);

    gamePhase = 'dayStart';
    startDay();
  };
  const backBtn = $('#btn-back-setup');
  if (backBtn) {
    backBtn.onclick = () => {
      playSound('btn');
      gamePhase = 'menu';
      showScreen('screen-menu');
    };
  }
}

/* ======================================================== */
/* DAY START                                                */
/* ======================================================== */
function startDay() {
  const s = getState();
  initDaySlots(s);
  resetDayStats();

  // Daily encounter — auto-meet characters whose unlockDay matches
  for (const [id, c] of Object.entries(CHARACTERS)) {
    if (c.unlockDay === s.day && !s.characters[id].met) {
      meetCharacter(s, id);
      s.dailyEvents.push(`🤝 You met ${c.name} the ${c.role}!`);
    }
  }

  setState(s);
  renderDayStart();
  showScreen('screen-day-start');
  startMusic('morning');
}

function wireDayStartEvents() {
  $('#btn-begin-day').onclick = () => {
    playSound('btn');
    gamePhase = 'playing';
    enterGameScreen();
  };
}

/* ======================================================== */
/* MAIN GAME SCREEN                                        */
/* ======================================================== */
function enterGameScreen() {
  showScreen('screen-game');
  refreshGame();
}

function refreshGame() {
  const s = getState();
  updateHUD();
  renderDayPlanner();
  renderTeamStatus();
  renderTaskArea(handleSelectTask, handleStartSlot, handleAssignOrRest);

  // Switch music based on time of day
  const slot = s.daySlots[s.currentSlot];
  if (slot) {
    const hour = parseInt(slot.time.split(':')[0]);
    if (hour < 12) startMusic('morning');
    else if (hour < 16) startMusic('afternoon');
    else startMusic('evening');
  }

  saveGame(s);
}

/* ======================================================== */
/* TASK SELECTION                                          */
/* ======================================================== */
function handleSelectTask(taskId) {
  playSound('btn');

  const s = getState();
  const slot = getCurrentSlot(s);
  if (!slot) return;

  // Handle interview tasks
  if (taskId.startsWith('interview_')) {
    const charId = taskId.replace('interview_', '');
    setSlotTask(s, taskId, `Interview: ${CHARACTERS[charId].name}`);
  } else {
    const task = TASKS[taskId];
    if (!task) return;
    setSlotTask(s, taskId, task.name);
  }

  setState(s);
  refreshGame();
}

/* ======================================================== */
/* TEAM ASSIGNMENT                                         */
/* ======================================================== */
function handleAssignOrRest() {
  playSound('btn');
  const s = getState();
  const slot = getCurrentSlot(s);

  if (!slot) return;

  // If evening slot — rest
  if (slot.type === 'evening') {
    handleRestAndEndDay();
    return;
  }

  // Show assignment modal
  if (slot.taskId) {
    showAssignModal(slot.taskId, (members) => {
      const st = getState();
      // Release all from this task first
      for (const [mid, ms] of Object.entries(st.teamMembers)) {
        if (ms.assignedTo === slot.taskId) releaseMember(st, mid);
      }
      // Assign selected
      members.forEach(mid => assignMemberToTask(st, mid, slot.taskId));
      setState(st);
      refreshGame();
    });
  }
}

/* ======================================================== */
/* SLOT EXECUTION                                          */
/* ======================================================== */
function handleStartSlot() {
  playSound('btn');
  const s = getState();
  const slot = getCurrentSlot(s);
  if (!slot) return;

  // AUTO BRIEFING
  if (slot.type === 'auto') {
    advanceSlot(s);
    setState(s);
    refreshGame();
    return;
  }

  // LUNCH
  if (slot.type === 'lunch') {
    handleLunch();
    return;
  }

  // EVENING
  if (slot.type === 'evening') {
    handleEvening();
    return;
  }

  // WORK SLOT — must have task selected
  if (!slot.taskId) {
    showToast('Select a task first!', 'warning');
    return;
  }

  // Check if it's an interview
  if (slot.taskId.startsWith('interview_')) {
    handleInterview(slot.taskId.replace('interview_', ''));
    return;
  }

  // Normal task → minigame
  const task = TASKS[slot.taskId];
  if (!task) return;

  const mgType = getTaskMinigame(task);
  if (mgType) {
    gamePhase = 'minigame';
    playSound('mgStart');
    startMinigame(mgType, null, (multiplier, bonusFame) => {
      gamePhase = 'playing';
      completeTask(task, multiplier, bonusFame || 0);
    });
  } else {
    completeTask(task, 1, 0);
  }
}

/* ======================================================== */
/* COMPLETE TASK                                           */
/* ======================================================== */
function completeTask(task, multiplier, bonus) {
  const s = getState();

  const fame = calculateTaskFame(task, s, multiplier) + bonus;
  const money = calculateTaskMoney(task, s);

  addFame(s, fame);
  addMoney(s, money);
  playSound('taskComplete');

  // Energy cost
  s.energy = Math.max(0, s.energy - 10);

  // Track
  s.tasksCompleted++;
  s.tasksCompletedToday++;
  s.dailyFame += fame;
  s.dailyMoneyEarned += money;
  s.dailyEvents.push(`✅ ${task.name}: +${Math.round(fame)} fame, +€${money}`);

  // Release team members
  for (const [mid, ms] of Object.entries(s.teamMembers)) {
    if (ms.assignedTo === task.id) releaseMember(s, mid);
  }

  setSlotResult(s, fame);
  setState(s);

  showToast(`✅ ${task.name} complete! +${Math.round(fame)} ⭐`, 'fame');
  showFamePopup(fame);
  playSound('fame');

  checkMilestones(s);
  advanceSlot(s);
  setState(s);

  // Check if day is over after advancing
  if (isDayOver(s)) {
    handleDayEnd();
  } else {
    refreshGame();
  }
}

/* ======================================================== */
/* INTERVIEW                                               */
/* ======================================================== */
function handleInterview(charId) {
  const s = getState();
  const c = CHARACTERS[charId];
  if (!c) return;

  // Show interview as a mini-date scene
  const scene = {
    character: c,
    charId,
    bg: c.dateBg,
    dialogue: c.dialogues.interview
  };

  gamePhase = 'date';
  showDateModal(scene, (effect) => {
    gamePhase = 'playing';
    const st = getState();

    // Apply interview results
    const baseFame = c.interviewFame;
    let mult = effect === 'best' ? 1.5 : effect === 'good' ? 1.0 : 0.6;
    const fame = Math.round(baseFame * mult);

    addFame(st, fame);
    applyDateChoice(st, charId, effect);

    st.characters[charId].interviewed = true;
    st.tasksCompleted++;
    st.tasksCompletedToday++;
    st.dailyFame += fame;
    st.dailyEvents.push(`🎤 Interview with ${c.name}: +${fame} fame`);

    setSlotResult(st, fame);
    advanceSlot(st);
    setState(st);

    playSound('fame');
    showToast(`🎤 Interview complete! +${fame} ⭐`, 'fame');

    if (isDayOver(st)) {
      handleDayEnd();
    } else {
      refreshGame();
    }
  });
}

/* ======================================================== */
/* LUNCH                                                   */
/* ======================================================== */
function handleLunch() {
  gamePhase = 'lunch';

  showLunchModal((choices) => {
    gamePhase = 'playing';
    const s = getState();

    let spent = 0;

    if (choices.food) {
      applyFood(s, choices.food);
      spent += choices.food.price;
      s.dailyEvents.push(`🍽️ Lunch: ${choices.food.name}`);
    }
    if (choices.drink) {
      applyDrink(s, choices.drink);
      spent += choices.drink.price;
    }
    if (choices.treat) {
      applyTeamTreat(s, choices.treat);
      spent += choices.treat.price;
      s.dailyEvents.push(`🎉 Team treat: ${choices.treat.name}`);
      playSound('teamHappy');
    }
    if (choices.invite) {
      inviteCharacterToLunch(s, choices.invite);
      s.dailyEvents.push(`💕 Lunch with ${CHARACTERS[choices.invite].name}`);
      playSound('loveUp');
    }

    if (spent > 0) spendMoney(s, spent);

    advanceSlot(s);
    setState(s);
    refreshGame();
  });
}

/* ======================================================== */
/* EVENING                                                 */
/* ======================================================== */
function handleEvening() {
  gamePhase = 'evening';

  showEveningOptions(
    (charId) => { handleDate(charId); },
    () => { handleRestAndEndDay(); }
  );
}

function handleDate(charId) {
  const s = getState();
  const scene = getDateScene(charId, s);
  if (!scene) {
    showToast('Cannot date this character right now.', 'warning');
    return;
  }

  playSound('dateStart');
  startMusic('evening');

  showDateModal(scene, (effect) => {
    const st = getState();
    applyDateChoice(st, charId, effect);
    const c = CHARACTERS[charId];

    if (effect === 'best') {
      showToast(`❤️ ${c.name} loved that! Love +15`, 'love');
      playSound('loveUp');
    } else if (effect === 'good') {
      showToast(`💛 ${c.name} liked that. Love +8`, 'love');
    } else {
      showToast(`💔 ${c.name} didn't appreciate that. Love -5`, 'love');
      playSound('loveDn');
    }

    st.dailyEvents.push(`💕 Date with ${c.name}: ${effect}`);

    // Check gift unlock
    const gift = checkGiftUnlock(st, charId);
    if (gift) {
      showToast(`🎁 ${c.name} gave you a gift! +${gift.fameBonus} fame`, 'fame');
      playSound('milestone');
      st.dailyEvents.push(`🎁 Received gift from ${c.name}!`);
    }

    setState(st);
    handleRestAndEndDay();
  });
}

function handleRestAndEndDay() {
  const s = getState();
  s.energy = Math.min(100, s.energy + 20);
  advanceSlot(s);
  setState(s);
  handleDayEnd();
}

function wireEveningEvents() {
  // Wired dynamically via showEveningOptions
}

/* ======================================================== */
/* DAY END                                                 */
/* ======================================================== */
function handleDayEnd() {
  const s = getState();

  // Pay wages
  payWages(s);

  // Apply happiness decay
  applyDailyHappinessDecay(s);

  // Day end bonuses — getDayEndBonus returns array and applies money/fame internally
  const bonuses = getDayEndBonus(s);
  for (const b of bonuses) {
    s.dailyEvents.push(b.text);
  }

  setState(s);
  startMusic('night');

  // Check if it's day 7
  if (s.day >= 7) {
    gamePhase = 'ending';
    showEndingScreen();
    return;
  }

  // Show evening summary
  gamePhase = 'eveningSummary';
  renderEveningSummary();
  showScreen('screen-evening');
}

function showEndingScreen() {
  const s = getState();
  const endingData = getEndingTier(s.fame);
  const romCharId = getHighestLoveCharacter(s);
  const romLove = romCharId ? s.characters[romCharId].love : 0;
  const romanceData = getRomanceEnding(romCharId, romLove);

  renderEnding(endingData, romanceData, romCharId);
  showScreen('screen-ending');

  if (endingData.tier >= 4) {
    playSound('milestone');
  }

  // Save to records
  const records = loadRecords();
  records.push({
    name: s.playerName,
    fame: s.fame,
    money: s.money,
    difficulty: s.difficulty,
    romance: romCharId ? CHARACTERS[romCharId].name : 'None',
    date: new Date().toISOString(),
  });
  saveRecords(records);
}

/* ======================================================== */
/* EVENING SUMMARY → NEXT DAY                              */
/* ======================================================== */
document.addEventListener('DOMContentLoaded', () => {
  const nextDayBtn = $('#btn-end-day');
  if (nextDayBtn) {
    nextDayBtn.onclick = () => {
      playSound('btn');
      const s = getState();
      s.day++;
      s.currentSlot = 0;
      s.energy = Math.min(100, s.energy + 30);
      setState(s);
      gamePhase = 'dayStart';
      startDay();
    };
  }
});

/* ======================================================== */
/* ENDING / RECORDS / HOW TO PLAY / SETTINGS               */
/* ======================================================== */
function wireEndingEvents() {
  $('#btn-play-again').onclick = () => {
    playSound('btn');
    gamePhase = 'menu';
    showScreen('screen-menu');
    stopMusic();
  };
  const endMenuBtn = $('#btn-end-menu');
  if (endMenuBtn) {
    endMenuBtn.onclick = () => {
      playSound('btn');
      gamePhase = 'menu';
      showScreen('screen-menu');
      stopMusic();
    };
  }
}

function wireRecordsEvents() {
  $('#btn-records-back').onclick = () => {
    playSound('btn');
    showScreen('screen-menu');
  };
}

function wireHowToPlayEvents() {
  $('#btn-howto-back').onclick = () => {
    playSound('btn');
    showScreen('screen-menu');
  };
}

function wireSettingsEvents() {
  const musicVol = $('#music-vol');
  if (musicVol) {
    musicVol.oninput = (e) => setMusicVolume(parseFloat(e.target.value) / 100);
  }
  const sfxBtn = $('#btn-sfx');
  if (sfxBtn) {
    sfxBtn.onclick = () => {
      sfxOn = !sfxOn;
      setSfxEnabled(sfxOn);
      sfxBtn.textContent = sfxOn ? 'ON' : 'OFF';
      sfxBtn.classList.toggle('active', sfxOn);
    };
  }
  $('#btn-settings-back').onclick = () => {
    playSound('btn');
    showScreen('screen-menu');
  };
}

/* ======================================================== */
/* GAME SCREEN SIDE PANELS                                 */
/* ======================================================== */
function wireGameEvents() {
  // Panel toggle buttons
  $('#btn-open-team').onclick = () => {
    playSound('btn');
    renderTeamPanel();
    showPanel('panel-team');
  };
  $('#btn-open-chars').onclick = () => {
    playSound('btn');
    renderCharactersPanel();
    showPanel('panel-characters');
  };
  $('#btn-open-shop').onclick = () => {
    playSound('btn');
    renderShopPanel(handleBuyEquipment);
    showPanel('panel-shop');
  };
  // Settings from HUD
  $('#btn-open-settings').onclick = () => {
    playSound('btn');
    showScreen('screen-settings');
  };

  // Close panel buttons
  document.querySelectorAll('.close-btn').forEach(btn => {
    const panel = btn.closest('.side-panel');
    if (panel) {
      btn.onclick = () => {
        playSound('btn');
        hidePanel(panel.id);
      };
    }
  });
}

function handleBuyEquipment(equipId) {
  const s = getState();
  const success = buyEquipment(s, equipId);
  if (success) {
    setState(s);
    playSound('money');
    showToast('🛒 Equipment purchased!', 'money');
    refreshGame();
    return true;
  } else {
    playSound('error');
    showToast('Cannot buy this item!', 'warning');
    return false;
  }
}
