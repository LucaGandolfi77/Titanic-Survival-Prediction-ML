/* ===== ALL SCREENS, MODALS, PANELS, NOTIFICATIONS ===== */
import { $, $$, el, showScreen, showPanel, hidePanel, showModal, hideModal, formatMoney, formatFame, formatFameExact, getDayName, getTimeLabel } from './utils.js';
import { getState, getTeamHappiness, getCharLoveLevel, getFameMilestone } from './state.js';
import { CHARACTERS, getAllCharacters, getUnlockedCharacters, getDateableCharacters } from './characters.js';
import { TASKS, TASK_CATEGORIES, getAvailableTasks, getLockedTasks, calculateTaskFame, calculateTaskMoney, getTaskMinigame } from './tasks.js';
import { TEAM_MEMBERS, getMemberStatus, getHappinessColor, getAvailableMembers } from './team.js';
import { EQUIPMENT, getAvailableEquipment } from './economy.js';
import { FOOD_MENU, DRINKS_MENU, TEAM_TREATS } from './lunch.js';

/* ===== HUD UPDATE ===== */
export function updateHUD() {
  const s = getState();
  if (!s) return;

  $('#hud-money').textContent = `💰 ${formatMoney(s.money)}`;
  $('#hud-fame').textContent = `⭐ ${formatFame(s.fame)}`;
  $('#hud-happiness').textContent = `😊 Team: ${getTeamHappiness()}%`;
  $('#hud-day').textContent = `Day ${s.day} — ${getDayName(s.day)}`;

  const slot = s.daySlots[s.currentSlot];
  $('#hud-clock').textContent = slot ? slot.time : '22:00';

  // Fame bar
  const pct = Math.min((s.fame / 1000000) * 100, 100);
  $('#fame-bar-fill').style.width = `${pct}%`;
  $('#fame-label').textContent = `${formatFameExact(s.fame)} / 1,000,000`;
}

/* ===== DAY PLANNER ===== */
export function renderDayPlanner() {
  const s = getState();
  if (!s) return;

  const list = $('#time-slots-list');
  list.innerHTML = '';

  s.daySlots.forEach((slot, i) => {
    const div = el('div', {
      class: `time-slot ${slot.status}`,
      onclick: () => {
        if (slot.status === 'active') {
          // Already on this slot
        }
      }
    }, [
      el('span', { class: 'slot-time', text: slot.time }),
      el('span', { class: 'slot-icon', text: slot.icon }),
      el('span', { class: 'slot-label', text: slot.taskId ? TASKS[slot.taskId]?.name || slot.name : slot.name }),
    ]);
    list.appendChild(div);
  });
}

/* ===== TEAM STATUS PANEL ===== */
export function renderTeamStatus() {
  const s = getState();
  if (!s) return;

  const container = $('#team-mini-cards');
  container.innerHTML = '';

  for (const [id, info] of Object.entries(TEAM_MEMBERS)) {
    const ms = s.teamMembers[id];
    const status = getMemberStatus(ms);
    const happColor = getHappinessColor(ms.happiness);

    const card = el('div', { class: 'team-mini-card' }, [
      el('div', { class: 'member-top' }, [
        el('span', { class: 'member-emoji', text: info.emoji }),
        el('div', {}, [
          el('span', { class: 'member-name', text: info.name }),
          el('span', { class: 'member-role', text: info.role }),
        ]),
      ]),
      el('div', { class: 'happiness-bar' }, [
        el('div', { class: 'happiness-fill', style: { width: `${ms.happiness}%`, background: happColor } }),
      ]),
      el('span', { class: `member-status ${status}`, text: status === 'available' ? '✓ Available' : status === 'busy' ? '📋 On Task' : '😤 Unhappy' }),
    ]);
    container.appendChild(card);
  }

  // Morale bar
  const morale = getTeamHappiness();
  const moraleColor = getHappinessColor(morale);
  $('#morale-bar-fill').style.width = `${morale}%`;
  $('#morale-bar-fill').style.background = moraleColor;
}

/* ===== TASK AREA ===== */
export function renderTaskArea(onSelectTask, onStartTask, onAssignTeam) {
  const s = getState();
  if (!s) return;

  const slot = s.daySlots[s.currentSlot];
  if (!slot) return;

  const area = $('#active-task-card');
  const bg = $('#task-location-bg');

  if (slot.type === 'auto') {
    // Morning briefing
    $('#task-icon').textContent = '☀️';
    $('#task-name').textContent = 'Morning Briefing';
    $('#task-description').textContent = `Good morning, ${s.playerName}! Day ${s.day} of your cruise aboard the MS Aurora Infinita. Review today's available tasks and plan your schedule.`;
    $('#task-fame-reward').textContent = '';
    $('#task-money-reward').textContent = '';
    $('#task-duration').textContent = '';
    bg.style.background = 'linear-gradient(135deg, #f97316, #ec4899)';
    $('#task-team-assign').innerHTML = '';
    $('#task-actions').innerHTML = '';
    const nextBtn = el('button', { class: 'task-btn primary', text: '▶ START THE DAY', onclick: onStartTask });
    $('#task-actions').appendChild(nextBtn);
    return;
  }

  if (slot.type === 'lunch') {
    $('#task-icon').textContent = '🍽️';
    $('#task-name').textContent = 'Lunch Break';
    $('#task-description').textContent = 'Time to eat, drink, and boost your team\'s morale!';
    $('#task-fame-reward').textContent = '';
    $('#task-money-reward').textContent = '';
    $('#task-duration').textContent = '⏱ 1 hour';
    bg.style.background = 'linear-gradient(135deg, #92400e, #b45309)';
    $('#task-team-assign').innerHTML = '';
    $('#task-actions').innerHTML = '';
    const lunchBtn = el('button', { class: 'task-btn primary', text: '🍽️ OPEN LUNCH MENU', onclick: onStartTask });
    $('#task-actions').appendChild(lunchBtn);
    return;
  }

  if (slot.type === 'evening') {
    $('#task-icon').textContent = '🌅';
    $('#task-name').textContent = 'Evening Free Time';
    $('#task-description').textContent = 'The day is winding down. You can go on a date, have a team dinner, or rest.';
    $('#task-fame-reward').textContent = '';
    $('#task-money-reward').textContent = '';
    $('#task-duration').textContent = '';
    bg.style.background = 'linear-gradient(135deg, #f97316, #ec4899, #8b5cf6)';
    $('#task-team-assign').innerHTML = '';
    $('#task-actions').innerHTML = '';
    const dateBtn = el('button', { class: 'task-btn primary', text: '💕 DATE / EVENTS', onclick: onStartTask });
    const restBtn = el('button', { class: 'task-btn secondary', text: '😴 REST & END DAY', onclick: onAssignTeam });
    $('#task-actions').appendChild(dateBtn);
    $('#task-actions').appendChild(restBtn);
    return;
  }

  // Work slot — show task selection
  if (!slot.taskId) {
    showTaskSelection(onSelectTask);
    return;
  }

  // Task is selected — show task details
  const task = TASKS[slot.taskId];
  if (!task) return;

  $('#task-icon').textContent = task.icon;
  $('#task-name').textContent = task.name;
  $('#task-description').textContent = task.description;
  const estFame = calculateTaskFame(task, s, 1);
  const estMoney = calculateTaskMoney(task, s);
  $('#task-fame-reward').textContent = `⭐ ~${formatFame(estFame)} fame`;
  $('#task-money-reward').textContent = `💰 +${formatMoney(estMoney)}`;
  $('#task-duration').textContent = '⏱ 1.5 hrs';
  bg.style.background = task.bg;

  // Team assignment display
  const assignDiv = $('#task-team-assign');
  assignDiv.innerHTML = '';
  for (const [mid, ms] of Object.entries(s.teamMembers)) {
    if (ms.assignedTo === task.id) {
      const info = TEAM_MEMBERS[mid];
      const badge = el('span', { class: 'assigned-member', text: `${info.emoji} ${info.name}` });
      assignDiv.appendChild(badge);
    }
  }

  $('#task-actions').innerHTML = '';
  const assignBtn = el('button', { class: 'task-btn secondary', text: '👥 ASSIGN TEAM', onclick: onAssignTeam });
  const startBtn = el('button', { class: 'task-btn primary', text: '▶ START TASK', onclick: onStartTask });
  $('#task-actions').appendChild(assignBtn);
  $('#task-actions').appendChild(startBtn);
}

function showTaskSelection(onSelectTask) {
  const s = getState();
  const available = getAvailableTasks(s);
  const locked = getLockedTasks(s);

  $('#task-icon').textContent = '📋';
  $('#task-name').textContent = 'Choose a Task';
  $('#task-description').textContent = 'Select a task for this time slot.';
  $('#task-fame-reward').textContent = '';
  $('#task-money-reward').textContent = '';
  $('#task-duration').textContent = '';
  $('#task-location-bg').style.background = 'linear-gradient(135deg, var(--bg-dark), var(--bg-medium))';
  $('#task-team-assign').innerHTML = '';
  $('#task-actions').innerHTML = '';

  const grid = el('div', { class: 'task-choice-grid' });

  available.forEach(task => {
    const card = el('div', { class: 'task-choice-card', onclick: () => onSelectTask(task.id) }, [
      el('div', { class: 'tcc-icon', text: task.icon }),
      el('div', { class: 'tcc-name', text: task.name }),
      el('div', { class: 'tcc-reward', text: `⭐ ${formatFame(task.baseFame)}` }),
      el('div', { class: 'tcc-money', text: `💰 ${formatMoney(task.money)}` }),
    ]);
    grid.appendChild(card);
  });

  // Add interview option
  const interviewable = Object.entries(CHARACTERS).filter(([id, c]) => {
    const cs = s.characters[id];
    if (!cs || !cs.met || cs.interviewed) return false;
    if (id === 'captain') return s.day >= 5 && s.captainTrust >= 80;
    return true;
  });

  interviewable.forEach(([id, c]) => {
    const card = el('div', { class: 'task-choice-card', onclick: () => onSelectTask(`interview_${id}`) }, [
      el('div', { class: 'tcc-icon', text: '🎤' }),
      el('div', { class: 'tcc-name', text: `Interview: ${c.name}` }),
      el('div', { class: 'tcc-reward', text: `⭐ ${formatFame(c.interviewFame)}` }),
      el('div', { class: 'tcc-money', text: `+❤️ Love +8` }),
    ]);
    grid.appendChild(card);
  });

  locked.forEach(task => {
    const card = el('div', { class: 'task-choice-card locked' }, [
      el('div', { class: 'tcc-icon', text: task.icon }),
      el('div', { class: 'tcc-name', text: task.name }),
      el('div', { class: 'tcc-lock', text: task.unlockText || `Unlocks Day ${task.unlockDay}` }),
    ]);
    grid.appendChild(card);
  });

  $('#task-actions').appendChild(grid);
}

/* ===== TEAM ASSIGNMENT MODAL ===== */
export function showAssignModal(taskId, onConfirm) {
  const s = getState();
  const modal = $('#modal-assign');
  modal.classList.remove('hidden');

  const rolesDiv = $('#assign-roles');
  rolesDiv.innerHTML = '';

  const available = getAvailableMembers(s);
  const selectedMembers = new Set();

  // Get current assignments for this task
  for (const [mid, ms] of Object.entries(s.teamMembers)) {
    if (ms.assignedTo === taskId) selectedMembers.add(mid);
  }

  const label = el('div', { class: 'assign-role' }, [
    el('label', { text: 'Select team members for this task:' }),
  ]);

  const options = el('div', { class: 'assign-options' });

  for (const [id, info] of Object.entries(TEAM_MEMBERS)) {
    const ms = s.teamMembers[id];
    const isAvail = ms.status === 'available' || ms.assignedTo === taskId;
    const opt = el('div', {
      class: `assign-option ${selectedMembers.has(id) ? 'selected' : ''} ${!isAvail ? 'unavailable' : ''}`,
      text: `${info.emoji} ${info.name}`,
      onclick: () => {
        if (!isAvail) return;
        if (selectedMembers.has(id)) {
          selectedMembers.delete(id);
          opt.classList.remove('selected');
        } else {
          selectedMembers.add(id);
          opt.classList.add('selected');
        }
      }
    });
    options.appendChild(opt);
  }

  label.appendChild(options);
  rolesDiv.appendChild(label);

  $('#btn-confirm-assign').onclick = () => {
    modal.classList.add('hidden');
    onConfirm([...selectedMembers]);
  };
  $('#btn-cancel-assign').onclick = () => {
    modal.classList.add('hidden');
  };
}

/* ===== LUNCH MODAL ===== */
export function showLunchModal(onConfirm) {
  const s = getState();
  showModal('modal-lunch');

  let selectedFood = null;
  let selectedDrink = null;
  let selectedTreat = null;
  let invitedChar = null;
  let currentTab = 'food';

  function renderTab(tab) {
    currentTab = tab;
    const grid = $('#lunch-menu-grid');
    grid.innerHTML = '';

    $$('.lunch-tab').forEach(t => t.classList.toggle('active', t.dataset.tab === tab));

    if (tab === 'food') {
      FOOD_MENU.forEach(item => {
        const div = el('div', {
          class: `lunch-item ${selectedFood?.id === item.id ? 'selected' : ''}`,
          onclick: () => { selectedFood = item; renderTab(tab); updateLunchTotal(); }
        }, [
          el('span', { class: 'food-emoji', text: item.emoji }),
          el('div', { class: 'food-info' }, [
            el('div', { class: 'food-name', text: item.name }),
            el('div', { class: 'food-effects', text: `Energy +${item.energy} | Happiness +${item.happiness}` }),
            item.bonusMember ? el('div', { class: 'food-bonus', text: `💕 ${TEAM_MEMBERS[item.bonusMember]?.name} bonus +${item.bonusAmount}` }) : null,
          ]),
          el('span', { class: 'food-price', text: formatMoney(item.price) }),
        ]);
        grid.appendChild(div);
      });
    } else if (tab === 'drinks') {
      DRINKS_MENU.forEach(item => {
        const div = el('div', {
          class: `lunch-item ${selectedDrink?.id === item.id ? 'selected' : ''}`,
          onclick: () => { selectedDrink = item; renderTab(tab); updateLunchTotal(); }
        }, [
          el('span', { class: 'food-emoji', text: item.emoji }),
          el('div', { class: 'food-info' }, [
            el('div', { class: 'food-name', text: item.name }),
            el('div', { class: 'food-effects', text: item.effect }),
          ]),
          el('span', { class: 'food-price', text: formatMoney(item.price) }),
        ]);
        grid.appendChild(div);
      });
    } else if (tab === 'team') {
      TEAM_TREATS.forEach(item => {
        const div = el('div', {
          class: `lunch-item ${selectedTreat?.id === item.id ? 'selected' : ''}`,
          onclick: () => { selectedTreat = item; renderTab(tab); updateLunchTotal(); }
        }, [
          el('span', { class: 'food-emoji', text: item.emoji }),
          el('div', { class: 'food-info' }, [
            el('div', { class: 'food-name', text: item.name }),
            el('div', { class: 'food-effects', text: `Team happiness +${item.teamHappiness}` + (item.fame > 0 ? ` | Fame +${item.fame}` : '') }),
          ]),
          el('span', { class: 'food-price', text: formatMoney(item.price) }),
        ]);
        grid.appendChild(div);
      });
    } else if (tab === 'invite') {
      const dateable = getDateableCharacters(s);
      if (dateable.length === 0) {
        grid.appendChild(el('div', { text: 'No characters available to invite yet. Build more relationships!', style: { color: 'var(--text-muted)', textAlign: 'center', padding: '20px' } }));
      } else {
        dateable.forEach(c => {
          const cs = s.characters[c.id];
          const div = el('div', {
            class: `lunch-item ${invitedChar === c.id ? 'selected' : ''}`,
            onclick: () => { invitedChar = invitedChar === c.id ? null : c.id; renderTab(tab); }
          }, [
            el('span', { class: 'food-emoji', text: c.emoji }),
            el('div', { class: 'food-info' }, [
              el('div', { class: 'food-name', text: c.name }),
              el('div', { class: 'food-effects', text: `Love: ${cs.love}/100 | +15 Love` }),
            ]),
            el('span', { class: 'food-price', text: 'Free' }),
          ]);
          grid.appendChild(div);
        });
      }
    }
  }

  function updateLunchTotal() {
    let total = 0;
    if (selectedFood) total += selectedFood.price;
    if (selectedDrink) total += selectedDrink.price;
    if (selectedTreat) total += selectedTreat.price;
    $('#lunch-total').textContent = `Total: ${formatMoney(total)}`;
    $('#lunch-energy').textContent = `Energy: ${s.energy}`;
  }

  $$('.lunch-tab').forEach(tab => {
    tab.onclick = () => renderTab(tab.dataset.tab);
  });

  renderTab('food');
  updateLunchTotal();

  $('#btn-lunch-confirm').onclick = () => {
    hideModal('modal-lunch');
    onConfirm({ food: selectedFood, drink: selectedDrink, treat: selectedTreat, invite: invitedChar });
  };
}

/* ===== DATE MODAL ===== */
export function showDateModal(scene, onComplete) {
  showModal('modal-date');

  const bg = $('#date-scene-bg');
  bg.style.background = scene.bg;

  $('#date-character-avatar').textContent = scene.character.emoji;
  $('#date-char-name').textContent = scene.character.name;
  $('#date-dialogue-text').textContent = scene.dialogue.text;

  const loveBar = $('#love-bar-fill');
  const s = getState();
  const cs = s.characters[scene.charId];
  loveBar.style.width = `${cs.love}%`;
  $('#love-value').textContent = cs.love;

  const choicesDiv = $('#date-choices');
  choicesDiv.innerHTML = '';

  if (scene.dialogue.choices) {
    scene.dialogue.choices.forEach(choice => {
      const btn = el('div', { class: 'date-choice', text: choice.text, onclick: () => {
        btn.classList.add(choice.effect === 'best' ? 'best' : choice.effect === 'bad' ? 'wrong' : '');
        choicesDiv.querySelectorAll('.date-choice').forEach(c => c.style.pointerEvents = 'none');

        setTimeout(() => {
          hideModal('modal-date');
          onComplete(choice.effect);
        }, 800);
      }});
      choicesDiv.appendChild(btn);
    });
  } else {
    // No choices — just dialogue
    const continueBtn = el('div', { class: 'date-choice', text: 'Continue...', onclick: () => {
      hideModal('modal-date');
      onComplete('good');
    }});
    choicesDiv.appendChild(continueBtn);
  }
}

/* ===== TEAM PANEL ===== */
export function renderTeamPanel() {
  const s = getState();
  const container = $('#team-full-cards');
  container.innerHTML = '';

  for (const [id, info] of Object.entries(TEAM_MEMBERS)) {
    const ms = s.teamMembers[id];
    const happColor = getHappinessColor(ms.happiness);

    const card = el('div', { class: 'team-card' }, [
      el('span', { class: 'tc-avatar', text: info.emoji }),
      el('div', { class: 'tc-info' }, [
        el('div', { class: 'tc-name', text: info.name }),
        el('div', { class: 'tc-title', text: `${info.role} | ${info.nationality}` }),
        el('div', { class: 'tc-catch', text: info.catchphrase }),
        el('div', { class: 'tc-stats' }, [
          el('span', { class: 'tc-stat', html: `📹 <span class="stat-val">${info.stats.video}</span>` }),
          el('span', { class: 'tc-stat', html: `📷 <span class="stat-val">${info.stats.photo}</span>` }),
          el('span', { class: 'tc-stat', html: `✂️ <span class="stat-val">${info.stats.editing}</span>` }),
        ]),
        el('div', { style: { fontSize: '0.75rem', color: 'var(--ocean-blue)', marginTop: '4px' } }, [
          el('span', { text: `⚡ ${info.specialSkill}: ${info.specialDesc}` }),
        ]),
      ]),
      el('div', { class: 'tc-happiness' }, [
        el('div', { text: `${ms.happiness}%`, style: { fontSize: '0.85rem', fontWeight: '600', color: happColor } }),
        el('div', { class: 'tc-happiness-bar' }, [
          el('div', { class: 'tc-happiness-fill', style: { width: `${ms.happiness}%`, background: happColor } }),
        ]),
        el('div', { class: 'tc-wage', text: `💰 ${formatMoney(info.wage)}/day` }),
      ]),
    ]);
    container.appendChild(card);
  }
}

/* ===== CHARACTERS PANEL ===== */
export function renderCharactersPanel() {
  const s = getState();
  const container = $('#characters-grid');
  container.innerHTML = '';

  for (const [id, c] of Object.entries(CHARACTERS)) {
    const cs = s.characters[id];
    const loveLevel = getCharLoveLevel(id);
    const hearts = Math.floor(cs.love / 20);

    const card = el('div', { class: 'character-card' }, [
      el('div', { class: `char-avatar ${c.avatarClass}` }, [
        el('span', { text: c.emoji }),
        el('span', { class: 'char-mood', text: cs.met ? (cs.mood === 'happy' ? '😊' : cs.mood === 'tired' ? '😴' : '😐') : '❓' }),
      ]),
      el('div', { class: 'char-name', text: cs.met ? c.name : '???' }),
      el('div', { class: 'char-role', text: cs.met ? `${c.role} ${c.flag}` : 'Unknown' }),
      el('div', { class: 'love-hearts', html: cs.met ?
        Array(5).fill(0).map((_, i) => `<span class="${i < hearts ? 'filled' : 'empty'}">♥</span>`).join('') :
        '<span class="empty">♥♥♥♥♥</span>'
      }),
      el('span', { class: `char-status-badge ${loveLevel}`, text: cs.met ? loveLevel : 'Not met' }),
    ]);

    container.appendChild(card);
  }
}

/* ===== SHOP PANEL ===== */
export function renderShopPanel(onBuy) {
  const s = getState();
  const container = $('#shop-items-grid');
  container.innerHTML = '';

  for (const eq of EQUIPMENT) {
    const owned = s.equipment[eq.id];
    const locked = eq.unlockDay && s.day < eq.unlockDay;
    const canAfford = s.money >= eq.price;

    const item = el('div', { class: `shop-item ${owned ? 'purchased' : ''} ${locked ? 'locked' : ''}` }, [
      el('span', { class: 'shop-icon', text: eq.icon }),
      el('div', { class: 'shop-info' }, [
        el('div', { class: 'shop-name', text: eq.name }),
        el('div', { class: 'shop-desc', text: formatMoney(eq.price) }),
        el('div', { class: 'shop-effect', text: eq.effect }),
        locked ? el('div', { style: { fontSize: '0.7rem', color: 'var(--text-muted)' }, text: `Unlocks Day ${eq.unlockDay}` }) : null,
      ]),
      !owned && !locked ? el('button', {
        class: 'shop-buy-btn',
        text: 'BUY',
        disabled: !canAfford ? '' : undefined,
        onclick: () => {
          if (onBuy(eq.id)) renderShopPanel(onBuy);
        }
      }) : null,
    ]);
    container.appendChild(item);
  }
}

/* ===== DAY START SCREEN ===== */
export function renderDayStart() {
  const s = getState();

  $('#day-start-title').textContent = `Day ${s.day} — ${getDayName(s.day)}`;
  $('#day-start-subtitle').textContent = `Good morning, ${s.playerName}! Day ${s.day} of your cruise aboard the MS Aurora Infinita.`;

  const tips = [
    '💡 Assign your best photographers to photo shoots for maximum fame!',
    '💡 Keep your team happy — low morale means less fame!',
    '💡 Lunch breaks are a great time to boost team happiness!',
    '💡 Build relationships with crew members for bonus fame and gifts!',
    '💡 Buy equipment early — the fame bonuses compound over time!',
    '💡 The Captain\'s interview is worth a massive fame boost!',
    '💡 Mini-game performance directly multiplies your task fame!',
  ];
  const tip = tips[(s.day - 1) % tips.length];
  $('#day-start-tip').textContent = tip;

  const tasksDiv = $('#day-start-tasks');
  tasksDiv.innerHTML = '<h4>📋 Available Tasks Today:</h4>';
  const available = getAvailableTasks(s);
  available.slice(0, 5).forEach(task => {
    const preview = el('div', { class: 'day-task-preview' }, [
      el('span', { text: task.icon }),
      el('span', { text: task.name }),
      el('span', { text: `⭐ ${formatFame(task.baseFame)}`, style: { marginLeft: 'auto', color: 'var(--fame-gold)', fontFamily: 'var(--font-mono)', fontSize: '0.8rem' } }),
    ]);
    tasksDiv.appendChild(preview);
  });

  // Character encounters for the day
  const encounters = [];
  for (const [id, c] of Object.entries(CHARACTERS)) {
    const cs = s.characters[id];
    if (c.unlockDay === s.day && !cs.met) {
      encounters.push(`🤝 You'll meet ${c.name} today!`);
    }
  }
  encounters.forEach(e => {
    const preview = el('div', { class: 'day-task-preview', text: e });
    tasksDiv.appendChild(preview);
  });
}

/* ===== EVENING SUMMARY ===== */
export function renderEveningSummary() {
  const s = getState();

  const statsDiv = $('#evening-stats');
  statsDiv.innerHTML = '';

  const stats = [
    { label: 'Tasks Completed', value: s.tasksCompletedToday, color: 'var(--ocean-blue)' },
    { label: 'Fame Earned', value: `+${formatFameExact(s.dailyFame)} ⭐`, color: 'var(--fame-gold)' },
    { label: 'Money Earned', value: `+${formatMoney(s.dailyMoneyEarned)}`, color: 'var(--grass-green)' },
    { label: 'Money Spent', value: `-${formatMoney(s.dailyMoneySpent)}`, color: 'var(--danger-red)' },
    { label: 'Team Happiness', value: `${getTeamHappiness()}%`, color: getHappinessColor(getTeamHappiness()) },
    { label: 'Total Fame', value: formatFameExact(s.fame), color: 'var(--fame-purple)' },
  ];

  stats.forEach(stat => {
    const div = el('div', { class: 'evening-stat' }, [
      el('span', { class: 'stat-value', text: stat.value, style: { color: stat.color } }),
      el('span', { class: 'stat-label', text: stat.label }),
    ]);
    statsDiv.appendChild(div);
  });

  const eventsDiv = $('#evening-events');
  eventsDiv.innerHTML = '';
  s.dailyEvents.forEach(ev => {
    eventsDiv.appendChild(el('div', { class: 'evening-event', text: ev }));
  });
}

/* ===== ENDING SCREEN ===== */
export function renderEnding(endingData, romanceData, romanceCharId) {
  const s = getState();

  const endingBg = $('#ending-bg');
  endingBg.style.background = endingData.tier >= 4 ?
    'linear-gradient(135deg, #fbbf24, #f59e0b, #f97316)' :
    'linear-gradient(135deg, #0ea5e9, #1e3a5f)';
  endingBg.style.opacity = '0.15';

  $('#ending-title').textContent = endingData.title;
  $('#ending-text').textContent = endingData.text;

  const romanceDiv = $('#ending-romance');
  romanceDiv.innerHTML = '';

  if (romanceData.level === 'romance' && romanceCharId) {
    const c = CHARACTERS[romanceCharId];
    romanceDiv.innerHTML = `
      <div style="text-align:center;margin-bottom:12px;">
        <span style="font-size:3rem;">${c.emoji}</span>
        <h3 style="color:var(--love-pink);margin-top:8px;">❤️ ${c.name}</h3>
      </div>
      <p style="color:var(--text-primary);font-style:italic;line-height:1.7;">${c.dialogues.ending}</p>
    `;
  } else {
    romanceDiv.innerHTML = `<p style="color:var(--text-muted);">${romanceData.text}</p>`;
  }

  const statsDiv = $('#ending-stats');
  statsDiv.innerHTML = '';
  const finalStats = [
    ['Total Fame', formatFameExact(s.fame)],
    ['Budget Remaining', formatMoney(s.money)],
    ['Tasks Completed', s.tasksCompleted],
    ['Team Final Morale', `${getTeamHappiness()}%`],
  ];
  finalStats.forEach(([label, value]) => {
    statsDiv.appendChild(el('div', { class: 'ending-stat-row' }, [
      el('span', { text: label, style: { color: 'var(--text-muted)' } }),
      el('span', { text: value, style: { fontFamily: 'var(--font-mono)', color: 'var(--text-gold)' } }),
    ]));
  });
}

/* ===== RECORDS ===== */
export function renderRecords(records) {
  const container = $('#records-list');
  container.innerHTML = '';

  if (records.length === 0) {
    container.appendChild(el('div', { text: 'No records yet. Complete a cruise to see your results!', style: { color: 'var(--text-muted)', textAlign: 'center', padding: '40px' } }));
    return;
  }

  records.sort((a, b) => b.fame - a.fame).slice(0, 10).forEach((r, i) => {
    const medals = ['🥇', '🥈', '🥉'];
    const entry = el('div', { class: 'record-entry' }, [
      el('span', { class: 'record-rank', text: medals[i] || `#${i + 1}` }),
      el('div', { style: { flex: '1' } }, [
        el('div', { text: r.name, style: { fontWeight: '600' } }),
        el('div', { text: `Day 7 | ${r.difficulty}`, style: { fontSize: '0.8rem', color: 'var(--text-muted)' } }),
      ]),
      el('div', { text: `⭐ ${formatFameExact(r.fame)}`, style: { fontFamily: 'var(--font-mono)', color: 'var(--fame-gold)' } }),
    ]);
    container.appendChild(entry);
  });
}

/* ===== HOW TO PLAY ===== */
export function renderHowToPlay() {
  const container = $('#howto-steps');
  container.innerHTML = '';

  const steps = [
    { title: 'Daily Schedule', text: 'Each day has <strong>8 time slots</strong>. Morning briefing, 5 work slots, a lunch break, and evening free time.' },
    { title: 'Work Tasks', text: 'Choose from <strong>photo shootings, safety courses, crew interviews, and social media tasks</strong>. Each earns fame and money.' },
    { title: 'Assign Your Team', text: 'Your team of 4 has different skills. <strong>Match the right person to the right task</strong> for bonus fame.' },
    { title: 'Mini-Games', text: 'Tasks trigger <strong>interactive mini-games</strong>. Your score multiplies the fame earned!' },
    { title: 'Lunch Break', text: 'Every day you eat lunch. <strong>Food restores energy and happiness</strong>. Treat your team to keep morale high.' },
    { title: 'Dating', text: 'Meet and romance <strong>8 unique crew members</strong>. Build love through choices and dates. Gifts unlock bonus content!' },
    { title: 'Equipment', text: 'Buy <strong>equipment upgrades</strong> to boost fame from specific task types. Invest wisely!' },
    { title: 'Win Condition', text: 'Reach <strong>1,000,000 followers</strong> (fame) by Day 7. Higher fame = better ending. Max love = romance ending!' },
  ];

  steps.forEach((step, i) => {
    const div = el('div', { class: 'howto-step' }, [
      el('div', { class: 'step-num', text: i + 1 }),
      el('div', { class: 'step-text', html: `<strong>${step.title}:</strong> ${step.text}` }),
    ]);
    container.appendChild(div);
  });
}

/* ===== TOAST NOTIFICATIONS ===== */
export function showToast(message, type = 'info') {
  const container = $('#toast-container');
  const toast = el('div', { class: `toast ${type}`, text: message });
  container.appendChild(toast);
  setTimeout(() => toast.remove(), 3000);
}

/* ===== FAME POPUP ===== */
export function showFamePopup(amount, x, y) {
  const container = $('#fame-popup-container');
  const popup = el('div', {
    class: `fame-popup ${amount >= 0 ? 'positive' : 'negative'}`,
    text: `${amount >= 0 ? '+' : ''}${formatFame(amount)} ⭐`,
    style: { left: `${x || 50}%`, top: `${y || 50}%` }
  });
  container.appendChild(popup);
  setTimeout(() => popup.remove(), 1500);
}

/* ===== EVENING OPTIONS ===== */
export function showEveningOptions(onDate, onRest) {
  const s = getState();
  const dateable = getDateableCharacters(s);

  if (dateable.length === 0) {
    showToast('No date options available tonight. Resting...', 'info');
    onRest();
    return;
  }

  const area = $('#task-actions');
  area.innerHTML = '';

  const grid = el('div', { class: 'task-choice-grid' });

  dateable.forEach(c => {
    const cs = s.characters[c.id];
    const card = el('div', { class: 'task-choice-card', onclick: () => onDate(c.id) }, [
      el('div', { class: 'tcc-icon', text: c.emoji }),
      el('div', { class: 'tcc-name', text: `Date: ${c.name}` }),
      el('div', { class: 'tcc-reward', text: `💕 Love: ${cs.love}` }),
    ]);
    grid.appendChild(card);
  });

  const restCard = el('div', { class: 'task-choice-card', onclick: onRest }, [
    el('div', { class: 'tcc-icon', text: '😴' }),
    el('div', { class: 'tcc-name', text: 'Rest & End Day' }),
    el('div', { class: 'tcc-reward', text: 'Energy +20' }),
  ]);
  grid.appendChild(restCard);

  area.appendChild(grid);
}
