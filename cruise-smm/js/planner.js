/* ===== DAY / TIME PLANNER ===== */
import { getState } from './state.js';
import { getTimeLabel, getSlotName, getSlotIcon } from './utils.js';

export const SLOTS = [
  { index: 0, time: '08:00', name: 'Morning Briefing', type: 'auto', icon: '☀️' },
  { index: 1, time: '09:00', name: 'Slot 1', type: 'work', icon: '📋' },
  { index: 2, time: '10:30', name: 'Slot 2', type: 'work', icon: '📋' },
  { index: 3, time: '12:00', name: 'Lunch Break', type: 'lunch', icon: '🍽️' },
  { index: 4, time: '13:30', name: 'Slot 3', type: 'work', icon: '📋' },
  { index: 5, time: '15:00', name: 'Slot 4', type: 'work', icon: '📋' },
  { index: 6, time: '16:30', name: 'Slot 5', type: 'work', icon: '📋' },
  { index: 7, time: '18:00', name: 'Evening Free Time', type: 'evening', icon: '🌅' },
];

export function initDaySlots(state) {
  state.daySlots = SLOTS.map(s => ({
    ...s,
    status: s.index === 0 ? 'active' : 'locked',
    taskId: null,
    result: null,
  }));
  state.currentSlot = 0;
}

export function getCurrentSlot(state) {
  return state.daySlots[state.currentSlot] || null;
}

export function advanceSlot(state) {
  if (state.currentSlot < state.daySlots.length) {
    state.daySlots[state.currentSlot].status = 'completed';
  }
  state.currentSlot++;
  if (state.currentSlot < state.daySlots.length) {
    state.daySlots[state.currentSlot].status = 'active';
    return state.daySlots[state.currentSlot];
  }
  return null; // Day over
}

export function isLunchSlot(state) {
  const slot = getCurrentSlot(state);
  return slot && slot.type === 'lunch';
}

export function isEveningSlot(state) {
  const slot = getCurrentSlot(state);
  return slot && slot.type === 'evening';
}

export function isBriefingSlot(state) {
  const slot = getCurrentSlot(state);
  return slot && slot.type === 'auto';
}

export function isWorkSlot(state) {
  const slot = getCurrentSlot(state);
  return slot && slot.type === 'work';
}

export function isDayOver(state) {
  return state.currentSlot >= state.daySlots.length;
}

export function setSlotTask(state, taskId, label) {
  const slot = getCurrentSlot(state);
  if (slot) {
    slot.taskId = taskId;
    if (label) slot.name = label;
  }
}

export function setSlotResult(state, result) {
  const slot = getCurrentSlot(state);
  if (slot) {
    slot.result = result;
  }
}

export function getCompletedSlotCount(state) {
  return state.daySlots.filter(s => s.status === 'completed').length;
}

export function getDayProgress(state) {
  return state.currentSlot / state.daySlots.length;
}
