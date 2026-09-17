import { FACTION_KEYS, FACTIONS } from '../data/factions.js';
import { showToast, haptic } from './ui/notifications.js';
import { rand } from '../utils/helpers.js';
import { units } from './unit.js';
import { mapState } from './map.js';
import { hexToPixelCenter } from '../utils/hex-math.js';
import { ai } from './ai.js';

const RELATION_STATES = ['allied', 'friendly', 'neutral', 'tense', 'at_war'];

export const diplomacy = {
  relations: {},
  reputation: {},
  events: [],
  eventTimer: 0,
  eventInterval: 45,
  lastEvent: 0,

  init() {
    this.relations = {};
    this.reputation = {};

    for (const f1 of FACTION_KEYS) {
      this.relations[f1] = {};
      this.reputation[f1] = 0;
      for (const f2 of FACTION_KEYS) {
        if (f1 === f2) {
          this.relations[f1][f2] = 'self';
        } else {
          this.relations[f1][f2] = 'neutral';
        }
      }
    }

    this.relations.roma.vikings = 'at_war';
    this.relations.roma.mongols = 'tense';
    this.relations.vikings.egyptians = 'friendly';
    this.relations.celts.japanese = 'tense';
    this.relations.mongols.egyptians = 'neutral';
  },

  getRelation(f1, f2) {
    return this.relations[f1]?.[f2] || 'neutral';
  },

  setRelation(f1, f2, state) {
    if (!RELATION_STATES.includes(state) && state !== 'self') return;
    if (f1 === f2) return;
    this.relations[f1][f2] = state;
    this.relations[f2][f1] = state;
    this.logEvent(`${FACTIONS[f1].emoji} ${FACTIONS[f1].name} ${state.replace('_', ' ')} ${FACTIONS[f2].name}`);
    haptic(15);
  },

  adjustReputation(faction, amount) {
    this.reputation[faction] = Math.max(-100, Math.min(100, (this.reputation[faction] || 0) + amount));
  },

  canAction(f1, f2, action) {
    const rel = this.getRelation(f1, f2);
    switch (action) {
      case 'declare_war':
        if (rel === 'allied') return false;
        if (rel === 'at_war') return false;
        return true;
      case 'peace':
        if (rel === 'neutral') return false;
        return true;
      case 'ally':
        if (rel !== 'neutral' && rel !== 'friendly' && rel !== 'tense') return false;
        return true;
      case 'trade':
        if (rel === 'at_war' || rel === 'tense') return false;
        return true;
      case 'nap':
        if (rel === 'allied' || rel === 'friendly' || rel === 'neutral') return true;
        return true;
      default:
        return true;
    }
  },

  performAction(f1, f2, action) {
    if (!this.canAction(f1, f2, action)) {
      showToast('Cannot perform this action');
      return false;
    }

    switch (action) {
      case 'declare_war':
        this.setRelation(f1, f2, 'at_war');
        this.adjustReputation(f1, -10);
        break;
      case 'peace':
        this.setRelation(f1, f2, 'neutral');
        this.adjustReputation(f1, 5);
        break;
      case 'ally':
        this.setRelation(f1, f2, 'allied');
        this.adjustReputation(f1, 15);
        break;
      case 'trade':
        this.setRelation(f1, f2, 'friendly');
        this.adjustReputation(f1, 8);
        break;
      case 'nap':
        this.setRelation(f1, f2, 'friendly');
        this.adjustReputation(f1, 3);
        break;
    }
    return true;
  },

  logEvent(msg) {
    this.events.unshift({ msg, time: Date.now() });
    if (this.events.length > 20) this.events.pop();
  },

  update(dt) {
    this.eventTimer += dt;
    if (this.eventTimer >= this.eventInterval) {
      this.eventTimer = 0;
      this.triggerRandomEvent();
    }
  },

  triggerRandomEvent() {
    const f1 = FACTION_KEYS[rand(0, FACTION_KEYS.length - 1)];
    const others = FACTION_KEYS.filter(f => f !== f1);
    const f2 = others[rand(0, others.length - 1)];
    const rel = this.getRelation(f1, f2);

    if (rel === 'at_war' && Math.random() < 0.4) {
      if (Math.random() < 0.5) {
        this.logEvent(`🕊️ ${FACTIONS[f1].name} proposes peace with ${FACTIONS[f2].name}`);
        showToast(`🕊️ ${FACTIONS[f1].name} seeks peace`);
      } else {
        this.setRelation(f1, f2, 'tense');
        this.logEvent(`⚔️ ${FACTIONS[f1].name} and ${FACTIONS[f2].name} escalated`);
        showToast(`⚔️ Tensions between ${FACTIONS[f1].name} and ${FACTIONS[f2].name}`);
      }
    } else if (rel === 'neutral' && Math.random() < 0.3) {
      if (Math.random() < 0.5) {
        this.setRelation(f1, f2, 'tense');
        this.logEvent(`⚠️ ${FACTIONS[f1].name} and ${FACTIONS[f2].name} are now tense`);
      } else {
        this.setRelation(f1, f2, 'friendly');
        this.logEvent(`🤝 ${FACTIONS[f1].name} and ${FACTIONS[f2].name} became friendly`);
      }
    } else if (rel === 'friendly' && Math.random() < 0.2) {
      this.setRelation(f1, f2, 'allied');
      this.logEvent(`🌟 ${FACTIONS[f1].name} and ${FACTIONS[f2].name} formed an alliance`);
      showToast(`🌟 Alliance: ${FACTIONS[f1].name} ↔ ${FACTIONS[f2].name}`);
    }

    if (Math.random() < 0.15) {
      this.adjustReputation(f1, rand(-3, 5));
    }
  },

  getEventLog() {
    return this.events;
  },
};
