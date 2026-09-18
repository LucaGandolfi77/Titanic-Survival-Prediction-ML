const CHARACTERS = [
  {
    id: 'jake',
    name: 'Jake',
    emoji: '🏃',
    color: 0x2196f3,
    hatColor: 0xff9800,
    ability: null,
    abilityDesc: 'None',
    unlockScore: 0
  },
  {
    id: 'tricky',
    name: 'Tricky',
    emoji: '🎩',
    color: 0x4caf50,
    hatColor: 0x9c27b0,
    ability: 'magnet',
    abilityDesc: '+2s Magnet',
    unlockScore: 10000
  },
  {
    id: 'fresh',
    name: 'Fresh',
    emoji: '🎧',
    color: 0xff5722,
    hatColor: 0x212121,
    ability: 'coin_boost',
    abilityDesc: '2x Coins 5s',
    unlockScore: 50000
  },
  {
    id: 'spike',
    name: 'Spike',
    emoji: '🎸',
    color: 0x795548,
    hatColor: 0xf44336,
    ability: 'speed',
    abilityDesc: 'Speed +10%',
    unlockScore: 100000
  },
  {
    id: 'yolanda',
    name: 'Yolanda',
    emoji: '💃',
    color: 0xe91e63,
    hatColor: 0xffeb3b,
    ability: 'auto_shield',
    abilityDesc: 'Shield/30s',
    unlockScore: 250000
  },
  {
    id: 'king',
    name: 'King',
    emoji: '👑',
    color: 0xffd600,
    hatColor: 0xd500f9,
    ability: 'triple_score',
    abilityDesc: '3x Score',
    unlockScore: 500000
  }
];

const CHARACTER_KEY = 'subwayRunner_selectedCharacter';
const UNLOCK_KEY = 'subwayRunner_unlockedCharacters';

export class CharacterManager {
  constructor() {
    this.characters = CHARACTERS;
    this.selectedId = this._loadSelected();
    this.unlocked = this._loadUnlocked();
  }

  getSelected() {
    return this.characters.find(c => c.id === this.selectedId) || this.characters[0];
  }

  select(id) {
    if (this.characters.find(c => c.id === id)) {
      this.selectedId = id;
      this._saveSelected();
    }
  }

  isUnlocked(id) {
    const char = this.characters.find(c => c.id === id);
    if (!char) return false;
    if (char.unlockScore === 0) return true;
    return this.unlocked.includes(id);
  }

  checkUnlocks(totalScore) {
    const newlyUnlocked = [];
    for (const char of this.characters) {
      if (char.unlockScore > 0 && !this.unlocked.includes(char.id) && totalScore >= char.unlockScore) {
        this.unlocked.push(char.id);
        newlyUnlocked.push(char);
      }
    }
    if (newlyUnlocked.length > 0) {
      this._saveUnlocked();
    }
    return newlyUnlocked;
  }

  getAll() {
    return this.characters.map(c => ({
      ...c,
      unlocked: this.isUnlocked(c.id),
      selected: c.id === this.selectedId
    }));
  }

  renderSelection(container) {
    container.innerHTML = '';
    for (const char of this.characters) {
      const card = document.createElement('div');
      card.className = 'character-card';
      if (char.id === this.selectedId) card.classList.add('selected');
      if (!this.isUnlocked(char.id)) card.classList.add('locked');

      card.innerHTML = `
        <div class="char-avatar">${char.emoji}</div>
        <div class="char-name">${char.name}</div>
      `;

      card.addEventListener('click', () => {
        if (!this.isUnlocked(char.id)) return;
        this.select(char.id);
        this.renderSelection(container);
      });

      container.appendChild(card);
    }
  }

  applyAbility(player, collectibleManager) {
    const char = this.getSelected();
    if (!char.ability) return;

    switch (char.ability) {
      case 'coin_boost':
        collectibleManager.coinValue = 2;
        break;
      case 'speed':
        return 1.1;
      case 'auto_shield':
        if (!collectibleManager.hasPowerup('shield') && Math.random() < 0.01) {
          collectibleManager.collectPowerup({ userData: { type: 'shield' }, visible: false });
        }
        break;
    }
    return 1.0;
  }

  resetAbility(collectibleManager) {
    collectibleManager.coinValue = 1;
  }

  _loadSelected() {
    try { return localStorage.getItem(CHARACTER_KEY) || 'jake'; } catch { return 'jake'; }
  }

  _saveSelected() {
    try { localStorage.setItem(CHARACTER_KEY, this.selectedId); } catch {}
  }

  _loadUnlocked() {
    try { return JSON.parse(localStorage.getItem(UNLOCK_KEY)) || ['jake']; } catch { return ['jake']; }
  }

  _saveUnlocked() {
    try { localStorage.setItem(UNLOCK_KEY, JSON.stringify(this.unlocked)); } catch {}
  }
}

export { CHARACTERS };
