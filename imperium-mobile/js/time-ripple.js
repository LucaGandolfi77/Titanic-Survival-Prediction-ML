import { showToast, haptic } from './ui/notifications.js';
import { gameSpeed } from './main.js';
import { resources } from './resource.js';

export const timeRipple = {
  state: 'normal',
  timer: 0,
  duration: 10,
  cooldown: 30,
  cooldownTimer: 0,
  isActive: false,

  canUse() {
    if (this.isActive) return false;
    if (this.cooldownTimer > 0) return false;
    return true;
  },

  use(type) {
    if (!this.canUse()) {
      showToast('Time Ripple on cooldown');
      return false;
    }
    if (resources.gold < (type === 'accelerate' ? 50 : 75)) {
      showToast('Need gold for Time Ripple');
      return false;
    }
    resources.spend({ gold: type === 'accelerate' ? 50 : 75 });
    this.state = type;
    this.timer = this.duration;
    this.isActive = true;
    haptic(40);
    showToast(type === 'accelerate' ? '⚡ Time Accelerated!' : type === 'slow' ? '🕰️ Time Slowed!' : '❄️ Time Frozen!');
  },

  update(dt) {
    if (this.isActive) {
      this.timer -= dt;
      if (this.timer <= 0) {
        this.isActive = false;
        this.cooldownTimer = this.cooldown;
        this.state = 'normal';
        showToast('Time Ripple ended');
      }
    }
    if (this.cooldownTimer > 0) {
      this.cooldownTimer = Math.max(0, this.cooldownTimer - dt);
    }
  },

  getSpeedMultiplier() {
    if (!this.isActive) return 1;
    switch (this.state) {
      case 'accelerate': return 2;
      case 'slow': return 0.5;
      case 'freeze': return 0;
      default: return 1;
    }
  },

  getCooldownPercent() {
    return this.cooldownTimer / this.cooldown;
  },
};
