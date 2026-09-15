export class GameState {
  constructor() {
    this.phase = 'menu';
    this.exitTriggered = false;
    this.debugMode = false;
    this.currentRoomTheme = null;
    this.sanity = 100;
    this.flashlightBattery = 100;
  }

  setPhase(phase) {
    this.phase = phase;
  }

  isPlaying() {
    return this.phase === 'playing';
  }

  isPaused() {
    return this.phase === 'paused';
  }

  isGameOver() {
    return this.phase === 'gameover';
  }

  isWin() {
    return this.phase === 'win';
  }

  modifySanity(currentSanity, amount) {
    this.sanity = Math.max(-100, Math.min(100, currentSanity + amount));
    return this.sanity;
  }

  setSanity(value) {
    this.sanity = Math.max(-100, Math.min(100, value));
  }

  modifyBattery(currentBattery, amount) {
    this.flashlightBattery = Math.max(0, Math.min(100, currentBattery + amount));
    return this.flashlightBattery;
  }

  reset() {
    this.exitTriggered = false;
    this.sanity = 100;
    this.flashlightBattery = 100;
    this.currentRoomTheme = null;
  }
}
