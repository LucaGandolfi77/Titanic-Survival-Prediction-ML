export class UIManager {
  constructor(game) {
    this.game = game;
    this.currentScreen = 'menu';
    
    // Bind button listeners
    this.setupMenuListeners();
    this.setupSettingsListeners();
  }

  setupMenuListeners() {
    const btnPlay = document.getElementById('btn-play');
    const btnHowto = document.getElementById('btn-howto');
    const btnSettings = document.getElementById('btn-settings');
    const btnHowtoBack = document.getElementById('btn-howto-back');
    const btnSetBack = document.getElementById('btn-set-back');
    const btnResume = document.getElementById('btn-resume');
    const btnRestart = document.getElementById('btn-restart');
    const btnPauseMenu = document.getElementById('btn-pause-menu');
    const btnPlayAgain = document.getElementById('btn-play-again');
    const btnWinMenu = document.getElementById('btn-win-menu');
    const btnTryAgain = document.getElementById('btn-tryagain');
    const btnGoMenu = document.getElementById('btn-go-menu');
    const btnCloseNote = document.getElementById('btn-close-note');

    if (btnPlay) btnPlay.addEventListener('click', () => this.startGame());
    if (btnHowto) btnHowto.addEventListener('click', () => this.showHowto());
    if (btnSettings) btnSettings.addEventListener('click', () => this.showSettings());
    if (btnHowtoBack) btnHowtoBack.addEventListener('click', () => this.showMenu());
    if (btnSetBack) btnSetBack.addEventListener('click', () => this.showMenu());
    if (btnResume) btnResume.addEventListener('click', () => this.resumeGame());
    if (btnRestart) btnRestart.addEventListener('click', () => this.restartGame());
    if (btnPauseMenu) btnPauseMenu.addEventListener('click', () => this.showMenu());
    if (btnPlayAgain) btnPlayAgain.addEventListener('click', () => this.restartGame());
    if (btnWinMenu) btnWinMenu.addEventListener('click', () => this.showMenu());
    if (btnTryAgain) btnTryAgain.addEventListener('click', () => this.restartGame());
    if (btnGoMenu) btnGoMenu.addEventListener('click', () => this.showMenu());
    if (btnCloseNote) btnCloseNote.addEventListener('click', () => this.closeNote());
  }

  setupSettingsListeners() {
    const sensSlider = document.getElementById('sens-slider');
    const btnSound = document.getElementById('btn-sound');
    const btnDebug = document.getElementById('btn-debug');

    if (sensSlider) {
      sensSlider.addEventListener('change', (e) => {
        if (this.game.player) {
          this.game.player.mouseSensitivity = e.target.value / 500;
        }
      });
    }

    if (btnSound) {
      btnSound.addEventListener('click', () => {
        const enabled = !btnSound.classList.contains('active');
        btnSound.classList.toggle('active');
        if (this.game.audio) {
          this.game.audio.setEnabled(enabled);
        }
      });
    }

    if (btnDebug) {
      btnDebug.addEventListener('click', () => {
        btnDebug.classList.toggle('active');
        if (this.game.controls) {
          this.game.debugMode = btnDebug.classList.contains('active');
        }
      });
    }
  }

  startGame() {
    this.hideAllScreens();
    this.game.startGame();
  }

  resumeGame() {
    this.hideAllScreens();
    this.game.resumeGame();
  }

  restartGame() {
    this.hideAllScreens();
    this.game.restartGame();
  }

  showMenu() {
    this.hideAllScreens();
    this.showScreen('screen-menu');
    this.game.pauseGame();
  }

  showPause() {
    this.hideAllScreens();
    this.showScreen('screen-pause');
    this.updatePauseInfo();
  }

  showHowto() {
    this.hideAllScreens();
    this.showScreen('screen-howto');
    this.populateHandbook();
  }

  showSettings() {
    this.hideAllScreens();
    this.showScreen('screen-settings');
  }

  showWin(stats) {
    this.hideAllScreens();
    this.showScreen('screen-win');
    this.populateWinScreen(stats);
  }

  showGameOver(reason) {
    this.hideAllScreens();
    this.showScreen('screen-gameover');
    this.populateGameOverScreen(reason);
  }

  showNote(title, content) {
    this.hideAllScreens();
    this.showScreen('screen-note');
    document.getElementById('note-title').textContent = title;
    document.getElementById('note-body').textContent = content;
  }

  closeNote() {
    this.showScreen('screen-menu');
  }

  hideAllScreens() {
    const screens = document.querySelectorAll('.screen');
    screens.forEach(s => s.classList.add('hidden'));
  }

  showScreen(id) {
    const screen = document.getElementById(id);
    if (screen) {
      screen.classList.remove('hidden');
      screen.classList.add('active');
    }
  }

  populateHandbook() {
    const content = document.getElementById('howto-content');
    content.innerHTML = `
      <div class="howto-section">
        <h3>OBJECTIVE</h3>
        <p>File Form 27-Γ at the EXIT DESK in EXIT HALL (Room 11) to terminate your employment and escape the Bureau.</p>
      </div>
      
      <div class="howto-section">
        <h3>CONTROLS</h3>
        <p><strong>Movement:</strong> WASD or arrow keys</p>
        <p><strong>Look Around:</strong> Mouse (click to lock cursor)</p>
        <p><strong>Interact:</strong> E key (near objects)</p>
        <p><strong>Crouch:</strong> C key (for tight spaces)</p>
        <p><strong>Flashlight:</strong> F key</p>
        <p><strong>Inventory:</strong> I key</p>
        <p><strong>Pause:</strong> P or ESC</p>
      </div>
      
      <div class="howto-section">
        <h3>REQUIRED ITEMS</h3>
        <p>To exit, collect:</p>
        <p>📄 Form 27-Γ - Records Room (cabine #3)</p>
        <p>🔑 Cabinet Key - Void Office</p>
        <p>🔴 Rubber Stamp - Void Office</p>
        <p>☕ Director's Coffee - Break Room</p>
        <p>📋 Meeting Minutes - Conference Room</p>
      </div>
      
      <div class="howto-section">
        <h3>SANITY SYSTEM</h3>
        <p>Sanity decreases from: loop portals, void proximity, upside-down rooms, NPC interaction.</p>
        <p>Recover by: collecting coffee, using correct portals, stamping forms correctly.</p>
        <p>At 0 sanity: hallucinations begin. Below -50: The Auditor appears.</p>
      </div>
      
      <div class="howto-section">
        <h3>PORTALS</h3>
        <p><strong>Normal (Blue):</strong> Standard portal connections</p>
        <p><strong>Upside Down (Red):</strong> Inverts gravity</p>
        <p><strong>Sideways (Orange):</strong> Rotates gravity 90°</p>
        <p><strong>Loop (Green):</strong> Connects room to itself (sanity -5)</p>
        <p><strong>Mirror (Blue):</strong> Destination is horizontally flipped</p>
        <p><strong>VOID (Black):</strong> DO NOT ENTER - instant game over</p>
      </div>
    `;
  }

  updatePauseInfo() {
    const info = document.getElementById('pause-room-info');
    if (this.game.player && this.game.player.currentRoom) {
      const roomName = `Room ${this.game.player.currentRoom.id}`;
      info.textContent = `Current Location: ${roomName}\nSanity: ${Math.round(this.game.player.sanity)} / 100`;
    }
  }

  populateWinScreen(stats) {
    const statsEl = document.getElementById('win-stats');
    const msgEl = document.getElementById('win-message');
    
    statsEl.innerHTML = `
      <p>Employee #4471-B</p>
      <p>Sanity Remaining: ${Math.round(this.game.player.sanity)}</p>
      <p>Rooms Explored: ${this.game.puzzle.completedObjectives.length} / 12</p>
      <p>Items Collected: ${this.game.puzzle.inventory.length} / 8</p>
    `;
    
    const hasSecret = this.game.puzzle.hasSecretEnding();
    if (hasSecret) {
      msgEl.innerHTML = `
        <p>YOU HAVE UNLOCKED THE SECRET ENDING!</p>
        <p>Greg's Portal Remote was discovered in the supply closet.</p>
        <p>The building's portals have been reset to correct installations.</p>
        <p>You and Greg escape into the mundane afternoon.</p>
      `;
    } else {
      msgEl.innerHTML = `
        <p>You have filed Form 27-Γ and been formally terminated.</p>
        <p>You walk out into the parking lot, squinting in the sunlight.</p>
        <p>Freedom feels strange.</p>
      `;
    }
  }

  populateGameOverScreen(reason) {
    const content = document.getElementById('gameover-content');
    content.innerHTML = `
      <p>REASON FOR TERMINATION:</p>
      <p style="color: #ff4444; margin-top: 1rem;">${reason}</p>
      <p style="margin-top: 1rem; opacity: 0.7;">Your employment has been permanently revoked.</p>
    `;
  }

  showNotification(title, type = 'info') {
    const container = document.getElementById('event-notifications');
    const notif = document.createElement('div');
    notif.className = `event-notification ${type}`;
    notif.textContent = title;

    container.appendChild(notif);
    setTimeout(() => notif.remove(), 2500);
  }

  showNPCDialog(position, text) {
    // Could show NPC speech bubbles overlaid on screen
    this.showNotification('NPC: ' + text, 'warning');
  }

  showItemExamine(item) {
    this.hideAllScreens();
    this.showScreen('screen-note');
    document.getElementById('note-title').textContent = item.type.toUpperCase();
    document.getElementById('note-body').textContent = item.getDescription();
  }
}