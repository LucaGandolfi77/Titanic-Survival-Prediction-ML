/* main.js — Game initialization, startGame, howToPlay, global key/touch handlers */
window.G = window.G || {};

/** Initialize and start a new game */
G.startGame = function() {
  G.initAudio();
  G.newGame();
  G.updateHUD();
  G.sfx.click();
  G.showScreen('screen-map');
};

/** Show the How to Play modal */
G.showHowToPlay = function() {
  G.sfx.click();
  G.showModal('modal-howto');
};

/** Resume AudioContext on first user interaction (required by browsers) */
(function() {
  var resumed = false;
  function resumeAudio() {
    if (resumed) return;
    resumed = true;
    if (!G.audioCtx) G.initAudio();
    if (G.audioCtx && G.audioCtx.state === 'suspended') {
      G.audioCtx.resume();
    }
  }
  document.addEventListener('click', resumeAudio, { once: false });
  document.addEventListener('touchstart', resumeAudio, { once: false });
  document.addEventListener('keydown', resumeAudio, { once: false });
})();

/** Prevent default on spacebar to avoid page scroll during minigames */
document.addEventListener('keydown', function(e) {
  if (e.code === 'Space' && G.currentScreen && G.currentScreen !== 'screen-menu') {
    e.preventDefault();
  }
});

/** Prevent pull-to-refresh and bounce on iOS */
document.addEventListener('touchmove', function(e) {
  if (e.target.closest('.gossip-feed, .screen, .modal-box')) return;
  // Allow scrollable containers
}, { passive: true });
