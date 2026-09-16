/* main.js — Game initialization, startGame, howToPlay, global key/touch handlers */
window.G = window.G || {}

/** Initialize and start a new game */
G.startGame = function (difficulty) {
  G.initAudio()
  G.newGame(difficulty)
  G.updateHUD()
  G.sfx.click()
  G.showScreen('screen-map')
  if (window.lucide) lucide.createIcons()
  if (!localStorage.getItem('concerts-tutorial-done')) G.startTutorial()
}

document.addEventListener('DOMContentLoaded', function () {
  if (window.lucide) lucide.createIcons()
})

/** Show the How to Play modal */
G.showHowToPlay = function () {
  G.sfx.click()
  G.showModal('modal-howto')
}

/** Resume AudioContext on first user interaction (required by browsers) */
;(function () {
  var resumed = false
  function resumeAudio() {
    if (resumed) return
    resumed = true
    if (!G.audioCtx) G.initAudio()
    if (G.audioCtx && G.audioCtx.state === 'suspended') {
      G.audioCtx.resume()
    }
  }
  document.addEventListener('click', resumeAudio, { once: false })
  document.addEventListener('touchstart', resumeAudio, { once: false })
  document.addEventListener('keydown', resumeAudio, { once: false })
  document.addEventListener('keydown', function (e) {
    if (e.key === 'Escape' && G.paused) {
      G.togglePause()
    }
  })
})()

/** Prevent default on spacebar to avoid page scroll during minigames */
document.addEventListener('keydown', function (e) {
  if (e.code === 'Space' && G.currentScreen && G.currentScreen !== 'screen-menu') {
    e.preventDefault()
  }
})

/** Prevent pull-to-refresh and bounce on iOS */
document.addEventListener(
  'touchmove',
  function (e) {
    if (e.target.closest('.gossip-feed, .screen, .modal-box')) return
    // Allow scrollable containers
  },
  { passive: true }
)

/** Global keyboard shortcuts */
document.addEventListener('keydown', function (e) {
  // Shortcuts overlay (only on game screens)
  if (e.key === '?' && G.currentScreen && G.currentScreen.startsWith('screen-') && !G.paused) {
    G.showModal('modal-shortcuts')
    return
  }
  // Pause toggle or close modals on Escape
  if (e.key === 'Escape') {
    if (G.paused) {
      G.togglePause()
    } else if (G.currentScreen && G.currentScreen.startsWith('screen-')) {
      G.closeAllModals()
    }
    return
  }
  // Merch shop shortcut (M key from map screen)
  if (e.key === 'm' || e.key === 'M') {
    if (G.currentScreen === 'screen-map') {
      G.showMerch()
      return
    }
  }
  // Photo mode shortcut (P key during concert)
  if (e.key === 'p' || e.key === 'P') {
    if (G.currentScreen === 'screen-concert' && G._photoCanvas) {
      G.photoMode(G._photoConcertId, G._photoCanvas, function () {
        G.showScreen('screen-concert')
      })
      return
    }
  }
})
