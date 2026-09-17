// Cached DOM references — resolved once at module load (main.js is a
// deferred module, so the DOM is fully parsed by then).

export const els = {
  setupScreen: document.getElementById('setup-screen'),
  gameScreen: document.getElementById('game-screen'),
  endScreen: document.getElementById('end-screen'),

  playerGender: document.getElementById('player-gender'),
  interestGender: document.getElementById('interest-gender'),
  startBtn: document.getElementById('start-btn'),
  restartBtn: document.getElementById('restart-btn'),
  playAgainBtn: document.getElementById('play-again-btn'),

  chapterLabel: document.getElementById('chapter-label'),
  characterName: document.getElementById('character-name'),
  characterMeta: document.getElementById('character-meta'),
  dialogueText: document.getElementById('dialogue-text'),
  avatar: document.getElementById('avatar'),

  timerText: document.getElementById('timer-text'),
  timerFill: document.getElementById('timer-fill'),

  scoreValue: document.getElementById('score-value'),
  streakValue: document.getElementById('streak-value'),
  secretValue: document.getElementById('secret-value'),
  bestValue: document.getElementById('best-value'),

  choices: document.getElementById('choices'),

  endBadge: document.getElementById('end-badge'),
  endTitle: document.getElementById('end-title'),
  endText: document.getElementById('end-text'),
  finalScore: document.getElementById('final-score'),
  finalStreak: document.getElementById('final-streak'),
  finalSecret: document.getElementById('final-secret'),
  shareBtn: document.getElementById('share-btn'),

  historyOverlay: document.getElementById('history-overlay'),
  historyList: document.getElementById('history-list'),
  historyClose: document.getElementById('history-close')
};
