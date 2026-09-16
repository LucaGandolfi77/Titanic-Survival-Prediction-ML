/* tutorial.js — Step-by-step onboarding for first-time players */
window.G = window.G || {}

G.tutorialStep = 0

G.startTutorial = function () {
  if (localStorage.getItem('concerts-tutorial-done')) return
  G.tutorialStep = 1
  var overlay = document.getElementById('tutorial-overlay')
  var card = document.getElementById('tutorial-card')
  var spotlight = document.getElementById('tutorial-spotlight')
  if (!overlay || !card) return
  overlay.style.display = 'flex'
  spotlight.style.display = 'none'
  card.style.display = 'block'
  card.style.left = '50%'
  card.style.top = '50%'
  card.style.transform = 'translate(-50%, -50%)'
  document.getElementById('tutorial-title').textContent = 'Welcome to Concerts Chase! 🎵'
  document.getElementById('tutorial-text').textContent =
    'Discover concerts around the world, book flights, and rock out. Let us show you how it works.'
  document.getElementById('tutorial-actions').innerHTML =
    '<button class="btn btn-primary" onclick="G.tutorialNext()">Start Tour</button>'
}

G.tutorialNext = function () {
  G.tutorialStep++
  if (G.tutorialStep === 2) {
    G.showTutorialStep()
  } else if (G.tutorialStep === 3) {
    G.showTutorialStep()
  } else {
    G.tutorialComplete()
  }
}

G.tutorialSkip = function () {
  G.tutorialStep = 0
  G.tutorialComplete()
}

G.tutorialComplete = function () {
  localStorage.setItem('concerts-tutorial-done', 'done')
  G.tutorialStep = 0
  var overlay = document.getElementById('tutorial-overlay')
  if (overlay) overlay.style.display = 'none'
}

G.showTutorialStep = function () {
  var overlay = document.getElementById('tutorial-overlay')
  var spotlight = document.getElementById('tutorial-spotlight')
  var card = document.getElementById('tutorial-card')
  var title = document.getElementById('tutorial-title')
  var text = document.getElementById('tutorial-text')
  var actions = document.getElementById('tutorial-actions')
  if (!overlay || !card) return
  overlay.style.display = 'flex'
  card.style.display = 'block'
  if (G.tutorialStep === 2) {
    if (spotlight) spotlight.style.display = 'block'
    card.style.transform = 'none'
    title.textContent = '🔍 Discover Concerts'
    text.textContent = 'Tap the ❓ markers on the map to reveal concert dates and artists'
    actions.innerHTML = '<button class="btn btn-primary" onclick="G.tutorialNext()">Got it!</button>'
    G.positionSpotlight()
  } else if (G.tutorialStep === 3) {
    if (spotlight) spotlight.style.display = 'none'
    card.style.left = '50%'
    card.style.top = '50%'
    card.style.transform = 'translate(-50%, -50%)'
    title.textContent = "🎸 You're Ready!"
    text.textContent = 'Discover concerts by tapping ❓ markers. Book early for cheaper flights!'
    actions.innerHTML = '<button class="btn btn-primary" onclick="G.tutorialComplete()">Start Playing</button>'
  }
}

G.positionSpotlight = function () {
  var hits = document.querySelectorAll('.map-hit')
  var target = null
  for (var i = 0; i < hits.length; i++) {
    var cid = parseInt(hits[i].getAttribute('data-cid'))
    if (!G.state.discovered.has(cid) && !G.state.expired.has(cid)) {
      target = hits[i]
      break
    }
  }
  if (!target) {
    G.tutorialStep = 3
    G.showTutorialStep()
    return
  }
  var rect = target.getBoundingClientRect()
  var spotlight = document.getElementById('tutorial-spotlight')
  var card = document.getElementById('tutorial-card')
  if (spotlight) {
    spotlight.style.left = rect.left + rect.width / 2 - 25 + 'px'
    spotlight.style.top = rect.top + rect.height / 2 - 25 + 'px'
    spotlight.style.width = '50px'
    spotlight.style.height = '50px'
  }
  if (card) {
    var cardLeft = rect.right + 16
    var cardTop = Math.max(10, rect.top - 20)
    var viewportW = window.innerWidth || 800
    if (cardLeft + 280 > viewportW) {
      cardLeft = Math.max(10, rect.left - 280)
    }
    card.style.left = cardLeft + 'px'
    card.style.top = cardTop + 'px'
    card.style.transform = 'none'
  }
}
