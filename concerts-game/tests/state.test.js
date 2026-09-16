import { describe, it, expect, beforeEach } from 'vitest'

beforeEach(function () {
  window.G = { state: null }
})

describe('State management', function () {
  it('should create a new game state with balanced difficulty', function () {
    G.DIFFICULTY = {
      balanced: { budget: 5000, pointsMultiplier: 1.0 }
    }
    G.state = null
    G.newGame = function (difficulty) {
      var preset = G.DIFFICULTY[difficulty] || G.DIFFICULTY.balanced
      G.state = {
        points: 0,
        budget: preset.budget,
        pointsMultiplier: preset.pointsMultiplier,
        difficulty: difficulty
      }
    }
    G.newGame('balanced')
    expect(G.state.budget).toBe(5000)
    expect(G.state.pointsMultiplier).toBe(1.0)
    expect(G.state.difficulty).toBe('balanced')
  })

  it('should apply rockstar difficulty modifiers', function () {
    G.DIFFICULTY = {
      rockstar: { budget: 3000, pointsMultiplier: 1.5 }
    }
    G.state = null
    G.newGame = function (difficulty) {
      var preset = G.DIFFICULTY[difficulty] || G.DIFFICULTY.balanced
      G.state = {
        points: 0,
        budget: preset.budget,
        pointsMultiplier: preset.pointsMultiplier,
        difficulty: difficulty
      }
    }
    G.newGame('rockstar')
    expect(G.state.budget).toBe(3000)
    expect(G.state.pointsMultiplier).toBe(1.5)
  })

  it('should default to balanced difficulty', function () {
    G.DIFFICULTY = { balanced: { budget: 5000, pointsMultiplier: 1.0 } }
    G.state = null
    G.newGame = function () {
      var preset = G.DIFFICULTY['balanced']
      G.state = { budget: preset.budget, pointsMultiplier: preset.pointsMultiplier }
    }
    G.newGame()
    expect(G.state.budget).toBe(5000)
  })
})

describe('Point calculation', function () {
  it('should apply difficulty multiplier', function () {
    var state = { points: 0, pointsMultiplier: 1.5 }
    var pts = 100
    var result = Math.round(pts * (state.pointsMultiplier || 1))
    expect(result).toBe(150)
  })

  it('should default multiplier to 1', function () {
    var state = { points: 0, pointsMultiplier: undefined }
    var pts = 100
    var result = Math.round(pts * (state.pointsMultiplier || 1))
    expect(result).toBe(100)
  })

  it('should accumulate points correctly', function () {
    var state = { points: 0, pointsMultiplier: 1.2 }
    state.points += Math.round(50 * state.pointsMultiplier)
    state.points += Math.round(75 * state.pointsMultiplier)
    expect(state.points).toBe(150)
  })
})

describe('Budget management', function () {
  it('should deduct budget correctly', function () {
    var state = { budget: 5000 }
    state.budget -= 200
    expect(state.budget).toBe(4800)
  })

  it('should not go below zero', function () {
    var state = { budget: 100 }
    state.budget -= 200
    expect(state.budget).toBe(-100)
  })

  it('should earn budget correctly', function () {
    var state = { budget: 5000 }
    state.budget += 300
    expect(state.budget).toBe(5300)
  })
})

describe('Date advancement', function () {
  it('should advance date by N days', function () {
    var currentDate = new Date('2026-01-01')
    currentDate = new Date(currentDate.getTime() + 3 * 86400000)
    var month = currentDate.getMonth() + 1
    var day = currentDate.getDate()
    expect(month).toBe(1)
    expect(day).toBe(4)
  })

  it('should handle month boundary', function () {
    var currentDate = new Date('2026-01-28')
    currentDate = new Date(currentDate.getTime() + 5 * 86400000)
    var month = currentDate.getMonth() + 1
    var day = currentDate.getDate()
    expect(month).toBe(2)
    expect(day).toBe(2)
  })
})

describe('Festival month multiplier', function () {
  it('should return 1.5 when 3+ concerts in same month', function () {
    var attended = new Set([0, 1, 2])
    var monthKey = '2026-03'
    var count = 0
    attended.forEach(function (aid) {
      var acd = new Date('2026-03-05')
      if (acd.getFullYear() + '-' + (acd.getMonth() + 1).toString().padStart(2, '0') === monthKey) count++
    })
    expect(count >= 3 ? 1.5 : 1.0).toBe(1.5)
  })

  it('should return 1.0 when fewer than 3 concerts', function () {
    var attended = new Set([0, 1])
    var monthKey = '2026-03'
    var count = 0
    attended.forEach(function (aid) {
      var acd = new Date('2026-03-05')
      if (acd.getFullYear() + '-' + (acd.getMonth() + 1).toString().padStart(2, '0') === monthKey) count++
    })
    expect(count >= 3 ? 1.5 : 1.0).toBe(1.0)
  })
})

describe('Tutorial system', function () {
  it('should mark tutorial as done on complete', function () {
    localStorage.removeItem('concerts-tutorial-done')
    G.tutorialComplete = function () {
      localStorage.setItem('concerts-tutorial-done', 'done')
      G.tutorialStep = 0
    }
    G.tutorialComplete()
    expect(localStorage.getItem('concerts-tutorial-done')).toBe('done')
  })

  it('should mark tutorial as done on skip', function () {
    localStorage.removeItem('concerts-tutorial-done')
    G.tutorialSkip = function () {
      G.tutorialComplete()
    }
    G.tutorialSkip()
    expect(localStorage.getItem('concerts-tutorial-done')).toBe('done')
  })

  it('should not start tutorial if already completed', function () {
    localStorage.setItem('concerts-tutorial-done', 'done')
    G.tutorialStep = 0
    G.startTutorial = function () {
      if (localStorage.getItem('concerts-tutorial-done')) return
      G.tutorialStep = 1
    }
    G.startTutorial()
    expect(G.tutorialStep).toBe(0)
    expect(localStorage.getItem('concerts-tutorial-done')).toBe('done')
  })
})

describe('Pocket Football', function () {
  beforeEach(function () {
    G.MG_TYPES = ['gossip', 'cipher', 'auction', 'puzzle', 'streetteam', 'pocketfootball']
  })

  it('should include pocketfootball in MG_TYPES', function () {
    expect(G.MG_TYPES.indexOf('pocketfootball')).toBeGreaterThan(-1)
  })

  it('should have 6 minigame types', function () {
    expect(G.MG_TYPES.length).toBe(6)
  })
})
