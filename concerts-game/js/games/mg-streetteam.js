/* mg-streetteam.js — "Street Team" discovery minigame
   Drag posters onto correct cities on a mini-map. Drop in wrong city = wrong tap.
   Need 2 correct placements out of 3 posters. Timer: 20s */
window.G = window.G || {}

G.mgStreetTeam = function (concertId, onComplete) {
  var c = G.CONCERTS[concertId]
  var artist = G.ARTISTS[c.artistKey]
  var el = document.getElementById('discovery-content')

  var timeLeft = 20
  var correctPlaced = 0
  var wrongPlaced = 0
  var timer = null
  var posters = []
  var cities = []

  // Pick 3 target cities for posters (one correct, 2 distractors)
  var targetIds = [concertId]
  while (targetIds.length < 3) {
    var rid = Math.floor(Math.random() * G.CONCERTS.length)
    if (targetIds.indexOf(rid) === -1) targetIds.push(rid)
  }
  targetIds.forEach(function (id) {
    var tc = G.CONCERTS[id]
    cities.push({ cid: id, mx: tc.mx, my: tc.my, city: tc.city, correct: id === concertId })
  })

  // Create poster HTML
  var html = '<div class="mg-container fade-in">'
  html += '<div class="mg-title">📰 Street Team</div>'
  html += '<div class="mg-subtitle">Drag concert posters to the right cities! 3 posters to place.</div>'
  html += '<div class="mg-timer" id="mg-timer">' + timeLeft + 's</div>'
  html += '<div id="street-status" style="margin-top:8px;color:var(--teal)">Posters placed: 0 / 3</div>'
  html +=
    '<div class="street-area" id="street-area" style="position:relative;width:100%;height:350px;background:var(--bg-card);border-radius:var(--radius);overflow:hidden;margin-top:12px">'

  // Target zones (cities)
  cities.forEach(function (city, i) {
    var left = (city.mx / 1000) * 100
    var top = (city.my / 500) * 100
    html +=
      '<div class="street-city" data-cidx="' +
      city.cid +
      '" data-correct="' +
      city.correct +
      '" style="position:absolute;left:' +
      left +
      '%;top:' +
      top +
      '%;transform:translate(-50%,-50%);padding:8px 14px;background:rgba(0,245,212,0.15);border:1px solid var(--teal);border-radius:var(--radius-sm);cursor:pointer;font-size:0.8rem;color:var(--teal);font-weight:600;text-align:center">' +
      artist.emoji +
      ' ' +
      city.city +
      '</div>'
  })

  // Posters
  for (var p = 0; p < 3; p++) {
    var posterColor = p === 0 ? 'var(--gold)' : 'var(--pink)'
    html +=
      '<div class="street-poster" data-poster="' +
      p +
      '" draggable="true" style="position:absolute;left:' +
      (20 + p * 25) +
      '%;top:10px;padding:10px 16px;background:' +
      posterColor +
      ';color:#000;border-radius:var(--radius-sm);cursor:grab;font-size:0.8rem;font-weight:700;z-index:10">📰 Poster ' +
      (p + 1) +
      '</div>'
  }

  html += '</div>'
  html += '</div>'
  el.innerHTML = html

  // Drag and drop
  var draggedPoster = null
  document.querySelectorAll('.street-poster').forEach(function (poster) {
    poster.addEventListener(
      'touchstart',
      function (e) {
        draggedPoster = this
        this.style.zIndex = '20'
        this.style.opacity = '0.7'
      },
      { passive: true }
    )

    poster.addEventListener(
      'touchmove',
      function (e) {
        if (!draggedPoster) return
        e.preventDefault()
        var touch = e.touches[0]
        var area = document.getElementById('street-area')
        if (!area) return
        var rect = area.getBoundingClientRect()
        draggedPoster.style.left = touch.clientX - rect.left + 'px'
        draggedPoster.style.top = touch.clientY - rect.top + 'px'
      },
      { passive: false }
    )

    poster.addEventListener(
      'touchend',
      function (e) {
        if (!draggedPoster) return
        e.preventDefault()
        var poster = this
        poster.style.zIndex = '10'
        poster.style.opacity = '1'

        var touch = e.changedTouches[0]
        var area = document.getElementById('street-area')
        if (!area) return
        var rect = area.getBoundingClientRect()
        var x = touch.clientX - rect.left
        var y = touch.clientY - rect.top

        // Check which city was targeted
        var placed = false
        document.querySelectorAll('.street-city').forEach(function (city) {
          var cityX = (parseFloat(city.style.left) / 100) * rect.width
          var cityY = (parseFloat(city.style.top) / 100) * rect.height
          var dist = Math.sqrt(Math.pow(x - cityX, 2) + Math.pow(y - cityY, 2))
          if (dist < 60) {
            placed = true
            var isCorrect = city.getAttribute('data-correct') === 'true'
            if (isCorrect) {
              city.classList.add('correct-city')
              correctPlaced++
              G.sfx.coin()
              city.style.background = 'rgba(68,255,136,0.2)'
              city.style.borderColor = 'var(--success)'
              city.innerHTML = '✅ ' + city.innerHTML
            } else {
              wrongPlaced++
              G.sfx.fail()
              city.classList.add('wrong-city')
              city.style.background = 'rgba(255,68,68,0.2)'
              city.style.borderColor = 'var(--danger)'
              poster.style.display = 'none'
            }
            var statusEl = document.getElementById('street-status')
            if (statusEl)
              statusEl.textContent =
                'Posters placed: ' + (correctPlaced + wrongPlaced) + ' / 3 · Correct: ' + correctPlaced
          }
        })

        if (!placed) {
          poster.style.left = 20 + parseInt(poster.getAttribute('data-poster')) * 25 + '%'
          poster.style.top = '10px'
        }

        if (correctPlaced >= 2) {
          clearInterval(timer)
          setTimeout(function () {
            finish(true)
          }, 500)
        }

        draggedPoster = null
      },
      { passive: false }
    )

    // Desktop drag fallback
    poster.addEventListener('dragstart', function () {
      draggedPoster = this
    })

    poster.addEventListener('dragend', function () {
      if (!draggedPoster) return
      draggedPoster = null
      this.style.opacity = '1'
      this.style.zIndex = '10'
      var area = document.getElementById('street-area')
      if (!area) return
      var rect = area.getBoundingClientRect()
      var posterRect = this.getBoundingClientRect()
      var x = posterRect.left - rect.left + posterRect.width / 2
      var y = posterRect.top - rect.top + posterRect.height / 2
      document.querySelectorAll('.street-city').forEach(function (city) {
        var cityX = (parseFloat(city.style.left) / 100) * rect.width
        var cityY = (parseFloat(city.style.top) / 100) * rect.height
        var dist = Math.sqrt(Math.pow(x - cityX, 2) + Math.pow(y - cityY, 2))
        if (dist < 60) {
          var isCorrect = city.getAttribute('data-correct') === 'true'
          if (isCorrect) {
            correctPlaced++
            G.sfx.coin()
            city.style.background = 'rgba(68,255,136,0.2)'
            city.style.borderColor = 'var(--success)'
            city.innerHTML = '✅ ' + city.innerHTML
          } else {
            wrongPlaced++
            G.sfx.fail()
            city.style.background = 'rgba(255,68,68,0.2)'
            city.style.borderColor = 'var(--danger)'
            poster.style.display = 'none'
          }
          var statusEl = document.getElementById('street-status')
          if (statusEl)
            statusEl.textContent =
              'Posters placed: ' + (correctPlaced + wrongPlaced) + ' / 3 · Correct: ' + correctPlaced
        }
      })
      if (correctPlaced >= 2) {
        clearInterval(timer)
        setTimeout(function () {
          finish(true)
        }, 500)
      }
    })
  })

  timer = setInterval(function () {
    timeLeft--
    var te = document.getElementById('mg-timer')
    if (te) te.textContent = timeLeft + 's'
    if (timeLeft <= 0) {
      clearInterval(timer)
      finish(false)
    }
  }, 1000)

  function finish(success) {
    if (success) {
      G.sfx.success()
      el.innerHTML =
        '<div class="mg-container fade-in" style="justify-content:center;align-items:center;min-height:200px">' +
        '<h2 style="color:var(--gold)">🎉 Posters Posted!</h2>' +
        '<p style="margin:12px 0">' +
        artist.emoji +
        ' ' +
        artist.name +
        ' — ' +
        artist.tour +
        '</p>' +
        '<p style="color:var(--teal)">' +
        c.city +
        ', ' +
        c.country +
        ' — ' +
        c.date +
        '</p>' +
        '<p style="color:var(--gray-light)">' +
        c.venue +
        '</p>' +
        '<button class="btn btn-primary" style="margin-top:16px" id="mg-done-btn">Continue</button>' +
        '</div>'
    } else {
      G.sfx.fail()
      el.innerHTML =
        '<div class="mg-container fade-in" style="justify-content:center;align-items:center;min-height:200px">' +
        '<h2 style="color:var(--danger)">❌ Time\'s Up!</h2>' +
        '<p style="color:var(--gray-light);margin:12px 0">You placed ' +
        correctPlaced +
        '/3 posters correctly. Try again!</p>' +
        '<button class="btn btn-secondary" style="margin-top:16px" id="mg-done-btn">Back to Map</button>' +
        '</div>'
    }
    document.getElementById('mg-done-btn').addEventListener('click', function () {
      onComplete(success)
    })
  }
}
