/* calendar.js — In-game calendar view */
window.G = window.G || {}

G.calFilter = 'all'
G.calArtistFilter = 'all'

G.calPrev = function () {
  G.calViewMonth--
  if (G.calViewMonth < 0) {
    G.calViewMonth = 11
    G.calViewYear--
  }
  G.renderCalendar()
}

G.calNext = function () {
  G.calViewMonth++
  if (G.calViewMonth > 11) {
    G.calViewMonth = 0
    G.calViewYear++
  }
  G.renderCalendar()
}

G.renderCalendar = function () {
  var s = G.state
  var year = G.calViewYear
  var month = G.calViewMonth
  var months = [
    'January',
    'February',
    'March',
    'April',
    'May',
    'June',
    'July',
    'August',
    'September',
    'October',
    'November',
    'December'
  ]

  document.getElementById('cal-month-label').textContent = months[month] + ' ' + year

  // Populate artist filter
  var afEl = document.getElementById('cal-artist-filter')
  if (afEl) {
    afEl.innerHTML = '<option value="all">All Artists</option>'
    Object.keys(G.ARTISTS).forEach(function (key) {
      var a = G.ARTISTS[key]
      afEl.innerHTML += '<option value="' + key + '">' + a.emoji + ' ' + a.name + '</option>'
    })
    afEl.value = G.calArtistFilter
  }

  var grid = document.getElementById('calendar-grid')
  var html = ''

  // Day headers
  var days = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
  days.forEach(function (d) {
    html += '<div class="cal-header">' + d + '</div>'
  })

  // First day of month
  var firstDay = new Date(year, month, 1).getDay()
  var daysInMonth = new Date(year, month + 1, 0).getDate()

  // Today in game
  var todayDate = s.currentDate
  var isCurrentMonth = todayDate.getFullYear() === year && todayDate.getMonth() === month

  // Build concert map for this month
  var concertMap = {} // day -> [concerts]
  G.CONCERTS.forEach(function (c) {
    var cd = new Date(c.date + 'T00:00:00')
    if (cd.getFullYear() === year && cd.getMonth() === month) {
      var day = cd.getDate()
      if (!concertMap[day]) concertMap[day] = []
      concertMap[day].push(c)
    }
  })

  // Empty cells before first day
  for (var i = 0; i < firstDay; i++) {
    html += '<div class="cal-day empty"></div>'
  }

  // Day cells
  for (var d = 1; d <= daysInMonth; d++) {
    var isToday = isCurrentMonth && todayDate.getDate() === d
    var hasConcert = !!concertMap[d]
    var classes = 'cal-day'
    if (isToday) classes += ' today'
    if (hasConcert) classes += ' has-concert'

    html += '<div class="' + classes + '"'

    // Apply filters
    var dayVisible = false
    if (G.calFilter === 'all' || G.calArtistFilter === 'all') {
      dayVisible = hasConcert
    } else {
      if (concertMap[d]) {
        concertMap[d].forEach(function (c) {
          var status
          if (s.attended.has(c.id)) status = 'attended'
          else if (s.expired.has(c.id)) status = 'expired'
          else if (s.discovered.has(c.id)) status = 'discovered'
          else status = 'upcoming'
          if (G.calFilter === status) dayVisible = true
          if (G.calArtistFilter !== 'all' && c.artistKey === G.calArtistFilter) {
            if (
              (G.calFilter === 'upcoming' && status === 'upcoming') ||
              (G.calFilter === 'discovered' && status === 'discovered') ||
              (G.calFilter === 'attended' && status === 'attended') ||
              (G.calFilter === 'expired' && status === 'expired')
            ) {
              dayVisible = true
            } else if (G.calFilter === 'all') {
              dayVisible = true
            }
          }
        })
      }
    }

    if (!dayVisible) classes += ' cal-day-hidden'
    html += ' class="' + classes + '"'
    html += '>'

    html += '<div>' + d + '</div>'

    if (hasConcert) {
      html += '<div class="cal-concerts">'
      concertMap[d].forEach(function (c) {
        var artist = G.ARTISTS[c.artistKey]
        var discovered = s.discovered.has(c.id)
        if (G.calArtistFilter !== 'all' && c.artistKey !== G.calArtistFilter) {
          html += '<span style="opacity:0.25" title="' + artist.name + ' — ' + c.city + '">' + artist.emoji + '</span> '
        } else if (discovered) {
          html += '<span title="' + artist.name + ' — ' + c.city + '">' + artist.emoji + '</span> '
        } else {
          html += '<span title="Unknown concert">❓</span> '
        }
      })
      html += '</div>'
    }

    html += '</div>'
  }

  grid.innerHTML = html
}
