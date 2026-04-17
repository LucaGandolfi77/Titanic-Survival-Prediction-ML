/* map.js — SVG world map rendering and concert dot interaction */
window.G = window.G || {};

G.renderMap = function() {
  var container = document.getElementById('map-container');
  var s = G.state;

  // Simplified SVG continent outlines
  var continents = [
    // North America
    'M 55,65 L 145,38 L 250,42 L 290,85 L 295,155 L 278,200 L 248,238 L 228,258 L 198,275 L 178,268 L 158,282 L 148,258 L 100,238 L 62,178 L 42,118 Z',
    // Central America
    'M 158,282 L 178,268 L 195,290 L 200,310 L 185,318 L 168,305 Z',
    // South America
    'M 228,318 L 280,295 L 330,305 L 352,335 L 358,382 L 342,432 L 312,468 L 278,455 L 258,418 L 238,368 Z',
    // Europe
    'M 438,55 L 498,42 L 548,58 L 558,105 L 542,142 L 518,162 L 498,172 L 468,158 L 442,128 L 432,92 Z',
    // Africa
    'M 442,198 L 498,185 L 548,195 L 572,225 L 588,305 L 572,382 L 538,418 L 498,408 L 462,368 L 442,292 L 438,232 Z',
    // Asia
    'M 558,42 L 648,32 L 748,48 L 838,82 L 868,132 L 878,182 L 848,222 L 798,252 L 718,262 L 658,248 L 608,222 L 578,182 L 558,128 Z',
    // Middle East
    'M 572,182 L 628,172 L 658,202 L 648,242 L 608,248 L 578,228 Z',
    // Australia
    'M 818,352 L 898,332 L 948,362 L 942,412 L 898,438 L 838,418 L 818,382 Z',
    // Japan (islands)
    'M 860,128 L 874,112 L 884,142 L 876,168 L 864,156 Z',
    // UK/Ireland
    'M 452,92 L 466,82 L 476,95 L 476,118 L 466,128 L 456,112 Z',
  ];

  var html = '<svg viewBox="0 0 1000 500" preserveAspectRatio="xMidYMid meet">';

  // Ocean
  html += '<rect width="1000" height="500" fill="#080816"/>';

  // Grid lines
  for (var i = 1; i < 10; i++) {
    html += '<line x1="'+i*100+'" y1="0" x2="'+i*100+'" y2="500" stroke="rgba(255,255,255,0.03)" stroke-width="0.5"/>';
  }
  for (var j = 1; j < 5; j++) {
    html += '<line x1="0" y1="'+j*100+'" x2="1000" y2="'+j*100+'" stroke="rgba(255,255,255,0.03)" stroke-width="0.5"/>';
  }

  // Continents
  continents.forEach(function(d) {
    html += '<path d="'+d+'" fill="#161630" stroke="rgba(0,245,212,0.12)" stroke-width="0.8"/>';
  });

  // Concert dots
  G.CONCERTS.forEach(function(c) {
    var discovered = s.discovered.has(c.id);
    var attended = s.attended.has(c.id);
    var booked = !!s.booked[c.id];
    var available = G.isConcertAvailable(c.id);
    var expired = s.expired.has(c.id);
    var artist = G.ARTISTS[c.artistKey];

    var dotColor, r, glowColor, opacity, label;

    if (attended) {
      dotColor = '#44ff88'; r = 5; glowColor = 'rgba(68,255,136,0.25)'; opacity = 0.7; label = '✓';
    } else if (expired) {
      dotColor = '#555'; r = 4; glowColor = 'none'; opacity = 0.4; label = '✗';
    } else if (available) {
      dotColor = '#ffd700'; r = 7; glowColor = 'rgba(255,215,0,0.4)'; opacity = 1; label = '🎶';
    } else if (booked) {
      dotColor = '#00f5d4'; r = 5; glowColor = 'rgba(0,245,212,0.3)'; opacity = 0.9; label = '✈';
    } else if (discovered) {
      dotColor = artist.color; r = 5; glowColor = artist.color.replace(')', ',0.3)').replace('rgb', 'rgba'); opacity = 0.9; label = artist.emoji;
    } else {
      dotColor = '#666'; r = 5; glowColor = 'rgba(255,255,255,0.1)'; opacity = 0.6; label = '❓';
    }

    // Glow circle
    if (glowColor !== 'none') {
      html += '<circle cx="'+c.mx+'" cy="'+c.my+'" r="'+(r+6)+'" fill="'+glowColor+'" opacity="0.5">';
      if (available) html += '<animate attributeName="r" values="'+(r+4)+';'+(r+10)+';'+(r+4)+'" dur="1.5s" repeatCount="indefinite"/>';
      html += '</circle>';
    }

    // Dot
    html += '<circle cx="'+c.mx+'" cy="'+c.my+'" r="'+r+'" fill="'+dotColor+'" opacity="'+opacity+'" style="cursor:pointer" data-cid="'+c.id+'">';
    if (!attended && !expired && discovered) {
      html += '<animate attributeName="r" values="'+r+';'+(r+2)+';'+r+'" dur="2s" repeatCount="indefinite"/>';
    }
    html += '</circle>';

    // Hit area (larger invisible circle for easier clicking)
    html += '<circle cx="'+c.mx+'" cy="'+c.my+'" r="14" fill="transparent" style="cursor:pointer" class="map-hit" data-cid="'+c.id+'"/>';
  });

  html += '</svg>';
  container.innerHTML = html;

  // Attach click + touch handlers (touchend eliminates 300ms iOS delay)
  container.querySelectorAll('.map-hit').forEach(function(el) {
    var touched = false;

    el.addEventListener('touchstart', function(e) {
      touched = true;
      // Show tooltip on first touch
      var cid = parseInt(this.getAttribute('data-cid'));
      var touch = e.touches[0];
      G.showMapTooltipXY(cid, touch.clientX, touch.clientY);
    }, { passive: true });

    el.addEventListener('touchend', function(e) {
      e.preventDefault(); // prevent ghost click
      var cid = parseInt(this.getAttribute('data-cid'));
      G.handleMapClick(cid, e);
      setTimeout(function() { G.hideMapTooltip(); }, 1200);
    }, { passive: false });

    el.addEventListener('click', function(e) {
      if (touched) { touched = false; return; } // already handled by touchend
      var cid = parseInt(this.getAttribute('data-cid'));
      G.handleMapClick(cid, e);
    });

    el.addEventListener('mouseenter', function(e) {
      var cid = parseInt(this.getAttribute('data-cid'));
      G.showMapTooltip(cid, e);
    });
    el.addEventListener('mouseleave', function() {
      G.hideMapTooltip();
    });
  });
};

/** Handle click on a concert dot */
G.handleMapClick = function(cid, event) {
  var s = G.state;
  var c = G.CONCERTS[cid];
  var artist = G.ARTISTS[c.artistKey];
  G.sfx.click();

  if (s.attended.has(cid)) {
    var sc = s.concertScores[cid];
    if (sc) {
      G.toast(artist.emoji + ' ' + artist.name + ' in ' + c.city + ' — ' + sc.score + '% · ' + sc.placement, 3000);
    }
    return;
  }
  if (s.expired.has(cid)) {
    G.toast('😢 This concert has expired. You missed ' + artist.name + ' in ' + c.city + '!', 3000);
    return;
  }
  if (!s.discovered.has(cid)) {
    // Start discovery minigame
    G.startDiscovery(cid);
    return;
  }
  if (G.isConcertAvailable(cid)) {
    // Concert date is NOW — start concert minigame
    G.startConcertGame(cid);
    return;
  }
  if (!s.booked[cid]) {
    if (G.isConcertBookable(cid)) {
      G.showBooking(cid);
    } else {
      G.toast('⏰ This concert date has already passed.', 2500);
    }
    return;
  }
  // Booked but not yet available
  var cd = new Date(c.date + "T00:00:00");
  var diff = Math.ceil((cd - s.currentDate) / 86400000);
  G.toast(artist.emoji + ' ' + artist.name + ' in ' + c.city + ' — ' + diff + ' day' + (diff !== 1 ? 's' : '') + ' away! ✈', 3000);
};

/** Build tooltip HTML for a concert id */
G.buildTooltipHTML = function(cid) {
  var s = G.state;
  var c = G.CONCERTS[cid];
  var artist = G.ARTISTS[c.artistKey];
  var html = '';
  if (s.discovered.has(cid)) {
    html += '<div class="tt-artist">' + artist.emoji + ' ' + artist.name + '</div>';
    html += '<div class="tt-city">' + c.city + ', ' + c.country + '</div>';
    html += '<div class="tt-date">' + c.date + ' — ' + c.venue + '</div>';
    if (s.attended.has(cid)) {
      html += '<div class="tt-status" style="color:#44ff88">✓ Attended</div>';
    } else if (s.expired.has(cid)) {
      html += '<div class="tt-status" style="color:#888">Expired</div>';
    } else if (s.booked[cid]) {
      html += '<div class="tt-status" style="color:#00f5d4">✈ Booked</div>';
    } else {
      html += '<div class="tt-status" style="color:#ffd700">Tap to book</div>';
    }
  } else if (s.expired.has(cid)) {
    html += '<div class="tt-hint">Expired — missed concert</div>';
  } else {
    html += '<div class="tt-hint">' + c.hint + '</div>';
    html += '<div class="tt-status" style="color:#888">Tap ❓ to discover</div>';
  }
  return html;
};

/** Show tooltip on mouse hover (desktop) */
G.showMapTooltip = function(cid, event) {
  var tt = document.getElementById('map-tooltip');
  tt.innerHTML = G.buildTooltipHTML(cid);
  tt.classList.add('visible');

  var rect = document.getElementById('map-container').getBoundingClientRect();
  var x = event.clientX - rect.left + 15;
  var y = event.clientY - rect.top - 10;
  if (x + 250 > rect.width) x = x - 270;
  if (y < 0) y = 10;
  tt.style.left = x + 'px';
  tt.style.top = y + 'px';
};

/** Show tooltip at absolute page coordinates (touch) */
G.showMapTooltipXY = function(cid, clientX, clientY) {
  var tt = document.getElementById('map-tooltip');
  tt.innerHTML = G.buildTooltipHTML(cid);
  tt.classList.add('visible');

  var rect = document.getElementById('map-container').getBoundingClientRect();
  var x = clientX - rect.left + 15;
  var y = clientY - rect.top - 70; // show above finger
  if (x + 250 > rect.width) x = x - 270;
  if (y < 0) y = 10;
  tt.style.left = x + 'px';
  tt.style.top = y + 'px';
};

G.hideMapTooltip = function() {
  document.getElementById('map-tooltip').classList.remove('visible');
};
