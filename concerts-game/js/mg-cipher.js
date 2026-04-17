/* mg-cipher.js — "Lyrics Cipher" discovery minigame
   Scrambled lyric phrase. Click letters in correct order. 20-second timer. */
window.G = window.G || {};

G.mgCipher = function(concertId, onComplete) {
  var c = G.CONCERTS[concertId];
  var artist = G.ARTISTS[c.artistKey];
  var el = document.getElementById('discovery-content');

  // Pick a random lyric for this artist
  var lyrics = G.LYRICS[c.artistKey];
  var target = lyrics[Math.floor(Math.random() * lyrics.length)];

  // Scramble letters (keep spaces in place)
  var letters = target.split('');
  var letterOnly = letters.filter(function(ch) { return ch !== ' '; });
  // Fisher-Yates shuffle
  for (var i = letterOnly.length - 1; i > 0; i--) {
    var j = Math.floor(Math.random() * (i + 1));
    var tmp = letterOnly[i]; letterOnly[i] = letterOnly[j]; letterOnly[j] = tmp;
  }

  var timeLeft = 20;
  var inputChars = [];
  var timer = null;
  var targetNoSpaces = target.replace(/ /g, '');

  var html = '<div class="mg-container fade-in">';
  html += '<div class="mg-title">🔤 Lyrics Cipher</div>';
  html += '<div class="mg-subtitle">Click letters in the correct order to unscramble the lyric!</div>';
  html += '<div class="mg-timer" id="mg-timer">' + timeLeft + 's</div>';
  html += '<div class="cipher-target" id="cipher-target">' + target.replace(/[A-Z]/g, '_') + '</div>';
  html += '<div class="cipher-input" id="cipher-input"></div>';
  html += '<div class="cipher-letters" id="cipher-letters">';
  letterOnly.forEach(function(ch, i) {
    html += '<div class="cipher-letter" data-idx="' + i + '">' + ch + '</div>';
  });
  html += '</div>';
  html += '<button class="btn btn-sm btn-secondary" id="cipher-undo" style="margin-top:8px">Undo</button>';
  html += '</div>';

  el.innerHTML = html;

  // Timer
  timer = setInterval(function() {
    timeLeft--;
    var te = document.getElementById('mg-timer');
    if (te) te.textContent = timeLeft + 's';
    if (timeLeft <= 0) {
      clearInterval(timer);
      finish(false);
    }
  }, 1000);

  // Letter click
  document.querySelectorAll('.cipher-letter').forEach(function(ltr) {
    ltr.addEventListener('click', function() {
      if (this.classList.contains('used')) return;
      var ch = this.textContent;
      var idx = parseInt(this.getAttribute('data-idx'));
      inputChars.push({ ch: ch, idx: idx });
      this.classList.add('used');
      G.sfx.tick();
      updateDisplay();
    });
  });

  // Undo
  document.getElementById('cipher-undo').addEventListener('click', function() {
    if (inputChars.length === 0) return;
    var last = inputChars.pop();
    var letters = document.querySelectorAll('.cipher-letter');
    letters[last.idx].classList.remove('used');
    updateDisplay();
  });

  function updateDisplay() {
    var inp = document.getElementById('cipher-input');
    var str = inputChars.map(function(o) { return o.ch; }).join('');
    inp.textContent = str;

    // Check win
    if (str === targetNoSpaces) {
      clearInterval(timer);
      setTimeout(function() { finish(true); }, 400);
    }
  }

  function finish(success) {
    clearInterval(timer);
    if (success) {
      G.sfx.success();
      el.innerHTML = '<div class="mg-container fade-in" style="justify-content:center;align-items:center;min-height:200px">' +
        '<h2 style="color:var(--gold)">🎉 Concert Discovered!</h2>' +
        '<p style="margin:12px 0;color:var(--pink)">"' + target + '"</p>' +
        '<p style="margin:8px 0">' + artist.emoji + ' ' + artist.name + ' — ' + artist.tour + '</p>' +
        '<p style="color:var(--teal)">' + c.city + ', ' + c.country + ' — ' + c.date + '</p>' +
        '<p style="color:var(--gray-light)">' + c.venue + '</p>' +
        '<button class="btn btn-primary" style="margin-top:16px" id="mg-done-btn">Continue</button>' +
        '</div>';
    } else {
      G.sfx.fail();
      el.innerHTML = '<div class="mg-container fade-in" style="justify-content:center;align-items:center;min-height:200px">' +
        '<h2 style="color:var(--danger)">❌ Time\'s Up!</h2>' +
        '<p style="color:var(--gray-light);margin:12px 0">The answer was: "' + target + '"</p>' +
        '<button class="btn btn-secondary" style="margin-top:16px" id="mg-done-btn">Back to Map</button>' +
        '</div>';
    }
    document.getElementById('mg-done-btn').addEventListener('click', function() {
      onComplete(success);
    });
  }
};
