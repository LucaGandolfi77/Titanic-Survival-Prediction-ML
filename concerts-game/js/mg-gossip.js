/* mg-gossip.js — "Gossip Feed" discovery minigame
   Scrolling fake social media posts. Tap posts that contain concert hints.
   10-second window, need 3+ correct taps. Wrong taps deduct time. */
window.G = window.G || {};

G.mgGossip = function(concertId, onComplete) {
  var c = G.CONCERTS[concertId];
  var artist = G.ARTISTS[c.artistKey];
  var el = document.getElementById('discovery-content');

  // Generate posts: 5 correct (contain hints) + 12 distractors
  var correctPosts = [
    '🎶 RUMOR: ' + artist.name + ' spotted scouting venues in ' + c.country + '!',
    '✈️ ' + artist.name + '\'s crew just booked flights to ' + c.city + '… 👀',
    '🎤 Insider says ' + artist.tour + ' will include a ' + c.city + ' stop!',
    '📅 Mark your calendars — ' + c.city + ' might see ' + artist.name + ' soon!',
    '🔥 Production trucks seen at ' + c.venue + '. Something BIG is coming!'
  ];
  var distractors = G.GOSSIP_DISTRACTORS.slice().sort(function() { return Math.random() - 0.5; }).slice(0, 12);

  // Mix posts
  var posts = [];
  correctPosts.forEach(function(t) { posts.push({ text: t, correct: true }); });
  distractors.forEach(function(t) { posts.push({ text: t, correct: false }); });
  // Shuffle
  posts.sort(function() { return Math.random() - 0.5; });

  var fakeUsers = ['@musicfan99','@concertgeek','@dailybuzz','@trendwatch','@livescoop',
    '@vibecheck','@starstruck','@soundwave','@nightowl','@backstage411',
    '@moshpit_mike','@melodyjane','@bassdropliz','@rhythmrick','@tunesmith',
    '@synth_sarah','@djcoolcat'];

  var timeLeft = 10;
  var correctTaps = 0;
  var timer = null;

  var html = '<div class="mg-container fade-in">';
  html += '<div class="mg-title">📱 Gossip Feed</div>';
  html += '<div class="mg-subtitle">Tap posts with real concert hints! Need 3 correct taps.</div>';
  html += '<div class="mg-timer" id="mg-timer">' + timeLeft + 's</div>';
  html += '<div class="gossip-feed" id="gossip-feed">';

  posts.forEach(function(p, i) {
    var user = fakeUsers[i % fakeUsers.length];
    html += '<div class="gossip-post" data-idx="' + i + '" data-correct="' + p.correct + '">';
    html += '<span class="gp-user">' + user + '</span>';
    html += p.text;
    html += '</div>';
  });

  html += '</div>';
  html += '<div id="mg-status" style="margin-top:10px;color:var(--teal)">Correct: 0 / 3</div>';
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

  // Click handlers
  document.querySelectorAll('.gossip-post').forEach(function(post) {
    post.addEventListener('click', function() {
      if (this.classList.contains('tapped-correct') || this.classList.contains('tapped-wrong')) return;
      var isCorrect = this.getAttribute('data-correct') === 'true';

      if (isCorrect) {
        this.classList.add('tapped-correct');
        correctTaps++;
        G.sfx.coin();
        var st = document.getElementById('mg-status');
        if (st) st.textContent = 'Correct: ' + correctTaps + ' / 3';
        if (correctTaps >= 3) {
          clearInterval(timer);
          setTimeout(function() { finish(true); }, 500);
        }
      } else {
        this.classList.add('tapped-wrong');
        timeLeft = Math.max(0, timeLeft - 2);
        G.sfx.fail();
        var te = document.getElementById('mg-timer');
        if (te) te.textContent = timeLeft + 's';
      }
    });
  });

  function finish(success) {
    clearInterval(timer);
    if (success) {
      G.sfx.success();
      el.innerHTML = '<div class="mg-container fade-in" style="justify-content:center;align-items:center;min-height:200px">' +
        '<h2 style="color:var(--gold)">🎉 Concert Discovered!</h2>' +
        '<p style="margin:12px 0">' + artist.emoji + ' ' + artist.name + ' — ' + artist.tour + '</p>' +
        '<p style="color:var(--teal)">' + c.city + ', ' + c.country + ' — ' + c.date + '</p>' +
        '<p style="color:var(--gray-light)">' + c.venue + '</p>' +
        '<button class="btn btn-primary" style="margin-top:16px" id="mg-done-btn">Continue</button>' +
        '</div>';
    } else {
      G.sfx.fail();
      el.innerHTML = '<div class="mg-container fade-in" style="justify-content:center;align-items:center;min-height:200px">' +
        '<h2 style="color:var(--danger)">❌ Time\'s Up!</h2>' +
        '<p style="color:var(--gray-light);margin:12px 0">You didn\'t find enough hints. Try again later.</p>' +
        '<button class="btn btn-secondary" style="margin-top:16px" id="mg-done-btn">Back to Map</button>' +
        '</div>';
    }
    document.getElementById('mg-done-btn').addEventListener('click', function() {
      onComplete(success);
    });
  }
};
