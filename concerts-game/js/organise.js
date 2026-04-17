/* organise.js — Organise your own fictional concert */
window.G = window.G || {};

G.showOrganise = function() {
  G.showScreen('screen-organise');
  G.sfx.click();

  var el = document.getElementById('organise-content');
  var selectedCity = 0;
  var selectedVenue = 0;
  var selectedFriends = []; // indices into G.FRIENDS
  var ticketPrice = 50;

  render();

  function render() {
    var html = '';

    // City selector
    html += '<div class="org-section">';
    html += '<label>📍 City</label>';
    html += '<select id="org-city">';
    G.ORGANISE_CITIES.forEach(function(c, i) {
      html += '<option value="' + i + '"' + (i === selectedCity ? ' selected' : '') + '>' + c + '</option>';
    });
    html += '</select></div>';

    // Venue tier
    html += '<div class="org-section">';
    html += '<label>🏟️ Venue</label>';
    html += '<select id="org-venue">';
    G.VENUE_TIERS.forEach(function(v, i) {
      html += '<option value="' + i + '"' + (i === selectedVenue ? ' selected' : '') + '>' +
        v.name + ' (€' + v.cost.toLocaleString() + ' · ' + v.capacity.toLocaleString() + ' capacity)</option>';
    });
    html += '</select></div>';

    // Artist roster (1-3)
    html += '<div class="org-section">';
    html += '<label>🎤 Artists (pick 1–3)</label>';
    html += '<div class="friend-grid">';
    G.FRIENDS.forEach(function(f, i) {
      var sel = selectedFriends.indexOf(i) !== -1;
      html += '<div class="friend-card' + (sel ? ' selected' : '') + '" data-fidx="' + i + '">';
      html += '<div class="fc-emoji">' + f.emoji + '</div>';
      html += '<div class="fc-name">' + f.name + '</div>';
      html += '<div class="fc-style">' + f.style + '</div>';
      html += '<div class="fc-stats">FR:' + f.friendship + ' CR:' + f.craziness + ' LV:' + f.love + '</div>';
      html += '</div>';
    });
    html += '</div></div>';

    // Ticket price
    html += '<div class="org-section">';
    html += '<label>🎫 Ticket Price <span class="org-slider-val" id="org-price-val">€' + ticketPrice + '</span></label>';
    html += '<input type="range" id="org-price" min="10" max="200" step="5" value="' + ticketPrice + '">';
    html += '</div>';

    // Cost preview
    var venue = G.VENUE_TIERS[selectedVenue];
    var cost = venue.cost;
    var canAfford = G.state.budget >= cost;

    html += '<div class="booking-total">';
    html += '<div>Venue Cost</div>';
    html += '<div class="total-price">€' + cost.toLocaleString() + '</div>';
    html += '<div style="color:var(--gray);font-size:0.8rem">Your budget: €' + G.state.budget.toLocaleString() + '</div>';
    if (!canAfford) html += '<div style="color:var(--danger);font-size:0.8rem">Not enough budget!</div>';
    html += '</div>';

    // Buttons
    html += '<div class="booking-btns" style="margin-bottom:24px">';
    html += '<button class="btn btn-secondary" id="org-back">← Back</button>';
    var launchDisabled = (selectedFriends.length === 0 || !canAfford) ? ' disabled' : '';
    html += '<button class="btn btn-gold"' + launchDisabled + ' id="org-launch">🎤 Launch Concert!</button>';
    html += '</div>';

    el.innerHTML = html;

    // Event listeners
    document.getElementById('org-city').addEventListener('change', function() {
      selectedCity = parseInt(this.value);
    });
    document.getElementById('org-venue').addEventListener('change', function() {
      selectedVenue = parseInt(this.value);
      render();
    });
    document.getElementById('org-price').addEventListener('input', function() {
      ticketPrice = parseInt(this.value);
      document.getElementById('org-price-val').textContent = '€' + ticketPrice;
    });

    el.querySelectorAll('.friend-card').forEach(function(card) {
      card.addEventListener('click', function() {
        var idx = parseInt(this.getAttribute('data-fidx'));
        var pos = selectedFriends.indexOf(idx);
        G.sfx.click();
        if (pos !== -1) {
          selectedFriends.splice(pos, 1);
        } else if (selectedFriends.length < 3) {
          selectedFriends.push(idx);
        }
        render();
      });
    });

    document.getElementById('org-back').addEventListener('click', function() {
      G.sfx.click();
      G.showScreen('screen-map');
    });

    var launchBtn = document.getElementById('org-launch');
    if (launchBtn && !launchBtn.disabled) {
      launchBtn.addEventListener('click', function() {
        launchConcert();
      });
    }
  }

  function launchConcert() {
    var venue = G.VENUE_TIERS[selectedVenue];
    var city = G.ORGANISE_CITIES[selectedCity];

    // Deduct venue cost
    G.spendBudget(venue.cost);
    G.state.organisedCount++;

    // Calculate outcome
    var avgFriendship = 0, avgCraziness = 0, avgLove = 0;
    var artists = [];
    selectedFriends.forEach(function(i) {
      var f = G.FRIENDS[i];
      avgFriendship += f.friendship;
      avgCraziness += f.craziness;
      avgLove += f.love;
      artists.push(f);
    });
    var n = selectedFriends.length;
    avgFriendship /= n;
    avgCraziness /= n;
    avgLove /= n;

    // Attendance formula:
    // Higher friendship = more loyal fans, love = attracts couples,
    // craziness can attract or scare people
    var attractionScore = (avgFriendship * 0.4 + avgLove * 0.35 + avgCraziness * 0.25) / 100;

    // Ticket price sweet spot: too cheap = low revenue, too high = fewer fans
    var priceOptimal = 60 + (selectedVenue * 20); // Club=60, Theatre=80, Arena=100
    var priceFactor = 1 - Math.abs(ticketPrice - priceOptimal) / 200;
    priceFactor = Math.max(0.2, Math.min(1.0, priceFactor));

    // Number of artists bonus
    var artistBonus = 1 + (n - 1) * 0.25;

    // Final attendance ratio
    var attendanceRatio = Math.min(1.0, attractionScore * priceFactor * artistBonus);
    attendanceRatio = Math.max(0.05, attendanceRatio);

    // Add randomness
    attendanceRatio += (Math.random() - 0.5) * 0.2;
    attendanceRatio = Math.max(0.05, Math.min(1.0, attendanceRatio));

    var attendance = Math.round(venue.capacity * attendanceRatio);
    var revenue = attendance * ticketPrice;
    var profit = revenue - venue.cost;

    // Chaos events triggered by high craziness
    var chaosEvent = null;
    var chaosBonus = 0;
    if (avgCraziness >= 70 && Math.random() < 0.6) {
      var chaosEvents = [
        { text: "🔥 " + artists[0].name + " set off fireworks inside the venue. The crowd went INSANE. Surprisingly, no one was hurt.", bonus: 300 },
        { text: "💀 Rex Phantom crowd-surfed into the mixing desk. Chaos ensued. Surprisingly, everyone loved it.", bonus: 300 },
        { text: "🎸 An impromptu mosh pit broke out during the guitar solo. Legendary.", bonus: 200 },
        { text: "🤪 " + artists[0].name + " challenged the audience to a scream-off. The venue shook.", bonus: 250 },
        { text: "🎤 The mic flew into the crowd and someone started rapping. It was actually good.", bonus: 150 },
        { text: "🕺 A conga line formed through the entire venue. Even security joined in.", bonus: 200 }
      ];
      chaosEvent = chaosEvents[Math.floor(Math.random() * chaosEvents.length)];
      chaosBonus = chaosEvent.bonus;
    }

    // Points from organising
    var basePoints = Math.round(attendance / 10);
    var totalPoints = basePoints + chaosBonus;

    G.addPoints(totalPoints);
    if (profit > 0) G.earnBudget(profit);

    // Advance 3-7 days
    G.advanceDays(Math.floor(Math.random() * 5) + 3);

    // Show result
    G.sfx.success();

    var artistNames = artists.map(function(a) { return a.emoji + ' ' + a.name; }).join(', ');
    var profitColor = profit >= 0 ? 'var(--success)' : 'var(--danger)';
    var profitSign = profit >= 0 ? '+' : '';

    var html = '<div class="fade-in" style="text-align:center;padding:20px 0">';
    html += '<h2 style="color:var(--gold);margin-bottom:16px">🎤 Concert Report</h2>';
    html += '<div class="card" style="text-align:left;margin-bottom:16px">';
    html += '<div style="margin-bottom:8px"><strong>City:</strong> ' + city + '</div>';
    html += '<div style="margin-bottom:8px"><strong>Venue:</strong> ' + venue.name + '</div>';
    html += '<div style="margin-bottom:8px"><strong>Artists:</strong> ' + artistNames + '</div>';
    html += '<div style="margin-bottom:8px"><strong>Ticket Price:</strong> €' + ticketPrice + '</div>';
    html += '<hr style="border-color:var(--gray-dark);margin:12px 0">';
    html += '<div style="margin-bottom:4px">👥 Attendance: <strong>' + attendance.toLocaleString() + '</strong> / ' + venue.capacity.toLocaleString() + '</div>';
    html += '<div style="margin-bottom:4px">💵 Revenue: <strong>€' + revenue.toLocaleString() + '</strong></div>';
    html += '<div style="margin-bottom:4px">📊 Venue Cost: -€' + venue.cost.toLocaleString() + '</div>';
    html += '<div style="font-size:1.2rem;font-weight:700;color:' + profitColor + ';margin:8px 0">' + profitSign + '€' + profit.toLocaleString() + '</div>';

    if (chaosEvent) {
      html += '<hr style="border-color:var(--gray-dark);margin:12px 0">';
      html += '<div style="color:var(--pink);font-style:italic;margin:8px 0">' + chaosEvent.text + '</div>';
      html += '<div style="color:var(--gold)">+' + chaosBonus + ' bonus points!</div>';
    }

    html += '<div style="color:var(--teal);margin-top:8px">+' + totalPoints + ' points earned</div>';
    html += '</div>';

    html += '<button class="btn btn-primary" id="org-done">Back to Map 🌍</button>';
    html += '</div>';

    el.innerHTML = html;

    document.getElementById('org-done').addEventListener('click', function() {
      G.sfx.click();
      G.showScreen('screen-map');
      G.checkGameEnd();
    });
  }
};
