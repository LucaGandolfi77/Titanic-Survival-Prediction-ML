/* mg-auction.js — "Ticket Auction" discovery minigame
   Bid within 10% of hidden market price before timer hits zero. */
window.G = window.G || {};

G.mgAuction = function(concertId, onComplete) {
  var c = G.CONCERTS[concertId];
  var artist = G.ARTISTS[c.artistKey];
  var el = document.getElementById('discovery-content');

  // Market price based on artist popularity
  var basePrices = { swift: 320, styles: 280, maneskin: 250, eilish: 260, beyonce: 350 };
  var marketPrice = basePrices[c.artistKey] + Math.floor(Math.random() * 80 - 40);
  var minAccept = Math.floor(marketPrice * 0.9);
  var maxAccept = Math.floor(marketPrice * 1.1);

  // Show hint range (wider than actual acceptance)
  var hintLow = Math.floor(marketPrice * 0.7);
  var hintHigh = Math.floor(marketPrice * 1.4);

  var timeLeft = 15;
  var timer = null;
  var bidPlaced = false;

  var html = '<div class="mg-container fade-in">';
  html += '<div class="mg-title">🔨 Ticket Auction</div>';
  html += '<div class="mg-subtitle">Bid within the right range to win the ticket!</div>';
  html += '<div class="mg-timer" id="mg-timer">' + timeLeft + 's</div>';
  html += '<div class="auction-range">Estimated value: €' + hintLow + ' — €' + hintHigh + '</div>';
  html += '<div class="auction-price" id="auction-current">Current bid: €0</div>';
  html += '<div class="auction-input">';
  html += '<span style="color:var(--gray)">€</span>';
  html += '<input type="number" id="auction-bid" min="50" max="999" placeholder="Your bid" />';
  html += '<button class="btn btn-gold" id="auction-submit">BID!</button>';
  html += '</div>';
  html += '<div class="auction-msg" id="auction-msg"></div>';
  html += '</div>';

  el.innerHTML = html;

  var bidInput = document.getElementById('auction-bid');
  bidInput.focus();

  // Simulate rising current bid
  var currentBid = Math.floor(marketPrice * 0.5);
  var bidEl = document.getElementById('auction-current');

  timer = setInterval(function() {
    timeLeft--;
    var te = document.getElementById('mg-timer');
    if (te) te.textContent = timeLeft + 's';

    // Current bid rises
    currentBid += Math.floor(Math.random() * 20 + 5);
    if (bidEl) bidEl.textContent = 'Current bid: €' + currentBid;
    G.sfx.tick();

    if (timeLeft <= 0) {
      clearInterval(timer);
      if (!bidPlaced) finish('timeout');
    }
  }, 1000);

  // Submit bid
  document.getElementById('auction-submit').addEventListener('click', function() {
    if (bidPlaced) return;
    var bid = parseInt(bidInput.value);
    if (isNaN(bid) || bid < 10) {
      document.getElementById('auction-msg').innerHTML = '<span style="color:var(--danger)">Enter a valid bid!</span>';
      return;
    }
    bidPlaced = true;
    clearInterval(timer);

    if (bid >= minAccept && bid <= maxAccept) {
      finish('win');
    } else if (bid > maxAccept) {
      // Overpay — still win but lose extra budget
      var overpay = bid - marketPrice;
      G.spendBudget(overpay);
      finish('overpay', overpay);
    } else {
      finish('low');
    }
  });

  // Also allow Enter key
  bidInput.addEventListener('keydown', function(e) {
    if (e.key === 'Enter') document.getElementById('auction-submit').click();
  });

  function finish(result, extra) {
    clearInterval(timer);
    var success = (result === 'win' || result === 'overpay');

    if (result === 'win') {
      G.sfx.success();
      el.innerHTML = '<div class="mg-container fade-in" style="justify-content:center;align-items:center;min-height:200px">' +
        '<h2 style="color:var(--gold)">🎉 Auction Won!</h2>' +
        '<p style="margin:12px 0">Perfect bid! Market price was €' + marketPrice + '</p>' +
        '<p>' + artist.emoji + ' ' + artist.name + ' — ' + artist.tour + '</p>' +
        '<p style="color:var(--teal)">' + c.city + ', ' + c.country + ' — ' + c.date + '</p>' +
        '<p style="color:var(--gray-light)">' + c.venue + '</p>' +
        '<button class="btn btn-primary" style="margin-top:16px" id="mg-done-btn">Continue</button>' +
        '</div>';
    } else if (result === 'overpay') {
      G.sfx.coin();
      el.innerHTML = '<div class="mg-container fade-in" style="justify-content:center;align-items:center;min-height:200px">' +
        '<h2 style="color:var(--gold)">🎉 Auction Won (Overpaid!)</h2>' +
        '<p style="margin:12px 0;color:var(--danger)">You overpaid by €' + extra + '! Market: €' + marketPrice + '</p>' +
        '<p>' + artist.emoji + ' ' + artist.name + ' — ' + artist.tour + '</p>' +
        '<p style="color:var(--teal)">' + c.city + ', ' + c.country + ' — ' + c.date + '</p>' +
        '<button class="btn btn-primary" style="margin-top:16px" id="mg-done-btn">Continue</button>' +
        '</div>';
    } else if (result === 'low') {
      G.sfx.fail();
      el.innerHTML = '<div class="mg-container fade-in" style="justify-content:center;align-items:center;min-height:200px">' +
        '<h2 style="color:var(--danger)">❌ Bid Too Low!</h2>' +
        '<p style="color:var(--gray-light);margin:12px 0">Market was €' + marketPrice + '. Try again later.</p>' +
        '<button class="btn btn-secondary" style="margin-top:16px" id="mg-done-btn">Back to Map</button>' +
        '</div>';
    } else {
      G.sfx.fail();
      el.innerHTML = '<div class="mg-container fade-in" style="justify-content:center;align-items:center;min-height:200px">' +
        '<h2 style="color:var(--danger)">⏰ Time\'s Up!</h2>' +
        '<p style="color:var(--gray-light);margin:12px 0">You didn\'t bid in time. Market was €' + marketPrice + '</p>' +
        '<button class="btn btn-secondary" style="margin-top:16px" id="mg-done-btn">Back to Map</button>' +
        '</div>';
    }

    document.getElementById('mg-done-btn').addEventListener('click', function() {
      onComplete(success);
    });
  }
};
