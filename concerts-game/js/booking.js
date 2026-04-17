/* booking.js — Flight & hotel booking panel */
window.G = window.G || {};

/**
 * Show the booking screen for a discovered concert.
 * Player picks a flight (3 options) and hotel tier, then confirms.
 */
G.showBooking = function(concertId) {
  var s = G.state;
  var c = G.CONCERTS[concertId];
  var artist = G.ARTISTS[c.artistKey];
  var el = document.getElementById('booking-content');

  G.showScreen('screen-booking');

  // Days until concert
  var concertDate = new Date(c.date + "T00:00:00");
  var daysUntil = Math.max(1, Math.ceil((concertDate - s.currentDate) / 86400000));

  // Price multiplier: booking earlier = cheaper (1.0 at 60+ days, up to 2.0 at 1 day)
  var urgency = Math.max(1.0, 1.0 + (1.0 - Math.min(daysUntil, 60) / 60));

  // Generate 3 flight options
  var flights = [
    { name: "Direct Flight ✈️",      detail: "~4h non-stop",   base: 320 },
    { name: "1-Stop Flight 🛫",      detail: "~8h, 1 layover", base: 180 },
    { name: "Budget Flight 💰",      detail: "~14h, 2 stops",  base: 90  }
  ];
  flights.forEach(function(f) {
    f.price = Math.round(f.base * urgency);
  });

  // Hotel options (per night, 2 nights assumed)
  var hotelNights = 2;
  var hotels = [
    { name: "Budget Hostel 🏠",   tier: "Budget",   base: 50,  detail: hotelNights + " nights", bonus: "" },
    { name: "Standard Hotel 🏨",  tier: "Standard", base: 120, detail: hotelNights + " nights", bonus: "" },
    { name: "VIP Suite 🌟",       tier: "VIP",      base: 300, detail: hotelNights + " nights", bonus: "+10% concert score" }
  ];
  hotels.forEach(function(h) {
    h.price = h.base * hotelNights;
  });

  var selectedFlight = -1;
  var selectedHotel = -1;

  render();

  function render() {
    var html = '<div class="booking-header">';
    html += '<h2>' + artist.emoji + ' ' + artist.name + '</h2>';
    html += '<div class="concert-info">' + c.city + ', ' + c.country + ' — ' + c.date + '</div>';
    html += '<div class="concert-info">' + c.venue + '</div>';
    html += '<div style="color:var(--teal);font-size:0.8rem;margin-top:4px">' + daysUntil + ' days away</div>';
    html += '</div>';

    // Flights
    html += '<div class="booking-section">';
    html += '<h3>✈️ Choose Flight</h3>';
    html += '<div class="booking-options">';
    flights.forEach(function(f, i) {
      var sel = (i === selectedFlight) ? ' selected' : '';
      html += '<div class="booking-option' + sel + '" data-type="flight" data-idx="' + i + '">';
      html += '<div><div class="bo-name">' + f.name + '</div><div class="bo-detail">' + f.detail + '</div></div>';
      html += '<div class="bo-price">€' + f.price + '</div>';
      html += '</div>';
    });
    html += '</div></div>';

    // Hotels
    html += '<div class="booking-section">';
    html += '<h3>🏨 Choose Hotel</h3>';
    html += '<div class="booking-options">';
    hotels.forEach(function(h, i) {
      var sel = (i === selectedHotel) ? ' selected' : '';
      html += '<div class="booking-option' + sel + '" data-type="hotel" data-idx="' + i + '">';
      html += '<div><div class="bo-name">' + h.name + '</div>';
      html += '<div class="bo-detail">' + h.detail;
      if (h.bonus) html += ' · <span style="color:var(--gold)">' + h.bonus + '</span>';
      html += '</div></div>';
      html += '<div class="bo-price">€' + h.price + '</div>';
      html += '</div>';
    });
    html += '</div></div>';

    // Total
    var total = 0;
    if (selectedFlight >= 0) total += flights[selectedFlight].price;
    if (selectedHotel >= 0) total += hotels[selectedHotel].price;

    var canAfford = (s.budget >= total && total > 0);
    var travelPts = 0;
    if (selectedFlight >= 0) travelPts += 50;
    if (selectedHotel >= 0) travelPts += 20 * hotelNights;

    html += '<div class="booking-total">';
    html += '<div>Total</div>';
    html += '<div class="total-price">€' + total.toLocaleString() + '</div>';
    if (travelPts > 0) html += '<div style="color:var(--teal);font-size:0.8rem">+' + travelPts + ' travel points</div>';
    if (!canAfford && total > 0) html += '<div style="color:var(--danger);font-size:0.8rem;margin-top:4px">Not enough budget!</div>';
    html += '</div>';

    html += '<div class="booking-btns">';
    html += '<button class="btn btn-secondary" id="booking-back">← Back</button>';
    var bookDisabled = (selectedFlight < 0 || selectedHotel < 0 || !canAfford) ? ' disabled' : '';
    html += '<button class="btn btn-primary"' + bookDisabled + ' id="booking-confirm">Book Now! 🎫</button>';
    html += '</div>';

    el.innerHTML = html;

    // Attach event listeners
    el.querySelectorAll('.booking-option').forEach(function(opt) {
      opt.addEventListener('click', function() {
        var type = this.getAttribute('data-type');
        var idx = parseInt(this.getAttribute('data-idx'));
        G.sfx.click();
        if (type === 'flight') selectedFlight = idx;
        else selectedHotel = idx;
        render();
      });
    });

    document.getElementById('booking-back').addEventListener('click', function() {
      G.sfx.click();
      G.showScreen('screen-map');
    });

    var confirmBtn = document.getElementById('booking-confirm');
    if (confirmBtn && !confirmBtn.disabled) {
      confirmBtn.addEventListener('click', function() {
        confirmBooking(total, travelPts);
      });
    }
  }

  function confirmBooking(total, travelPts) {
    G.sfx.success();
    G.spendBudget(total);
    G.addPoints(travelPts);

    s.booked[concertId] = {
      flightIdx: selectedFlight,
      hotelIdx: selectedHotel,
      flightCost: flights[selectedFlight].price,
      hotelCost: hotels[selectedHotel].price,
      hotelTier: hotels[selectedHotel].tier
    };

    // Advance 2-5 days for booking
    G.advanceDays(Math.floor(Math.random() * 4) + 2);

    // Show confirmation
    el.innerHTML = '<div class="fade-in" style="text-align:center;padding:40px 16px">' +
      '<h2 style="color:var(--gold);margin-bottom:16px">✅ Booking Confirmed!</h2>' +
      '<p style="margin:8px 0">' + artist.emoji + ' ' + artist.name + '</p>' +
      '<p style="color:var(--teal)">' + c.city + ' — ' + c.date + '</p>' +
      '<p style="margin:8px 0">' + flights[selectedFlight].name + ' + ' + hotels[selectedHotel].name + '</p>' +
      '<p style="color:var(--gold);margin:12px 0">-€' + total + ' · +' + travelPts + ' pts</p>' +
      '<button class="btn btn-primary" id="booking-done" style="margin-top:16px">Back to Map 🌍</button>' +
      '</div>';

    document.getElementById('booking-done').addEventListener('click', function() {
      G.sfx.click();
      G.showScreen('screen-map');
    });
  }
};
