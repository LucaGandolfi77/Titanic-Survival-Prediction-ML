/* travel.js — Travel upgrade system (integrated with booking) */
window.G = window.G || {}

G.showTravelUpgrades = function () {
  var s = G.state
  var el = document.getElementById('booking-content')
  G.showScreen('screen-booking')

  var html = '<div class="booking-header">'
  html += '<h2>✈️ Travel Upgrades</h2>'
  html += '<div style="color:var(--teal);margin-top:4px">Spend budget to unlock cheaper flights & hotels</div>'
  html += '</div>'

  // Flight upgrades
  html += '<div class="booking-section"><h3>✈️ Flight Tier</h3><div class="booking-options">'
  G.FLIGHT_TIERS.forEach(function (tier, i) {
    var selected = i === s.travelFlightTier ? ' selected' : ''
    var isUpgrade = i === s.travelFlightTier + 1
    var disabled = !isUpgrade ? ' disabled' : ''
    var unlockCost = isUpgrade ? G.getFlightUpgradeCost(i) : 0
    html += '<div class="booking-option' + selected + '" data-upgrade="flight" data-tier="' + i + '"' + disabled + '>'
    html +=
      '<div><div class="bo-name">' +
      tier.emoji +
      ' ' +
      tier.name +
      '</div><div class="bo-detail">' +
      tier.desc +
      '</div></div>'
    if (i === s.travelFlightTier) {
      html += '<div class="bo-price" style="color:var(--success)">✓ Active</div>'
    } else if (isUpgrade) {
      html += '<div class="bo-price">€' + unlockCost + '</div>'
    } else {
      html += '<div class="bo-price" style="opacity:0.3">Locked</div>'
    }
    html += '</div>'
  })
  html += '</div></div>'

  // Hotel upgrades
  html += '<div class="booking-section"><h3>🏨 Hotel Tier</h3><div class="booking-options">'
  G.HOTEL_TIERS.forEach(function (tier, i) {
    var selected = i === s.travelHotelTier ? ' selected' : ''
    var isUpgrade = i === s.travelHotelTier + 1
    var disabled = !isUpgrade ? ' disabled' : ''
    var unlockCost = isUpgrade ? G.getHotelUpgradeCost(i) : 0
    html += '<div class="booking-option' + selected + '" data-upgrade="hotel" data-tier="' + i + '"' + disabled + '>'
    html +=
      '<div><div class="bo-name">' +
      tier.emoji +
      ' ' +
      tier.name +
      '</div><div class="bo-detail">' +
      tier.desc +
      '</div></div>'
    if (i === s.travelHotelTier) {
      html += '<div class="bo-price" style="color:var(--success)">✓ Active</div>'
    } else if (isUpgrade) {
      html += '<div class="bo-price">€' + unlockCost + '</div>'
    } else {
      html += '<div class="bo-price" style="opacity:0.3">Locked</div>'
    }
    html += '</div>'
  })
  html += '</div></div>'

  html += '<button class="btn btn-secondary" id="travel-back" style="margin-top:16px">← Back to Map</button>'
  el.innerHTML = html

  el.querySelectorAll('.booking-option[data-upgrade]').forEach(function (opt) {
    opt.addEventListener('click', function () {
      var type = this.getAttribute('data-upgrade')
      var tier = parseInt(this.getAttribute('data-tier'))
      var cost = type === 'flight' ? G.getFlightUpgradeCost(tier) : G.getHotelUpgradeCost(tier)
      if (s.budget >= cost) {
        G.sfx.success()
        G.spendBudget(cost)
        if (type === 'flight') s.travelFlightTier = tier
        else s.travelHotelTier = tier
        G.toast('Upgraded to ' + (type === 'flight' ? G.FLIGHT_TIERS[tier].name : G.HOTEL_TIERS[tier].name))
        G.showTravelUpgrades()
      } else {
        G.sfx.fail()
        G.toast('Not enough budget!')
      }
    })
  })

  document.getElementById('travel-back').addEventListener('click', function () {
    G.sfx.click()
    G.showScreen('screen-map')
  })
}

G.getFlightUpgradeCost = function (tier) {
  return tier * 800
}

G.getHotelUpgradeCost = function (tier) {
  return tier * 600
}
