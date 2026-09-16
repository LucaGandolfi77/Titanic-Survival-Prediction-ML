/* merch.js — Merch shop screen */
window.G = window.G || {}

G.showMerch = function () {
  var s = G.state
  var el = document.getElementById('merch-content')
  G.showScreen('screen-merch')

  var html = '<div style="text-align:center;padding:16px">'
  html += '<h3 style="color:var(--gold);margin-bottom:16px">Points: ' + s.points + '</h3>'
  html += '<div style="display:flex;flex-direction:column;gap:12px;max-width:480px;margin:0 auto">'

  G.MERCH_ITEMS.forEach(function (item) {
    var owned = s.merchPurchases && s.merchPurchases.indexOf(item.id) >= 0
    var canAfford = s.points >= item.cost && !owned
    html += '<div class="merch-item' + (owned ? ' owned' : '') + '">'
    html += '<div class="merch-emoji">' + item.emoji + '</div>'
    html += '<div class="merch-info">'
    html += '<div class="merch-name">' + item.name + '</div>'
    html += '<div class="merch-desc">' + item.desc + '</div>'
    html += '<div class="merch-cost">' + item.emoji + ' ' + item.cost + ' pts</div>'
    html += '</div>'
    if (owned) {
      html += '<div class="merch-btn owned">✓ Owned</div>'
    } else {
      html +=
        '<button class="btn btn-sm btn-primary merch-buy" data-merch="' +
        item.id +
        '"' +
        (canAfford ? '' : ' disabled') +
        '>Buy</button>'
    }
    html += '</div>'
  })

  html += '</div>'
  html +=
    '<button class="btn btn-secondary" style="margin-top:20px" onclick="G.showScreen(\'screen-map\')">← Back to Map</button>'
  html += '</div>'
  el.innerHTML = html

  el.querySelectorAll('.merch-buy').forEach(function (btn) {
    btn.addEventListener('click', function () {
      var mId = this.getAttribute('data-merch')
      buyMerch(mId)
    })
  })
}

function buyMerch(mId) {
  var item = G.MERCH_ITEMS.find(function (m) {
    return m.id === mId
  })
  if (!item) return
  if (G.state.points < item.cost) return
  if (G.state.merchPurchases.indexOf(mId) >= 0) return

  G.state.points -= item.cost
  G.state.merchPurchases.push(mId)
  G.sfx.success()
  G.toast(item.emoji + ' ' + item.name + ' purchased!')
  G.showMerch()
}
