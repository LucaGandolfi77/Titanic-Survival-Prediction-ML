/* photo.js — Photo mode with artistic filters */
window.G = window.G || {}

G.photoMode = function (concertId, canvasElement, onComplete) {
  var el = document.getElementById('concert-content')
  var W = canvasElement.width
  var H = canvasElement.height

  // Get canvas pixels for filter application
  var ctx = canvasElement.getContext('2d')
  var imageData = ctx.getImageData(0, 0, W, H)

  var filters = ['none', 'bw', 'sepia', 'vignette', 'shift']
  var filterNames = { none: 'Original', bw: 'B&W', sepia: 'Sepia', vignette: 'Vignette', shift: 'Color Shift' }

  var html = '<div class="cg-container fade-in">'
  html += '<div class="cg-title">📷 Photo Mode</div>'
  html += '<div class="cg-instruction">Choose a filter and save your concert moment!</div>'
  html += '<div id="photo-filter-name" style="color:var(--gold);margin:8px 0;font-weight:700">Original</div>'
  html += '<div class="photo-filters" id="photo-filters">'
  filters.forEach(function (f) {
    html +=
      '<button class="photo-filter-btn' +
      (f === 'none' ? ' active' : '') +
      '" data-filter="' +
      f +
      '">' +
      filterNames[f] +
      '</button>'
  })
  html += '</div>'
  html += '<div class="cg-canvas-wrap"><canvas id="photo-canvas" width="' + W + '" height="' + H + '"></canvas></div>'
  html += '<div style="display:flex;gap:10px;justify-content:center;margin-top:12px;flex-wrap:wrap">'
  html += '<button class="btn btn-primary" id="photo-save"><i data-lucide="download"></i> Save Photo</button>'
  html += '<button class="btn btn-secondary" id="photo-back"><i data-lucide="arrow-left"></i> Back</button>'
  html += '<button class="btn btn-secondary" id="photo-share"><i data-lucide="share-2"></i> Share</button>'
  html += '</div>'
  html += '</div>'
  el.innerHTML = html

  if (window.lucide) lucide.createIcons()

  // Render filtered canvas
  var photoCanvas = document.getElementById('photo-canvas')
  var pCtx = G.setupHDCanvas(photoCanvas, W, H)

  function applyFilter(filter, srcData) {
    var data = new Uint8ClampedArray(srcData)
    var len = data.length
    var i

    if (filter === 'bw') {
      for (i = 0; i < len; i += 4) {
        var gray = data[i] * 0.3 + data[i + 1] * 0.59 + data[i + 2] * 0.11
        data[i] = gray
        data[i + 1] = gray
        data[i + 2] = gray
      }
    } else if (filter === 'sepia') {
      for (i = 0; i < len; i += 4) {
        var r = data[i]
        var g = data[i + 1]
        var b = data[i + 2]
        data[i] = Math.min(255, r * 0.393 + g * 0.769 + b * 0.189)
        data[i + 1] = Math.min(255, r * 0.349 + g * 0.686 + b * 0.168)
        data[i + 2] = Math.min(255, r * 0.272 + g * 0.534 + b * 0.131)
      }
    } else if (filter === 'vignette') {
      for (i = 0; i < len; i += 4) {
        var x = (i / 4) % W
        var y = Math.floor(i / 4 / W)
        var dx = x - W / 2
        var dy = y - H / 2
        var dist = Math.sqrt(dx * dx + dy * dy) / (Math.sqrt(W * W + H * H) / 2)
        var vignette = 1 - dist * 0.6
        data[i] *= vignette
        data[i + 1] *= vignette
        data[i + 2] *= vignette
      }
    } else if (filter === 'shift') {
      for (i = 0; i < len; i += 4) {
        var tmp = data[i]
        data[i] = data[i + 1]
        data[i + 1] = data[i + 2]
        data[i + 2] = tmp
      }
    }
    return new ImageData(data, W, H)
  }

  function renderPhoto(filter) {
    var filtered = applyFilter(filter, imageData.data)
    pCtx.putImageData(filtered, 0, 0)
    var nameEl = document.getElementById('photo-filter-name')
    if (nameEl) nameEl.textContent = filterNames[filter]
  }

  renderPhoto('none')

  // Filter buttons
  el.querySelectorAll('.photo-filter-btn').forEach(function (btn) {
    btn.addEventListener('click', function () {
      var filter = this.getAttribute('data-filter')
      el.querySelectorAll('.photo-filter-btn').forEach(function (b) {
        b.classList.remove('active')
      })
      this.classList.add('active')
      renderPhoto(filter)
      G.sfx.click()
    })
  })

  // Save photo
  document.getElementById('photo-save').addEventListener('click', function () {
    try {
      var link = document.createElement('a')
      link.download = 'concerts-chase-photo.png'
      link.href = photoCanvas.toDataURL('image/png')
      link.click()
      G.sfx.success()
      G.toast('📷 Photo saved!')
    } catch (e) {
      G.sfx.fail()
      G.toast('Could not save photo')
    }
  })

  // Share photo
  document.getElementById('photo-share').addEventListener('click', function () {
    try {
      var text = '🎵 I just scored ' + (G.concertScores ? G.concertScores[concertId] : '') + ' at a concert!'
      if (navigator.share) {
        navigator
          .share({
            title: 'Concerts Chase Photo',
            text: text,
            url: photoCanvas.toDataURL('image/png')
          })
          .catch(function () {})
      } else {
        G.clipboardWrite(photoCanvas.toDataURL('image/png'))
        G.toast('📋 Photo link copied!')
      }
    } catch (e) {
      G.toast('Share not available')
    }
  })

  // Back
  document.getElementById('photo-back').addEventListener('click', function () {
    if (onComplete) onComplete()
  })
}

G.clipboardWrite = function (text) {
  if (navigator.clipboard && navigator.clipboard.writeText) {
    navigator.clipboard.writeText(text).catch(function () {})
  }
}
