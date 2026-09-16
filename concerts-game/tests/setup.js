global.window = { G: {} }
global.G = window.G
global.localStorage = {
  _data: {},
  getItem: function (k) { return this._data[k] || null },
  setItem: function (k, v) { this._data[k] = String(v) },
  removeItem: function (k) { delete this._data[k] },
  clear: function () { this._data = {} }
}
global.document = {
  getElementById: function () { return null },
  querySelector: function () { return null },
  createElement: function () { return {} },
  addEventListener: function () {}
}
