global.document = {
  createElement: () => ({
    width: 0,
    height: 0,
    getContext: () => ({ drawImage: () => {}, clearRect: () => {}, fillRect: () => {}, beginPath: () => {}, arc: () => {}, fill: () => {}, stroke: () => {}, moveTo: () => {}, lineTo: () => {}, strokeText: () => {}, fillText: () => {} }),
    toBlob: (cb) => cb(null),
    getContext2d: null,
  }),
};
global.window = {
  innerWidth: 1920,
  innerHeight: 1080,
  devicePixelRatio: 1,
  addEventListener: () => {},
  removeEventListener: () => {},
  requestAnimationFrame: () => {},
  cancelAnimationFrame: () => {},
};
