import { EventBus } from './utils.js';

const bus = new EventBus();
bus.debug = false;

const originalEmit = bus.emit.bind(bus);
bus.emit = function (event, data) {
  if (this.debug) {
    console.log(`[BUS] ${event}`, data);
  }
  originalEmit(event, data);
};

export { bus };
export default bus;
