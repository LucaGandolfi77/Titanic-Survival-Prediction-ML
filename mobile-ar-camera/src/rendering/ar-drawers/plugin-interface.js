/**
 * AR Filter Plugin Interface
 * Third-party developers can implement this interface to add custom AR filters
 * without modifying the core codebase.
 */
export class ARFilterPlugin {
  constructor(config) {
    this.id = config.id;
    this.name = config.name;
    this.category = config.category;
    this.tracker = config.tracker;
    this.enabled = false;
  }

  async load() {
    throw new Error('Plugin must implement load()');
  }

  async unload() {
    throw new Error('Plugin must implement unload()');
  }

  update(/* mediaResult, arState, dt */) {
    throw new Error('Plugin must implement update()');
  }

  draw(/* ctx, width, height */) {
    throw new Error('Plugin must implement draw()');
  }
}
