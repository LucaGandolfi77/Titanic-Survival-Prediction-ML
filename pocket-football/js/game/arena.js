const SURFACE_TYPES = {
  GRASS: { name: 'Erba', color: '#4a8f3f', stripeColor: '#3f7a35', friction: 0.97, bounce: 0.6, speedMultiplier: 1.0 },
  DIRT: { name: 'Terra', color: '#c4a35a', stripeColor: '#a08440', friction: 0.94, bounce: 0.5, speedMultiplier: 0.9 },
  ICE: { name: 'Ghiaccio', color: '#a8d8ea', stripeColor: '#8ecae6', friction: 0.99, bounce: 0.3, speedMultiplier: 1.3 },
  CONCRETE: { name: 'Cemento', color: '#808080', stripeColor: '#6e6e6e', friction: 0.96, bounce: 0.5, speedMultiplier: 1.1 },
  SAND: { name: 'Sabbia', color: '#f4d35e', stripeColor: '#e6c252', friction: 0.92, bounce: 0.4, speedMultiplier: 0.75 }
};

const ENVIRONMENT_TYPES = {
  CLEAR: { name: 'Sereno', skyColor: '#87CEEB', ambient: 1.0 },
  CLOUDY: { name: 'Nuvoloso', skyColor: '#b0b0b0', ambient: 0.8 },
  RAIN: { name: 'Pioggia', skyColor: '#4a5568', ambient: 0.6 },
  NIGHT: { name: 'Notte', skyColor: '#1a1a3e', ambient: 0.4 }
};

class ArenaManager {
  constructor(storageService) {
    this.storage = storageService;
    this.currentArena = null;
  }

  async init() {
    await this.storage.ready();
    const saved = await this.storage.getSetting('currentArena');
    if (saved) {
      this.currentArena = saved;
    } else {
      this.currentArena = this._defaultArena();
    }
  }

  _defaultArena() {
    return {
      id: 'default',
      name: 'Stadio Standard',
      surface: 'GRASS',
      environment: 'CLEAR',
      fieldWidth: 800,
      fieldHeight: 500,
      hasFloodlights: false,
      crowdDensity: 0.7
    };
  }

  getAvailableArenas() {
    return Object.entries(SURFACE_TYPES).map(([key, surface]) => ({
      key,
      ...surface
    }));
  }

  getEnvironments() {
    return Object.entries(ENVIRONMENT_TYPES).map(([key, env]) => ({
      key,
      ...env
    }));
  }

  async createArena(name, surface, environment, options = {}) {
    const arena = {
      id: `arena_${Date.now()}`,
      name,
      surface,
      environment,
      fieldWidth: options.fieldWidth || 800,
      fieldHeight: options.fieldHeight || 500,
      hasFloodlights: options.hasFloodlights || false,
      crowdDensity: options.crowdDensity || 0.7,
      createdAt: new Date().toISOString()
    };

    await this.storage.saveArena(arena);
    this.currentArena = arena;
    await this.storage.setSetting('currentArena', arena);
    return arena;
  }

  async setCurrentArena(arenaId) {
    const arenas = await this.storage.getArenas();
    const arena = arenas.find(a => a.id === arenaId);
    if (arena) {
      this.currentArena = arena;
      await this.storage.setSetting('currentArena', arena);
    }
    return arena;
  }

  getCurrentArena() {
    return this.currentArena || this._defaultArena();
  }

  getSurfaceProperties(surfaceKey) {
    return SURFACE_TYPES[surfaceKey] || SURFACE_TYPES.GRASS;
  }

  getEnvironmentProperties(envKey) {
    return ENVIRONMENT_TYPES[envKey] || ENVIRONMENT_TYPES.CLEAR;
  }

  getFieldColors() {
    const surface = this.getSurfaceProperties(this.currentArena?.surface);
    const env = this.getEnvironmentProperties(this.currentArena?.environment);
    return {
      fieldPrimary: surface.color,
      fieldStripe: surface.stripeColor,
      skyColor: env.skyColor,
      ambient: env.ambient
    };
  }

  getPhysicsModifiers() {
    const surface = this.getSurfaceProperties(this.currentArena?.surface);
    return {
      frictionMultiplier: surface.friction,
      bounceMultiplier: surface.bounce,
      speedMultiplier: surface.speedMultiplier
    };
  }

  async saveCustomArena(arena) {
    await this.storage.saveArena(arena);
  }

  async getAllCustomArenas() {
    const all = await this.storage.getArenas();
    return all.filter(a => a.id !== 'default');
  }
}

export { ArenaManager, SURFACE_TYPES, ENVIRONMENT_TYPES };
