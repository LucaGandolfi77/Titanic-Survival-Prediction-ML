/* global localStorage */

export class SaveSystem {
  constructor(key = 'bureau-portals-save') {
    this.key = key;
  }

  save(data) {
    try {
      const payload = { version: 1, timestamp: Date.now(), data };
      localStorage.setItem(this.key, JSON.stringify(payload));
      return true;
    } catch (e) {
      return false;
    }
  }

  load() {
    try {
      const raw = localStorage.getItem(this.key);
      if (!raw) return null;
      const payload = JSON.parse(raw);
      if (payload.version !== 1) return null;
      return payload.data;
    } catch (e) {
      return null;
    }
  }

  delete() {
    localStorage.removeItem(this.key);
  }

  hasSave() {
    return localStorage.getItem(this.key) !== null;
  }

  serialize(game) {
    const player = game.player;
    const puzzle = game.puzzle;
    const npcs = game.npcs;

    return {
      player: {
        position: player.position.toArray(),
        yaw: player.yaw,
        pitch: player.pitch,
        sanity: player.sanity,
        flashlightBattery: player.flashlightBattery,
        hasFlashlight: player.hasFlashlight
      },
      puzzle: {
        inventory: puzzle.inventory.map((item) => ({ type: item.type, data: item.data })),
        completedObjectives: puzzle.completedObjectives,
        objectives: puzzle.objectives.map((o) => ({ id: o.id, completed: o.completed })),
        loopLeverPulled: puzzle.loopLeverPulled
      },
      portalTypes: game.portals.portals.map((p) => p.type),
      auditorAppeared: npcs.auditorAppeared,
      auditorAlive: !!npcs.auditor,
      currentRoomId: game.player.currentRoom ? game.player.currentRoom.id : 0,
      state: game.state,
      formOpen: game.formOpen,
      formFields: game.formFields
    };
  }

  deserialize(game, saveData) {
    if (!saveData) return false;

    const player = game.player;
    if (saveData.player) {
      if (saveData.player.position) player.position.fromArray(saveData.player.position);
      if (typeof saveData.player.yaw === 'number') player.yaw = saveData.player.yaw;
      if (typeof saveData.player.pitch === 'number') player.pitch = saveData.player.pitch;
      if (typeof saveData.player.sanity === 'number') player.sanity = saveData.player.sanity;
      if (typeof saveData.player.flashlightBattery === 'number')
        player.flashlightBattery = saveData.player.flashlightBattery;
      if (typeof saveData.player.hasFlashlight === 'boolean')
        player.hasFlashlight = saveData.player.hasFlashlight;
    }

    const puzzle = game.puzzle;
    if (saveData.puzzle) {
      if (Array.isArray(saveData.puzzle.inventory)) puzzle.inventory = saveData.puzzle.inventory;
      if (Array.isArray(saveData.puzzle.completedObjectives))
        puzzle.completedObjectives = saveData.puzzle.completedObjectives;
      if (Array.isArray(saveData.puzzle.objectives)) {
        puzzle.objectives = puzzle.objectives.map((o) => {
          const saved = saveData.puzzle.objectives.find((s) => s.id === o.id);
          if (saved) o.completed = saved.completed;
          return o;
        });
      }
      if (typeof saveData.puzzle.loopLeverPulled === 'boolean')
        puzzle.loopLeverPulled = saveData.puzzle.loopLeverPulled;
    }

    if (saveData.portalTypes && Array.isArray(saveData.portalTypes)) {
      saveData.portalTypes.forEach((type, idx) => {
        if (game.portals.portals[idx]) game.portals.portals[idx].type = type;
      });
    }

    if (saveData.npcs) {
      if (typeof saveData.npcs.auditorAppeared === 'boolean')
        game.npcs.auditorAppeared = saveData.npcs.auditorAppeared;
      if (saveData.npcs.auditorAlive && !game.npcs.auditor) {
        game.npcs.spawnAuditor(game.player);
      }
    }

    if (saveData.formFields) {
      game.formFields = saveData.formFields;
      game.formOpen = !!saveData.formOpen;
    }

    return true;
  }
}
