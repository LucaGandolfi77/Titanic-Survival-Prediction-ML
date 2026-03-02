export class PuzzleManager {
  constructor(player, itemManager, npcManager) {
    this.player = player;
    this.itemManager = itemManager;
    this.npcManager = npcManager;
    
    this.inventory = [];
    this.maxInventory = 6;
    this.objectives = [];
    this.completedObjectives = [];
    
    this.loopLeverPulled = false;
    this.combinationLockSolved = false;
    this.currentCombination = '0000';
    this.correctCombination = '4471'; // Employee number
    
    this.setupObjectives();
  }

  setupObjectives() {
    this.objectives = [
      { id: 'find_form', text: 'Find Form 27-Γ (Records Department)', completed: false },
      { id: 'find_key', text: 'Find Cabinet Key (Void Office)', completed: false },
      { id: 'find_stamp', text: 'Find Rubber Stamp', completed: false },
      { id: 'find_coffee', text: 'Find Director\'s Coffee (Break Room)', completed: false },
      { id: 'find_minutes', text: 'Find Meeting Minutes (Conference Room)', completed: false },
      { id: 'find_master_key', text: 'Find Master Key (Director\'s Office)', completed: false },
      { id: 'find_calibration', text: 'Find Portal Calibration Device (Server Room)', completed: false },
      { id: 'break_loop', text: 'Break the Infinite Loop (Pull Lever)', completed: false },
      { id: 'final', text: 'File Form 27-Γ at EXIT DESK', completed: false }
    ];
  }

  addItemToInventory(item) {
    if (this.inventory.length < this.maxInventory) {
      this.inventory.push(item);
      
      // Update objectives
      if (item.type === 'form' && !this.getObjective('find_form').completed) {
        this.completeObjective('find_form');
      } else if (item.type === 'key' && !this.getObjective('find_key').completed && !item.data.ismaster) {
        this.completeObjective('find_key');
      } else if (item.type === 'stamp' && !this.getObjective('find_stamp').completed) {
        this.completeObjective('find_stamp');
      } else if (item.type === 'coffee' && !this.getObjective('find_coffee').completed) {
        this.completeObjective('find_coffee');
      } else if (item.type === 'minutes' && !this.getObjective('find_minutes').completed) {
        this.completeObjective('find_minutes');
      } else if (item.type === 'key' && item.data.ismaster && !this.getObjective('find_master_key').completed) {
        this.completeObjective('find_master_key');
      } else if (item.type === 'calibration' && !this.getObjective('find_calibration').completed) {
        this.completeObjective('find_calibration');
      }
      
      return true;
    }
    return false;
  }

  completeObjective(id) {
    const obj = this.getObjective(id);
    if (obj) {
      obj.completed = true;
      this.completedObjectives.push(id);
      
      if (window.game && window.game.ui) {
        window.game.ui.showNotification(`OBJECTIVE: ${obj.text}`, 'success');
      }
    }
  }

  getObjective(id) {
    return this.objectives.find(o => o.id === id);
  }

  pulledLoopLever() {
    if (!this.loopLeverPulled) {
      this.loopLeverPulled = true;
      this.completeObjective('break_loop');
      this.player.modifySanity(10); // Reward
      return true;
    }
    return false;
  }

  solveCombinationLock(code) {
    if (code === this.correctCombination) {
      this.combinationLockSolved = true;
      if (window.game && window.game.audio) {
        window.game.audio.playClick();
      }
      return true;
    }
    return false;
  }

  canExitGame() {
    // Check if all required items are in inventory
    const hasForm = this.inventory.some(i => i.type === 'form');
    const hasStamp = this.inventory.some(i => i.type === 'stamp');
    const hasKey = this.inventory.some(i => i.type === 'key');
    const hasCoffee = this.inventory.some(i => i.type === 'coffee');
    const hasMinutes = this.inventory.some(i => i.type === 'minutes');
    
    return hasForm && hasStamp && hasKey && hasCoffee && hasMinutes;
  }

  attemptExit() {
    if (this.canExitGame()) {
      this.completeObjective('final');
      return { success: true, message: 'FORM 27-Γ APPROVED. YOUR EMPLOYMENT HAS BEEN TERMINATED.' };
    } else {
      const missing = [];
      if (!this.inventory.some(i => i.type === 'form')) missing.push('Form 27-Γ');
      if (!this.inventory.some(i => i.type === 'stamp')) missing.push('Rubber Stamp');
      if (!this.inventory.some(i => i.type === 'key')) missing.push('Cabinet Key');
      if (!this.inventory.some(i => i.type === 'coffee')) missing.push('Director\'s Coffee');
      if (!this.inventory.some(i => i.type === 'minutes')) missing.push('Meeting Minutes');
      
      return { success: false, message: `MISSING: ${missing.join(', ')}` };
    }
  }

  findAllNotes() {
    return this.inventory.filter(i => i.type === 'note');
  }

  hasSecretEnding() {
    return this.findAllNotes().length === 8;
  }
}