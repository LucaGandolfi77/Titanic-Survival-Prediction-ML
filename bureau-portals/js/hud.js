export class HUDManager {
  constructor(player, puzzle) {
    this.player = player;
    this.puzzle = puzzle;
    
    this.hudElement = document.getElementById('hud');
    this.sanityBar = document.getElementById('sanity-bar-inner');
    this.sanityValue = document.getElementById('sanity-value');
    this.sanityStatus = document.getElementById('sanity-status');
    this.batteryBar = document.getElementById('battery-bar-inner');
    this.batteryValue = document.getElementById('battery-value');
    this.inventorySlots = document.getElementById('inventory-slots');
    this.objectiveText = document.getElementById('objective-text');
    this.objectiveStamps = document.getElementById('objective-stamps');
    this.mapCanvas = document.getElementById('map-canvas');
    this.currentRoomLabel = document.getElementById('current-room-label');
    this.eventNotifications = document.getElementById('event-notifications');
    this.crosshair = document.getElementById('crosshair');
    this.interactionHint = document.getElementById('interaction-hint');
    
    this.hintKey = document.getElementById('hint-key');
    this.hintText = document.getElementById('hint-text');
    
    this.mapCtx = this.mapCanvas.getContext('2d');
    this.roomLayout = this.buildRoomLayout();
    
    this.updateInterval = 0.1;
    this.timeSinceUpdate = 0;
  }

  buildRoomLayout() {
    // Simple 2D representation of room connections
    return {
      0: { name: 'LOBBY', x: 70, y: 70, color: '#4488ff' },
      1: { name: 'RECORDS', x: 20, y: 70, color: '#4488ff' },
      2: { name: 'INVERTED', x: 120, y: 120, color: '#ff2244' },
      3: { name: 'CORRIDOR', x: 50, y: 30, color: '#22ff88' },
      4: { name: 'BREAK', x: 120, y: 70, color: '#ff8800' },
      5: { name: 'VOID', x: 70, y: 120, color: '#000000' },
      6: { name: 'DIRECTOR', x: 20, y: 30, color: '#4488ff' },
      7: { name: 'COPY', x: 120, y: 30, color: '#4488ff' },
      8: { name: 'CONFERENCE', x: 50, y: 70, color: '#4488ff' },
      9: { name: 'SERVER', x: 30, y: 120, color: '#aa44ff' },
      10: { name: 'MAINT', x: 100, y: 120, color: '#4488ff' },
      11: { name: 'EXIT', x: 70, y: 10, color: '#4488ff' }
    };
  }

  show() {
    this.hudElement.classList.remove('hidden');
  }

  hide() {
    this.hudElement.classList.add('hidden');
  }

  update(dt) {
    this.timeSinceUpdate += dt;
    if (this.timeSinceUpdate < this.updateInterval) return;
    this.timeSinceUpdate = 0;

    this.updateSanity(this.player.sanity);
    this.updateBattery(this.player.flashlightBattery);
    this.updateInventory();
    this.updateObjectives();
    this.updateMap(this.player.currentRoom);
    this.updateCrosshair();
  }

  updateSanity(value) {
    const percentage = Math.max(0, Math.min(100, (value + 100) / 2)); // Remap -100 to 100 as 0 to 100
    this.sanityBar.style.width = `${percentage}%`;
    this.sanityValue.textContent = `${Math.round(value)} / 100`;

    let status = 'ACCEPTABLE';
    let color = '#22c55e';

    if (value < 70) {
      status = 'MODERATE CONCERN';
      color = '#f59e0b';
    }
    if (value < 50) {
      status = 'SIGNIFICANT IRREGULARITY';
      color = '#f59e0b';
    }
    if (value < 30) {
      status = 'CRITICAL DEVIATION';
      color = '#ef4444';
    }
    if (value < 10) {
      status = 'NON-COMPLIANT';
      color = '#ef4444';
    }
    if (value < 0) {
      status = 'BEHAVIORAL ANOMALY';
      color = '#bb00ff';
    }

    this.sanityStatus.textContent = status;
    this.sanityBar.style.backgroundColor = color;
  }

  updateBattery(percentage) {
    this.batteryBar.style.width = `${percentage}%`;
    this.batteryValue.textContent = `${Math.round(percentage)}%`;

    if (percentage > 60) {
      this.batteryBar.style.backgroundColor = '#22c55e';
    } else if (percentage > 20) {
      this.batteryBar.style.backgroundColor = '#f59e0b';
    } else {
      this.batteryBar.style.backgroundColor = '#ef4444';
    }
  }

  updateInventory() {
    this.inventorySlots.innerHTML = '';
    
    for (let i = 0; i < 6; i++) {
      const slot = document.createElement('div');
      slot.className = 'inventory-slot';
      
      if (i < this.puzzle.inventory.length) {
        const item = this.puzzle.inventory[i];
        const icon = this.getItemIcon(item.type);
        slot.textContent = icon;
        slot.title = item.type;
        
        const tooltip = document.createElement('div');
        tooltip.className = 'inventory-slot-tooltip';
        tooltip.textContent = item.type;
        slot.appendChild(tooltip);
      }
      
      this.inventorySlots.appendChild(slot);
    }
  }

  updateObjectives() {
    const currentObj = this.puzzle.objectives.find(o => !o.completed);
    if (currentObj) {
      this.objectiveText.textContent = currentObj.text;
    }

    this.objectiveStamps.innerHTML = '';
    for (let completed of this.puzzle.completedObjectives) {
      const stamp = document.createElement('div');
      stamp.className = 'objective-stamp';
      stamp.textContent = '✓';
      this.objectiveStamps.appendChild(stamp);
    }
  }

  updateMap(currentRoom) {
    this.mapCtx.clearRect(0, 0, this.mapCanvas.width, this.mapCanvas.height);
    
    // Background
    this.mapCtx.fillStyle = '#1a1a2e';
    this.mapCtx.fillRect(0, 0, this.mapCanvas.width, this.mapCanvas.height);

    // Draw rooms
    for (let roomId in this.roomLayout) {
      const room = this.roomLayout[roomId];
      const size = 6;
      
      this.mapCtx.fillStyle = room.color;
      this.mapCtx.fillRect(room.x - size/2, room.y - size/2, size, size);

      // Current room highlight
      if (currentRoom && currentRoom.id === parseInt(roomId)) {
        this.mapCtx.strokeStyle = '#ffff00';
        this.mapCtx.lineWidth = 2;
        this.mapCtx.strokeRect(room.x - size/2 - 2, room.y - size/2 - 2, size + 4, size + 4);
      }
    }

    // Update label
    if (currentRoom) {
      const room = this.roomLayout[currentRoom.id];
      if (room) {
        this.currentRoomLabel.textContent = `ROOM: ${room.name}`;
      }
    }
  }

  updateCrosshair() {
    // Default: blue dot
    const dot = this.crosshair.querySelector('#crosshair-dot');
    if (dot) {
      dot.style.background = '#4488ff';
    }

    // Check for nearby interactions
    if (window.game) {
      const item = window.game.itemManager?.checkInteraction(this.player.position, 2.0);
      if (item) {
        this.crosshair.classList.add('interact');
        this.showInteractionHint('[E] Pick up ' + item.type);
      } else {
        this.crosshair.classList.remove('interact');
        this.hideInteractionHint();
      }
    }
  }

  showInteractionHint(text) {
    this.hintText.textContent = text;
    this.interactionHint.classList.remove('hidden');
  }

  hideInteractionHint() {
    this.interactionHint.classList.add('hidden');
  }

  showNotification(message, type = 'info') {
    const notif = document.createElement('div');
    notif.className = `event-notification ${type}`;
    notif.textContent = message;
    
    this.eventNotifications.appendChild(notif);

    setTimeout(() => {
      notif.remove();
    }, 2000);
  }

  getItemIcon(type) {
    const icons = {
      'form': '📄',
      'key': '🔑',
      'stamp': '🔴',
      'coffee': '☕',
      'minutes': '📋',
      'battery': '🔋',
      'note': '📝',
      'calibration': '⚙️'
    };
    return icons[type] || '?';
  }

  toggleInventoryView() {
    // Could expand inventory here
  }
}