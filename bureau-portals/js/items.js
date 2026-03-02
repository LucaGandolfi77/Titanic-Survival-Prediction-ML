import * as THREE from 'https://cdnjs.cloudflare.com/ajax/libs/three.js/r158/three.min.js';
import { MathUtils } from './utils.js';

export class Item {
  constructor(scene, position, type, data = {}) {
    this.scene = scene;
    this.position = position.clone();
    this.type = type; // 'form', 'key', 'stamp', 'coffee', 'minutes', 'battery', 'note', 'calibration'
    this.data = data;
    this.collected = false;
    this.interactRadius = 2.0;
    
    this.group = new THREE.Group();
    this.group.position.copy(position);
    
    this.buildMesh();
    scene.add(this.group);
    
    // Floating animation
    this.floatOffset = 0;
    this.floatSpeed = 2;
  }

  buildMesh() {
    switch (this.type) {
      case 'form':
        this.buildForm();
        break;
      case 'key':
        this.buildKey();
        break;
      case 'stamp':
        this.buildStamp();
        break;
      case 'coffee':
        this.buildCoffee();
        break;
      case 'minutes':
        this.buildMinutes();
        break;
      case 'battery':
        this.buildBattery();
        break;
      case 'note':
        this.buildNote();
        break;
      case 'calibration':
        this.buildCalibration();
        break;
    }
  }

  buildForm() {
    const paper = new THREE.Mesh(
      new THREE.BoxGeometry(0.2, 0.3, 0.01),
      new THREE.MeshLambertMaterial({ color: 0xfffacd })
    );
    this.group.add(paper);

    const stamp = new THREE.Mesh(
      new THREE.BoxGeometry(0.1, 0.05, 0.01),
      new THREE.MeshBasicMaterial({ color: 0xc41e3a })
    );
    stamp.position.set(0.05, 0.08, 0.01);
    this.group.add(stamp);
  }

  buildKey() {
    const shaft = new THREE.Mesh(
      new THREE.CylinderGeometry(0.02, 0.02, 0.1),
      new THREE.MeshLambertMaterial({ color: 0xffd700 })
    );
    shaft.rotation.z = Math.PI / 2;
    this.group.add(shaft);

    const head = new THREE.Mesh(
      new THREE.SphereGeometry(0.04),
      new THREE.MeshLambertMaterial({ color: 0xffd700 })
    );
    head.position.x = -0.08;
    this.group.add(head);

    const teeth = new THREE.Mesh(
      new THREE.BoxGeometry(0.02, 0.04, 0.02),
      new THREE.MeshLambertMaterial({ color: 0xffd700 })
    );
    teeth.position.x = 0.08;
    this.group.add(teeth);
  }

  buildStamp() {
    const handle = new THREE.Mesh(
      new THREE.CylinderGeometry(0.05, 0.05, 0.08),
      new THREE.MeshLambertMaterial({ color: 0x8b4513 })
    );
    handle.position.y = 0.05;
    this.group.add(handle);

    const stampPad = new THREE.Mesh(
      new THREE.BoxGeometry(0.08, 0.02, 0.08),
      new THREE.MeshLambertMaterial({ color: 0xc41e3a })
    );
    stampPad.position.y = 0.01;
    this.group.add(stampPad);
  }

  buildCoffee() {
    const cup = new THREE.Mesh(
      new THREE.CylinderGeometry(0.05, 0.06, 0.08),
      new THREE.MeshLambertMaterial({ color: 0xffffff })
    );
    this.group.add(cup);

    const coffee = new THREE.Mesh(
      new THREE.CylinderGeometry(0.045, 0.055, 0.06),
      new THREE.MeshLambertMaterial({ color: 0x664422 })
    );
    coffee.position.y = 0.02;
    this.group.add(coffee);

    const handle = new THREE.Mesh(
      new THREE.TorusGeometry(0.04, 0.01),
      new THREE.MeshLambertMaterial({ color: 0xffffff })
    );
    handle.position.set(0.05, 0.02, 0);
    handle.rotation.z = Math.PI / 2;
    this.group.add(handle);
  }

  buildMinutes() {
    const paper = new THREE.Mesh(
      new THREE.BoxGeometry(0.15, 0.25, 0.01),
      new THREE.MeshLambertMaterial({ color: 0xf5f5f5 })
    );
    this.group.add(paper);

    // Binding
    const binding = new THREE.Mesh(
      new THREE.BoxGeometry(0.01, 0.26, 0.01),
      new THREE.MeshLambertMaterial({ color: 0x333333 })
    );
    binding.position.x = -0.07;
    this.group.add(binding);
  }

  buildBattery() {
    const body = new THREE.Mesh(
      new THREE.BoxGeometry(0.03, 0.06, 0.03),
      new THREE.MeshLambertMaterial({ color: 0x333333 })
    );
    this.group.add(body);

    const top = new THREE.Mesh(
      new THREE.BoxGeometry(0.02, 0.01, 0.02),
      new THREE.MeshBasicMaterial({ color: 0xff0000 })
    );
    top.position.y = 0.035;
    this.group.add(top);
  }

  buildNote() {
    const note = new THREE.Mesh(
      new THREE.BoxGeometry(0.1, 0.1, 0.01),
      new THREE.MeshLambertMaterial({ color: 0xffff00 })
    );
    this.group.add(note);

    // Pin
    const pin = new THREE.Mesh(
      new THREE.SphereGeometry(0.01),
      new THREE.MeshBasicMaterial({ color: 0xff0000 })
    );
    pin.position.set(0, 0.03, 0.01);
    this.group.add(pin);
  }

  buildCalibration() {
    const box = new THREE.Mesh(
      new THREE.BoxGeometry(0.15, 0.1, 0.08),
      new THREE.MeshLambertMaterial({ color: 0x4488ff })
    );
    this.group.add(box);

    const display = new THREE.Mesh(
      new THREE.BoxGeometry(0.1, 0.04, 0.01),
      new THREE.MeshBasicMaterial({ color: 0x00ff00 })
    );
    display.position.y = 0.04;
    this.group.add(display);

    const button = new THREE.Mesh(
      new THREE.CylinderGeometry(0.02, 0.02, 0.03),
      new THREE.MeshLambertMaterial({ color: 0xff0000 })
    );
    button.position.y = -0.02;
    this.group.add(button);
  }

  update(dt, playerPos) {
    // Floating animation
    this.floatOffset += dt * this.floatSpeed;
    this.group.position.y = this.position.y + Math.sin(this.floatOffset) * 0.1;

    // Rotation
    this.group.rotation.y += dt * 1.0;

    // Highlight on player approach
    const dist = MathUtils.distance(this.group.position, playerPos);
    if (dist < this.interactRadius && !this.collected) {
      this.showHighlight();
    } else {
      this.hideHighlight();
    }
  }

  showHighlight() {
    this.group.scale.set(1.1, 1.1, 1.1);
  }

  hideHighlight() {
    this.group.scale.set(1, 1, 1);
  }

  collect() {
    this.collected = true;
    
    // Animate out
    const duration = 0.5;
    const startScale = this.group.scale.clone();
    const startY = this.group.position.y;
    let elapsed = 0;

    const animate = () => {
      elapsed += 0.016;
      const progress = elapsed / duration;

      if (progress >= 1) {
        this.scene.remove(this.group);
        return;
      }

      this.group.scale.copy(startScale.clone().multiplyScalar(1 - progress * 0.5));
      this.group.position.y = startY + progress * 1.0;

      requestAnimationFrame(animate);
    };

    animate();
  }

  getDescription() {
    const descriptions = {
      'form': "Request for Voluntary Exit (Form 27-Γ). Must be filed in triplicate. This is copy 1 of 1.",
      'key': "Small brass key. Fits filing cabinet drawer #3.",
      'stamp': "Rubber stamp. Applies official exit approval to forms.",
      'coffee': "Director's premium coffee. Smells like regret and desperation.",
      'minutes': "Meeting minutes. Nobody remembers what was discussed.",
      'battery': "Flashlight battery replacement. Keep it dry.",
      'note': "Sticky note with cryptic message from Greg.",
      'calibration': "Portal calibration device. Manual lost in broken portal."
    };
    return descriptions[this.type] || "Unknown item.";
  }
}

export class ItemManager {
  constructor(scene, world) {
    this.scene = scene;
    this.world = world;
    this.items = [];
    this.spawnItems();
  }

  spawnItems() {
    // Form 27-Γ in Records room, hidden in cabinet
    const records = this.world.getRoomByID(1);
    if (records) {
      this.items.push(new Item(
        this.scene,
        new THREE.Vector3(3, 1.5, -4),
        'form',
        { location: 'cabinet-3' }
      ));
    }

    // Small key in Void office
    const voidOffice = this.world.getRoomByID(5);
    if (voidOffice) {
      this.items.push(new Item(
        this.scene,
        new THREE.Vector3(1, 1.0, 1),
        'key',
        { location: 'void-desk' }
      ));
    }

    // Rubber stamp in Void office
    if (voidOffice) {
      this.items.push(new Item(
        this.scene,
        new THREE.Vector3(-1, 1.0, 1),
        'stamp',
        { location: 'void-desk' }
      ));
    }

    // Director's coffee in Break room
    const breakRoom = this.world.getRoomByID(4);
    if (breakRoom) {
      this.items.push(new Item(
        this.scene,
        new THREE.Vector3(-2, 1.2, 2),
        'coffee',
        { location: 'coffee-machine' }
      ));
    }

    // Meeting minutes in Conference room
    const conference = this.world.getRoomByID(8);
    if (conference) {
      this.items.push(new Item(
        this.scene,
        new THREE.Vector3(0, 1.0, 0),
        'minutes',
        { location: 'conference-table' }
      ));
    }

    // Master key in Director's office
    const director = this.world.getRoomByID(6);
    if (director) {
      this.items.push(new Item(
        this.scene,
        new THREE.Vector3(1.5, 1.5, 0),
        'key',
        { location: 'director-drawer', ismaster: true }
      ));
    }

    // Portal Calibration Device in Server room
    const server = this.world.getRoomByID(9);
    if (server) {
      this.items.push(new Item(
        this.scene,
        new THREE.Vector3(-2, 2.0, 0),
        'calibration',
        { location: 'server-shelf' }
      ));
    }

    // Batteries scattered (3 total)
    if (breakRoom) {
      this.items.push(new Item(
        this.scene,
        new THREE.Vector3(2, 1.0, -2),
        'battery'
      ));
    }
    if (records) {
      this.items.push(new Item(
        this.scene,
        new THREE.Vector3(-3, 0.8, 3),
        'battery'
      ));
    }
    if (director) {
      this.items.push(new Item(
        this.scene,
        new THREE.Vector3(-2, 1.2, -2),
        'battery'
      ));
    }

    // Sticky notes (8 total, scattered for secret ending hints)
    const noteLocations = [
      [this.world.getRoomByID(0), new THREE.Vector3(4, 1.2, 0)],
      [this.world.getRoomByID(1), new THREE.Vector3(-2, 2.5, -4)],
      [this.world.getRoomByID(3), new THREE.Vector3(1.5, 1.5, 5)],
      [this.world.getRoomByID(5), new THREE.Vector3(0, 1.8, 4)],
      [this.world.getRoomByID(6), new THREE.Vector3(-4, 1.0, 0)],
      [this.world.getRoomByID(8), new THREE.Vector3(6, 1.0, 0)],
      [this.world.getRoomByID(9), new THREE.Vector3(0, 3.0, 0)],
      [this.world.getRoomByID(11), new THREE.Vector3(-7, 1.0, 0)]
    ];

    for (let [room, pos] of noteLocations) {
      if (room) {
        const worldPos = pos.clone().add(room.group.position);
        this.items.push(new Item(this.scene, worldPos, 'note'));
      }
    }
  }

  update(dt, playerPos) {
    this.items.forEach(item => {
      if (!item.collected) {
        item.update(dt, playerPos);
      }
    });
  }

  checkInteraction(playerPos, radius = 2.0) {
    for (let item of this.items) {
      if (!item.collected) {
        const dist = MathUtils.distance(playerPos, item.group.position);
        if (dist < radius) {
          return item;
        }
      }
    }
    return null;
  }

  collectItem(item) {
    if (item && !item.collected) {
      item.collect();
      return item.type;
    }
    return null;
  }
}