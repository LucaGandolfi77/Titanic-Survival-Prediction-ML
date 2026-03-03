import * as THREE from 'three';
import { MathUtils } from './utils.js';

export class WorldGenerator {
  constructor(scene) {
    this.scene = scene;
    this.rooms = [];
    this.currentRoomID = 0;
    
    // Materials
    this.matConcrete = new THREE.MeshLambertMaterial({ color: 0x6a6a6a });
    this.matWall = new THREE.MeshLambertMaterial({ color: 0x8a8a90 });
    this.matDoor = new THREE.MeshLambertMaterial({ color: 0x3a3a3a });
    this.matDesk = new THREE.MeshLambertMaterial({ color: 0x8b4513 });
    this.matChair = new THREE.MeshLambertMaterial({ color: 0x4a4a4a });
    this.matMetal = new THREE.MeshLambertMaterial({ color: 0xc0c0c0 });
    this.matFiling = new THREE.MeshLambertMaterial({ color: 0x654321 });
    this.matCeiling = new THREE.MeshLambertMaterial({ color: 0xe8e4de });
    this.matOffice = new THREE.MeshLambertMaterial({ color: 0xd4cfc8 });
    
    this.buildAllRooms();
  }

  buildAllRooms() {
    // Room 0: Lobby
    this.buildLobby();
    // Room 1: Records
    this.buildRecords();
    // Room 2: Inverted Office
    this.buildInvertedOffice();
    // Room 3: Infinite Corridor (loop)
    this.buildInfiniteCorridor();
    // Room 4: Sideways Break Room
    this.buildBreakRoom();
    // Room 5: The Void Office
    this.buildVoidOffice();
    // Room 6: Director's Office
    this.buildDirectorOffice();
    // Room 7: Copy Room (mirror)
    this.buildCopyRoom();
    // Room 8: Conference Room
    this.buildConferenceRoom();
    // Room 9: Server Room
    this.buildServerRoom();
    // Room 10: Maintenance Shaft
    this.buildMaintenanceShaft();
    // Room 11: Exit Hall
    this.buildExitHall();
  }

  createRoom(id, width, height, depth) {
    const room = {
      id: id,
      group: new THREE.Group(),
      width: width,
      height: height,
      depth: depth,
      portals: [],
      items: [],
      npcs: [],
      bounds: new THREE.Box3(
        new THREE.Vector3(-width / 2, 0, -depth / 2),
        new THREE.Vector3(width / 2, height, depth / 2)
      )
    };
    
    this.scene.add(room.group);
    return room;
  }

  buildFloor(room, color = 0x6a6a6a) {
    const floor = new THREE.Mesh(
      new THREE.BoxGeometry(room.width, 0.1, room.depth),
      new THREE.MeshLambertMaterial({ color: color })
    );
    floor.position.y = 0;
    floor.receiveShadow = true;
    floor.castShadow = false;
    room.group.add(floor);
  }

  buildCeiling(room, color = 0xe8e4de) {
    const ceiling = new THREE.Mesh(
      new THREE.BoxGeometry(room.width, 0.1, room.depth),
      new THREE.MeshLambertMaterial({ color: color })
    );
    ceiling.position.y = room.height;
    ceiling.receiveShadow = true;
    room.group.add(ceiling);
  }

  buildWalls(room, color = 0x8a8a90) {
    const walls = [
      // Left
      new THREE.Mesh(new THREE.BoxGeometry(0.2, room.height, room.depth), 
        new THREE.MeshLambertMaterial({ color: color })),
      // Right
      new THREE.Mesh(new THREE.BoxGeometry(0.2, room.height, room.depth), 
        new THREE.MeshLambertMaterial({ color: color })),
      // Front
      new THREE.Mesh(new THREE.BoxGeometry(room.width, room.height, 0.2), 
        new THREE.MeshLambertMaterial({ color: color })),
      // Back
      new THREE.Mesh(new THREE.BoxGeometry(room.width, room.height, 0.2), 
        new THREE.MeshLambertMaterial({ color: color }))
    ];
    
    walls[0].position.x = -room.width / 2;
    walls[0].position.y = room.height / 2;
    walls[1].position.x = room.width / 2;
    walls[1].position.y = room.height / 2;
    walls[2].position.z = -room.depth / 2;
    walls[2].position.y = room.height / 2;
    walls[3].position.z = room.depth / 2;
    walls[3].position.y = room.height / 2;
    
    walls.forEach(w => {
      w.receiveShadow = true;
      w.castShadow = true;
      room.group.add(w);
    });
    
    return walls;
  }

  buildDesk(room, x, y, z) {
    const group = new THREE.Group();
    
    const top = new THREE.Mesh(
      new THREE.BoxGeometry(2, 0.08, 1),
      this.matDesk
    );
    top.position.set(0, 0.8, 0);
    top.castShadow = true;
    group.add(top);
    
    // Legs
    for (let i = -1; i <= 1; i += 2) {
      for (let j = -1; j <= 1; j += 2) {
        const leg = new THREE.Mesh(
          new THREE.CylinderGeometry(0.05, 0.05, 0.8),
          this.matMetal
        );
        leg.position.set(i * 0.8, 0.4, j * 0.4);
        leg.castShadow = true;
        group.add(leg);
      }
    }
    
    group.position.set(x, y, z);
    room.group.add(group);
    return group;
  }

  buildChair(room, x, y, z) {
    const group = new THREE.Group();
    
    const seat = new THREE.Mesh(
      new THREE.BoxGeometry(0.5, 0.1, 0.5),
      this.matChair
    );
    seat.position.y = 0.45;
    group.add(seat);
    
    const back = new THREE.Mesh(
      new THREE.BoxGeometry(0.5, 0.6, 0.1),
      this.matChair
    );
    back.position.set(0, 0.8, -0.25);
    back.castShadow = true;
    group.add(back);
    
    // Legs
    for (let i = -1; i <= 1; i += 2) {
      for (let j = -1; j <= 1; j += 2) {
        const leg = new THREE.Mesh(
          new THREE.CylinderGeometry(0.04, 0.04, 0.45),
          this.matMetal
        );
        leg.position.set(i * 0.2, 0.225, j * 0.2);
        leg.castShadow = true;
        group.add(leg);
      }
    }
    
    group.position.set(x, y, z);
    room.group.add(group);
    return group;
  }

  buildFilingCabinet(room, x, y, z) {
    const { group, parts } = this.createFilingCabinet();
    group.position.set(x, y, z);
    room.group.add(group);
    return { group, parts };
  }

  createFilingCabinet() {
    const group = new THREE.Group();
    const parts = [];
    
    const body = new THREE.Mesh(
      new THREE.BoxGeometry(0.6, 1.2, 0.5),
      this.matFiling
    );
    body.castShadow = true;
    group.add(body);
    parts.push(body);
    
    // Drawers (4)
    for (let i = 0; i < 4; i++) {
      const drawer = new THREE.Mesh(
        new THREE.BoxGeometry(0.55, 0.25, 0.4),
        new THREE.MeshLambertMaterial({ color: 0x5a3a1a })
      );
      drawer.position.y = -0.35 + i * 0.3;
      drawer.userData.isDrawer = true;
      drawer.userData.drawerIndex = i;
      group.add(drawer);
    }
    
    return { group, parts };
  }

  buildCoffeeMachine(room, x, y, z) {
    const group = new THREE.Group();
    
    const body = new THREE.Mesh(
      new THREE.BoxGeometry(0.5, 0.8, 0.4),
      this.matMetal
    );
    body.castShadow = true;
    group.add(body);
    
    const spout = new THREE.Mesh(
      new THREE.CylinderGeometry(0.08, 0.08, 0.15),
      this.matMetal
    );
    spout.position.set(0.1, -0.3, 0.15);
    spout.castShadow = true;
    group.add(spout);
    
    group.position.set(x, y, z);
    room.group.add(group);
    return group;
  }

  buildVendingMachine(room, x, y, z) {
    const group = new THREE.Group();
    
    const body = new THREE.Mesh(
      new THREE.BoxGeometry(0.4, 1.0, 0.3),
      new THREE.MeshLambertMaterial({ color: 0xdd0000 })
    );
    body.castShadow = true;
    group.add(body);
    
    // Shelves
    for (let i = 0; i < 4; i++) {
      const shelf = new THREE.Mesh(
        new THREE.BoxGeometry(0.35, 0.05, 0.25),
        new THREE.MeshLambertMaterial({ color: 0x333333 })
      );
      shelf.position.y = -0.35 + i * 0.25;
      group.add(shelf);
    }
    
    group.position.set(x, y, z);
    room.group.add(group);
    return group;
  }

  buildServerRack(room, x, y, z) {
    const group = new THREE.Group();
    
    for (let i = 0; i < 4; i++) {
      const rack = new THREE.Mesh(
        new THREE.BoxGeometry(0.3, 0.6, 0.6),
        new THREE.MeshLambertMaterial({ color: 0x1a1a1a })
      );
      rack.position.y = 0.3 + i * 0.65;
      
      // LED lights (emissive)
      const led = new THREE.Mesh(
        new THREE.BoxGeometry(0.02, 0.02, 0.02),
        new THREE.MeshLambertMaterial({ color: 0x00ff00, emissive: 0x00ff00 })
      );
      led.position.set(0.1, 0.1, 0.25);
      rack.add(led);
      
      group.add(rack);
    }
    
    group.position.set(x, y, z);
    room.group.add(group);
    return group;
  }

  // ROOM IMPLEMENTATIONS
  buildLobby() {
    const room = this.createRoom(0, 12, 4, 10);
    this.buildFloor(room, 0x6a6a6a);
    this.buildCeiling(room);
    this.buildWalls(room);
    
    // Reception desk
    this.buildDesk(room, 0, 0, -3);
    this.buildChair(room, 2, 0, -3);
    
    this.rooms.push(room);
  }

  buildRecords() {
    const room = this.createRoom(1, 10, 3.5, 14);
    this.buildFloor(room);
    this.buildCeiling(room);
    this.buildWalls(room);
    
    // Filing cabinets grid
    for (let x = -3; x <= 3; x += 2) {
      for (let z = -4; z <= 4; z += 2) {
        this.buildFilingCabinet(room, x, 0, z);
      }
    }
    
    this.rooms.push(room);
  }

  buildInvertedOffice() {
    const room = this.createRoom(2, 10, 4, 8);
    this.buildFloor(room, 0x7a7a7a);
    this.buildCeiling(room, 0xd8d4ce);
    this.buildWalls(room, 0x9a9a9a);
    
    // Gravity will be inverted for this room in game logic
    // For now, build it normally but it will be rotated during gameplay
    this.buildDesk(room, -2, 0, 0);
    this.buildChair(room, 0, 0, 0);
    
    this.rooms.push(room);
  }

  buildInfiniteCorridor() {
    const room = this.createRoom(3, 4, 3, 20);
    this.buildFloor(room);
    this.buildCeiling(room);
    this.buildWalls(room);
    
    // Corridor lights create endless feeling
    for (let z = -8; z <= 8; z += 2) {
      const light = new THREE.Mesh(
        new THREE.BoxGeometry(3.5, 0.1, 0.1),
        new THREE.MeshLambertMaterial({ color: 0xffff00, emissive: 0xffff00 })
      );
      light.position.set(0, 2.8, z);
      room.group.add(light);
    }
    
    // Lever to break the loop
    const lever = new THREE.Mesh(
      new THREE.CylinderGeometry(0.1, 0.1, 0.5),
      this.matMetal
    );
    lever.position.set(1.5, 1.5, 0);
    lever.userData.isLever = true;
    lever.castShadow = true;
    room.group.add(lever);
    
    this.rooms.push(room);
  }

  buildBreakRoom() {
    const room = this.createRoom(4, 8, 3.5, 8);
    this.buildFloor(room);
    this.buildCeiling(room);
    this.buildWalls(room);
    
    // Coffee machine
    this.buildCoffeeMachine(room, -2, 0, 2);
    
    // Vending machine
    this.buildVendingMachine(room, 2, 0, 2);
    
    // Small table
    this.buildDesk(room, 0, 0, -2);
    this.buildChair(room, -1, 0, -1.5);
    this.buildChair(room, 1, 0, -1.5);
    
    this.rooms.push(room);
  }

  buildVoidOffice() {
    const room = this.createRoom(5, 10, 3.5, 10);
    
    // Minimal furniture, mostly darkness
    const floor = new THREE.Mesh(
      new THREE.BoxGeometry(10, 0.1, 10),
      new THREE.MeshLambertMaterial({ color: 0x1a1a1a })
    );
    floor.position.y = 0;
    floor.receiveShadow = true;
    room.group.add(floor);
    
    const ceiling = new THREE.Mesh(
      new THREE.BoxGeometry(10, 0.1, 10),
      new THREE.MeshLambertMaterial({ color: 0x2a2a2a })
    );
    ceiling.position.y = 3.5;
    room.group.add(ceiling);
    
    this.buildWalls(room, 0x3a3a3a);
    
    // Single desk with light
    this.buildDesk(room, 0, 0, 0);
    
    // Desk lamp
    const lampBase = new THREE.Mesh(
      new THREE.CylinderGeometry(0.1, 0.1, 0.3),
      this.matMetal
    );
    lampBase.position.set(1, 0.9, 0);
    
    const lampShade = new THREE.Mesh(
      new THREE.CylinderGeometry(0.2, 0.2, 0.05),
      new THREE.MeshLambertMaterial({ color: 0xffff00, emissive: 0x666600 })
    );
    lampShade.position.set(1, 1.3, 0);
    
    room.group.add(lampBase, lampShade);
    
    this.rooms.push(room);
  }

  buildDirectorOffice() {
    const room = this.createRoom(6, 12, 4, 10);
    this.buildFloor(room, 0x5a3a2a);
    this.buildCeiling(room);
    this.buildWalls(room, 0x6a5a4a);
    
    // Large imposing desk
    const bigDesk = new THREE.Mesh(
      new THREE.BoxGeometry(4, 0.08, 1.5),
      new THREE.MeshLambertMaterial({ color: 0x8b0000 })
    );
    bigDesk.position.set(0, 0.8, -2);
    bigDesk.castShadow = true;
    room.group.add(bigDesk);
    
    // Director's chair
    this.buildChair(room, 0, 0, -3.5);
    
    // Portrait on wall
    const portraitFrame = new THREE.Mesh(
      new THREE.BoxGeometry(2, 2.5, 0.1),
      new THREE.MeshLambertMaterial({ color: 0x4a4a4a })
    );
    portraitFrame.position.set(0, 2, 4.9);
    portraitFrame.userData.isPortrait = true;
    room.group.add(portraitFrame);
    
    this.rooms.push(room);
  }

  buildCopyRoom() {
    const room = this.createRoom(7, 8, 3.5, 8);
    this.buildFloor(room);
    this.buildCeiling(room);
    this.buildWalls(room);
    
    // Copy machines
    const copier = new THREE.Mesh(
      new THREE.BoxGeometry(0.8, 1.2, 0.6),
      this.matMetal
    );
    copier.position.set(-2, 0, 0);
    copier.castShadow = true;
    room.group.add(copier);
    
    // Paper dispenser
    const paperDisp = new THREE.Mesh(
      new THREE.BoxGeometry(0.5, 1.5, 0.4),
      new THREE.MeshLambertMaterial({ color: 0xffffff })
    );
    paperDisp.position.set(2, 0, 0);
    room.group.add(paperDisp);
    
    this.rooms.push(room);
  }

  buildConferenceRoom() {
    const room = this.createRoom(8, 14, 3.5, 8);
    this.buildFloor(room);
    this.buildCeiling(room);
    this.buildWalls(room);
    
    // Long conference table
    const table = new THREE.Mesh(
      new THREE.BoxGeometry(12, 0.08, 1.5),
      this.matDesk
    );
    table.position.set(0, 0.8, 0);
    table.castShadow = true;
    room.group.add(table);
    
    // Chairs around table (8 chairs for NPCs)
    for (let i = 0; i < 4; i++) {
      const x = -4 + i * 3;
      this.buildChair(room, x, 0, 1.5);
      this.buildChair(room, x, 0, -1.5);
    }
    
    this.rooms.push(room);
  }

  buildServerRoom() {
    const room = this.createRoom(9, 10, 4, 12);
    this.buildFloor(room);
    this.buildCeiling(room);
    this.buildWalls(room);
    
    // Rows of server racks
    for (let x = -3; x <= 3; x += 3) {
      for (let z = -4; z <= 4; z += 2) {
        this.buildServerRack(room, x, 0, z);
      }
    }
    
    this.rooms.push(room);
  }

  buildMaintenanceShaft() {
    const room = this.createRoom(10, 3, 1.5, 8);
    this.buildFloor(room, 0x4a4a4a);
    this.buildCeiling(room, 0x3a3a3a);
    this.buildWalls(room, 0x5a5a5a);
    
    // Low crawlspace - player must crouch
    room.isCrawlspace = true;
    
    this.rooms.push(room);
  }

  buildExitHall() {
    const room = this.createRoom(11, 16, 6, 12);
    this.buildFloor(room, 0x7a7a7a);
    this.buildCeiling(room);
    this.buildWalls(room, 0x9a9a9a);
    
    // Escalator (animated steps)
    const escalatorSteps = 12;
    for (let i = 0; i < escalatorSteps; i++) {
      const step = new THREE.Mesh(
        new THREE.BoxGeometry(6, 0.2, 0.8),
        new THREE.MeshLambertMaterial({ color: 0xc0c0c0 })
      );
      step.position.set(0, 1 + i * 0.4, -4 + i * 0.7);
      step.castShadow = true;
      room.group.add(step);
    }
    
    // Exit desk
    const exitDesk = new THREE.Mesh(
      new THREE.BoxGeometry(4, 0.08, 1),
      new THREE.MeshLambertMaterial({ color: 0x1a3a6e })
    );
    exitDesk.position.set(0, 0.8, 5);
    exitDesk.castShadow = true;
    exitDesk.userData.isExitDesk = true;
    room.group.add(exitDesk);
    
    // Sign
    const sign = new THREE.Mesh(
      new THREE.BoxGeometry(3, 0.5, 0.1),
      new THREE.MeshLambertMaterial({ color: 0xffff00 })
    );
    sign.position.set(0, 3, 5.5);
    room.group.add(sign);
    
    this.rooms.push(room);
  }

  getRoomByID(id) {
    return this.rooms.find(r => r.id === id);
  }
}