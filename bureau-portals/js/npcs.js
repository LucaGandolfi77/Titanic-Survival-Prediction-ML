import * as THREE from 'three';
import { MathUtils } from './utils.js';

export class NPC {
  constructor(scene, position, type) {
    this.scene = scene;
    this.position = position.clone();
    this.type = type; // 'bureaucrat', 'guard', 'auditor'
    this.group = new THREE.Group();
    this.group.position.copy(position);
    
    this.velocity = new THREE.Vector3();
    this.targetPos = position.clone();
    this.speed = 1.5;
    this.time = 0;
    
    this.isTalking = false;
    this.talkTimer = 0;
    this.pathIndex = 0;
    this.patrolPath = [];
    
    scene.add(this.group);
    
    this.buildMesh();
  }

  buildMesh() {
    const matSuit = new THREE.MeshLambertMaterial({ color: 0x4a4a4a });
    const matSkin = new THREE.MeshLambertMaterial({ color: 0xccb8a8 });
    const matHair = new THREE.MeshLambertMaterial({ color: 0x2a2020 });

    // Body
    this.body = new THREE.Mesh(
      new THREE.BoxGeometry(0.6, 1.0, 0.3),
      matSuit
    );
    this.body.position.y = 0.5;
    this.body.castShadow = true;
    this.group.add(this.body);

    // Head
    const head = new THREE.Mesh(
      new THREE.BoxGeometry(0.4, 0.4, 0.35),
      matSkin
    );
    head.position.y = 1.45;
    head.castShadow = true;
    this.group.add(head);

    // Hair
    const hair = new THREE.Mesh(
      new THREE.BoxGeometry(0.42, 0.15, 0.38),
      matHair
    );
    hair.position.set(0, 1.62, 0);
    this.group.add(hair);

    // Glasses (if bureaucrat)
    if (this.type !== 'auditor') {
      const glassL = new THREE.Mesh(
        new THREE.TorusGeometry(0.06, 0.01, 8, 16),
        new THREE.MeshBasicMaterial({ color: 0x4488ff })
      );
      glassL.position.set(-0.1, 1.5, 0.15);
      glassL.rotation.z = Math.PI / 4;
      this.group.add(glassL);

      const glassR = glassL.clone();
      glassR.position.x = 0.1;
      this.group.add(glassR);
    }

    // Arms
    for (let side of [-1, 1]) {
      const arm = new THREE.Mesh(
        new THREE.BoxGeometry(0.15, 0.5, 0.15),
        matSuit
      );
      arm.position.set(side * 0.35, 0.9, 0);
      arm.castShadow = true;
      this.group.add(arm);
    }

    // Legs
    for (let side of [-1, 1]) {
      const leg = new THREE.Mesh(
        new THREE.BoxGeometry(0.2, 0.5, 0.2),
        matSuit
      );
      leg.position.set(side * 0.15, 0.25, 0);
      leg.castShadow = true;
      this.group.add(leg);
    }

    // Briefcase
    if (this.type === 'bureaucrat' || this.type === 'guard') {
      const briefcase = new THREE.Mesh(
        new THREE.BoxGeometry(0.3, 0.25, 0.1),
        new THREE.MeshLambertMaterial({ color: 0x2a1a0a })
      );
      briefcase.position.set(0.4, 0.3, 0);
      briefcase.castShadow = true;
      this.group.add(briefcase);
    }
  }

  update(dt, player) {
    this.time += dt;

    if (this.type === 'auditor') {
      this.updateAuditor(dt, player);
    } else {
      this.updateBureaucrat(dt, player);
    }
  }

  updateBureaucrat(dt, player) {
    // Simple patrol AI
    const distToPlayer = MathUtils.distance(this.group.position, player.position);

    if (distToPlayer < 3) {
      // Stop and face player
      const dir = new THREE.Vector3().subVectors(player.position, this.group.position);
      this.group.lookAt(player.position);
      
      if (distToPlayer < 2 && !this.isTalking) {
        this.startTalking();
      }
    } else {
      // Patrol
      if (this.patrolPath.length > 0) {
        this.targetPos = this.patrolPath[this.pathIndex].clone();
        
        const distToTarget = MathUtils.distance(this.group.position, this.targetPos);
        if (distToTarget < 0.5) {
          this.pathIndex = (this.pathIndex + 1) % this.patrolPath.length;
        }

        const direction = new THREE.Vector3().subVectors(this.targetPos, this.group.position).normalize();
        this.group.position.add(direction.multiplyScalar(this.speed * dt));
      }
    }

    // Update talk timer
    if (this.isTalking) {
      this.talkTimer -= dt;
      if (this.talkTimer <= 0) {
        this.isTalking = false;
      }
    }
  }

  updateAuditor(dt, player) {
    // Auditor moves toward player relentlessly
    const direction = new THREE.Vector3()
      .subVectors(player.position, this.group.position)
      .normalize();

    // Move through walls
    this.group.position.add(direction.multiplyScalar(2.0 * dt)); // Faster than player

    // Look at player
    this.group.lookAt(player.position);

    // Check collision with player
    const dist = MathUtils.distance(this.group.position, player.position);
    if (dist < 0.8) {
      player.modifySanity(-20);
      player.position.add(direction.multiplyScalar(-2)); // Push player back
    }
  }

  startTalking() {
    this.isTalking = true;
    this.talkTimer = 3;

    const quotes = [
      "Have you submitted form 27-B/6?",
      "That portal is scheduled for maintenance in Q4 2047.",
      "Per regulation 44-Ω, you cannot be here.",
      "I've been in this room for 11 years.",
      "Your badge says VISITOR but that portal ate our visitor log."
    ];

    const quote = quotes[Math.floor(Math.random() * quotes.length)];
    
    if (window.game && window.game.ui) {
      window.game.ui.showNPCDialog(this.group.position, quote);
    }

    if (window.game && window.game.audio) {
      window.game.audio.playDialog();
    }
  }

  setPatrolPath(points) {
    this.patrolPath = points.map(p => p.clone());
    this.pathIndex = 0;
  }
}

export class PortraitEyes {
  constructor(scene, position) {
    this.scene = scene;
    this.position = position;
    this.group = new THREE.Group();
    this.group.position.copy(position);
    
    scene.add(this.group);
    
    this.buildPortrait();
  }

  buildPortrait() {
    // Portrait frame (already in world)
    // Just add eyes
    const eyeBg = new THREE.Mesh(
      new THREE.CircleGeometry(0.15),
      new THREE.MeshBasicMaterial({ color: 0xcccccc })
    );
    eyeBg.position.set(-0.3, 0.3, 0.1);
    this.group.add(eyeBg);

    const eyeL = new THREE.Mesh(
      new THREE.CircleGeometry(0.08),
      new THREE.MeshBasicMaterial({ color: 0x000000 })
    );
    eyeL.position.copy(eyeBg.position);
    eyeL.position.z += 0.05;
    this.eyeL = eyeL;
    this.group.add(eyeL);

    const eyeBgR = eyeBg.clone();
    eyeBgR.position.x = 0.3;
    this.group.add(eyeBgR);

    const eyeR = eyeL.clone();
    eyeR.position.x = 0.3;
    this.eyeR = eyeR;
    this.group.add(eyeR);
  }

  updateEyesFollowCamera(cameraPos) {
    // Make eyes follow camera
    const portraitWorldPos = new THREE.Vector3();
    this.group.getWorldPosition(portraitWorldPos);

    const dir = new THREE.Vector3().subVectors(cameraPos, portraitWorldPos).normalize();

    // Move pupils toward camera direction
    const maxOffset = 0.04;
    const eyeOffset = dir.clone().multiplyScalar(maxOffset);

    this.eyeL.position.add(eyeOffset);
    this.eyeR.position.add(eyeOffset);
  }
}

export class NPCManager {
  constructor(scene, world) {
    this.scene = scene;
    this.world = world;
    this.npcs = [];
    this.auditor = null;
    this.auditorAppeared = false;

    this.buildNPCs();
  }

  buildNPCs() {
    // Lobby guards
    const lobby = this.world.getRoomByID(0);
    if (lobby) {
      const guard1 = new NPC(this.scene, new THREE.Vector3(-3, 0, -2), 'guard');
      guard1.setPatrolPath([
        new THREE.Vector3(-3, 0, -2),
        new THREE.Vector3(-3, 0, 2)
      ]);
      this.npcs.push(guard1);

      const guard2 = new NPC(this.scene, new THREE.Vector3(3, 0, -2), 'guard');
      guard2.setPatrolPath([
        new THREE.Vector3(3, 0, -2),
        new THREE.Vector3(3, 0, 2)
      ]);
      this.npcs.push(guard2);
    }

    // Conference room bureaucrats
    const conference = this.world.getRoomByID(8);
    if (conference) {
      for (let i = 0; i < 8; i++) {
        const angle = (i / 8) * Math.PI * 2;
        const x = Math.cos(angle) * 4;
        const z = Math.sin(angle) * 2;
        const bureaucrat = new NPC(this.scene, new THREE.Vector3(x, 0, z), 'bureaucrat');
        this.npcs.push(bureaucrat);
      }
    }

    // Records room bureaucrat
    const records = this.world.getRoomByID(1);
    if (records) {
      const clerk = new NPC(this.scene, new THREE.Vector3(0, 0, 0), 'bureaucrat');
      clerk.setPatrolPath([
        new THREE.Vector3(-4, 0, 0),
        new THREE.Vector3(4, 0, 0)
      ]);
      this.npcs.push(clerk);
    }

    // Director's office bureaucrat
    const director = this.world.getRoomByID(6);
    if (director) {
      const directorSecretary = new NPC(this.scene, new THREE.Vector3(2, 0, -3), 'bureaucrat');
      directorSecretary.setPatrolPath([
        new THREE.Vector3(2, 0, -3),
        new THREE.Vector3(2, 0, -1)
      ]);
      this.npcs.push(directorSecretary);
    }
  }

  checkIfAuditorShouldAppear(player) {
    if (!this.auditorAppeared && player.sanity < 50) {
      this.auditorAppeared = true;
      this.spawnAuditor(player);
    }
  }

  spawnAuditor(player) {
    const exitHall = this.world.getRoomByID(11);
    const spawnPos = new THREE.Vector3(0, 0, 0);
    if (exitHall) {
      spawnPos.copy(exitHall.group.position);
    }

    this.auditor = new NPC(this.scene, spawnPos, 'auditor');
    this.npcs.push(this.auditor);
  }

  update(dt, player) {
    this.npcs.forEach(npc => npc.update(dt, player));
    this.checkIfAuditorShouldAppear(player);
  }
}