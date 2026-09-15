import * as THREE from 'three';
import { MathUtils } from './utils.js';
import { bus } from './event-bus.js';

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
    this.isAttacking = type === 'auditor';

    this.isTalking = false;
    this.talkTimer = 0;
    this.pathIndex = 0;
    this.patrolPath = [];
    this.interactionCount = 0;

    scene.add(this.group);

    this.buildMesh();
  }

  buildMesh() {
    const matSuit =
      NPC._matSuit ||
      (NPC._matSuit = new THREE.MeshStandardMaterial({
        color: 0x4a4a4a,
        roughness: 0.8,
        metalness: 0.05
      }));
    const matSkin =
      NPC._matSkin ||
      (NPC._matSkin = new THREE.MeshStandardMaterial({
        color: 0xccb8a8,
        roughness: 0.5,
        metalness: 0.0
      }));
    const matHair =
      NPC._matHair ||
      (NPC._matHair = new THREE.MeshStandardMaterial({
        color: 0x2a2020,
        roughness: 0.9,
        metalness: 0.0
      }));
    const matGlass =
      NPC._matGlass ||
      (NPC._matGlass = new THREE.MeshStandardMaterial({
        color: 0x4488ff,
        emissive: 0x4488ff,
        emissiveIntensity: 0.5,
        roughness: 0.1,
        metalness: 0.9,
        transparent: true
      }));
    const matBriefcase =
      NPC._matBriefcase ||
      (NPC._matBriefcase = new THREE.MeshStandardMaterial({
        color: 0x2a1a0a,
        roughness: 0.7,
        metalness: 0.05
      }));

    const geoBody = NPC._geoBody || (NPC._geoBody = new THREE.BoxGeometry(0.6, 1.0, 0.3));
    const geoHead = NPC._geoHead || (NPC._geoHead = new THREE.BoxGeometry(0.4, 0.4, 0.35));
    const geoHair = NPC._geoHair || (NPC._geoHair = new THREE.BoxGeometry(0.42, 0.15, 0.38));
    const geoGlass = NPC._geoGlass || (NPC._geoGlass = new THREE.TorusGeometry(0.06, 0.01, 8, 16));
    const geoArm = NPC._geoArm || (NPC._geoArm = new THREE.BoxGeometry(0.15, 0.5, 0.15));
    const geoLeg = NPC._geoLeg || (NPC._geoLeg = new THREE.BoxGeometry(0.2, 0.5, 0.2));
    const geoBriefcase =
      NPC._geoBriefcase || (NPC._geoBriefcase = new THREE.BoxGeometry(0.3, 0.25, 0.1));

    // Body
    this.body = new THREE.Mesh(geoBody, matSuit);
    this.body.position.y = 0.5;
    this.body.castShadow = true;
    this.group.add(this.body);

    // Head
    const head = new THREE.Mesh(geoHead, matSkin);
    head.position.y = 1.45;
    head.castShadow = true;
    this.group.add(head);

    // Hair
    const hair = new THREE.Mesh(geoHair, matHair);
    hair.position.set(0, 1.62, 0);
    this.group.add(hair);

    // Glasses (if bureaucrat)
    if (this.type !== 'auditor') {
      const glassL = new THREE.Mesh(geoGlass, matGlass);
      glassL.position.set(-0.1, 1.5, 0.15);
      glassL.rotation.z = Math.PI / 4;
      this.group.add(glassL);

      const glassR = glassL.clone();
      glassR.position.x = 0.1;
      this.group.add(glassR);
    }

    // Arms
    for (const side of [-1, 1]) {
      const arm = new THREE.Mesh(geoArm, matSuit);
      arm.position.set(side * 0.35, 0.9, 0);
      arm.castShadow = true;
      this.group.add(arm);
    }

    // Legs
    for (const side of [-1, 1]) {
      const leg = new THREE.Mesh(geoLeg, matSuit);
      leg.position.set(side * 0.15, 0.25, 0);
      leg.castShadow = true;
      this.group.add(leg);
    }

    // Briefcase
    if (this.type === 'bureaucrat' || this.type === 'guard') {
      const briefcase = new THREE.Mesh(geoBriefcase, matBriefcase);
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
        this.startTalking(player);
      }
    } else {
      // Patrol
      if (this.patrolPath.length > 0) {
        this.targetPos = this.patrolPath[this.pathIndex].clone();

        const distToTarget = MathUtils.distance(this.group.position, this.targetPos);
        if (distToTarget < 0.5) {
          this.pathIndex = (this.pathIndex + 1) % this.patrolPath.length;
        }

        const direction = new THREE.Vector3()
          .subVectors(this.targetPos, this.group.position)
          .normalize();
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
      this._damageTimer = (this._damageTimer || 0) + dt;
      if (this._damageTimer > 1.0) {
        player.modifySanity(-20);
        this._damageTimer = 0;
      }
      player.position.add(direction.multiplyScalar(-2)); // Push player back
    }
  }

  startTalking(player) {
    this.isTalking = true;
    this.talkTimer = 3;
    this.interactionCount++;

    let quotes, penalty;
    if (this.interactionCount === 1) {
      quotes = [
        'Have you submitted form 27-B/6?',
        'That portal is scheduled for maintenance in Q4 2047.',
        'Per regulation 44-Ω, you cannot be here.',
        "I've been in this room for 11 years.",
        'Your badge says VISITOR but that portal ate our visitor log.'
      ];
      penalty = 3;
    } else if (this.interactionCount === 2) {
      quotes = [
        "Oh, it's you again. Still no form?",
        "I remember your face. Desk 7, wasn't it?",
        'The auditor watches those who wander.',
        'Your paperwork will be filed. Eventually.',
        'Have you considered a transfer? To the void, perhaps?'
      ];
      penalty = 5;
    } else {
      quotes = [
        'Interaction logged. I remember you now. Complete your forms.',
        'The auditor knows your face. Avoid the EXIT HALL.',
        'Try the Rubber Stamp in Void Office. It helps with the process.',
        'We have been watching. Every. Step. Of. The. Way.',
        'Your file grows heavier each visit. So does my suspicion.'
      ];
      penalty = 7;
    }

    const quote = quotes[Math.floor(Math.random() * quotes.length)];

    bus.emit('npc:dialog', { position: this.group.position, text: quote });
    bus.emit('npc:interacted', { npc: this });
    player.modifySanity(-penalty);

    bus.emit('audio:dialog');
  }

  setPatrolPath(points) {
    this.patrolPath = points.map((p) => p.clone());
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

    // Move pupils toward camera direction (absolute positioning to prevent drift)
    const maxOffset = 0.04;
    const eyeOffset = dir.clone().multiplyScalar(maxOffset);

    this.eyeLBase = this.eyeLBase || this.eyeL.position.clone();
    this.eyeRBase = this.eyeRBase || this.eyeR.position.clone();

    this.eyeL.position.copy(this.eyeLBase).add(eyeOffset);
    this.eyeR.position.copy(this.eyeRBase).add(eyeOffset);
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
      guard1.setPatrolPath([new THREE.Vector3(-3, 0, -2), new THREE.Vector3(-3, 0, 2)]);
      this.npcs.push(guard1);

      const guard2 = new NPC(this.scene, new THREE.Vector3(3, 0, -2), 'guard');
      guard2.setPatrolPath([new THREE.Vector3(3, 0, -2), new THREE.Vector3(3, 0, 2)]);
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
      clerk.setPatrolPath([new THREE.Vector3(-4, 0, 0), new THREE.Vector3(4, 0, 0)]);
      this.npcs.push(clerk);
    }

    // Director's office bureaucrat
    const director = this.world.getRoomByID(6);
    if (director) {
      const directorSecretary = new NPC(this.scene, new THREE.Vector3(2, 0, -3), 'bureaucrat');
      directorSecretary.setPatrolPath([new THREE.Vector3(2, 0, -3), new THREE.Vector3(2, 0, -1)]);
      this.npcs.push(directorSecretary);
    }
  }

  checkIfAuditorShouldAppear(player) {
    if (!this.auditorAppeared && player.sanity < -50) {
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
    this.npcs.forEach((npc) => npc.update(dt, player));
    this.checkIfAuditorShouldAppear(player);
  }
}
