// js/character3d.js — 3D character builder (body + outfit layers)
import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.169.0/build/three.module.js';

export function buildCharacter(charDef){
  const group = new THREE.Group();
  group.userData.animation = 'idle';
  group.userData.baseY = 0;
  group.userData.homeZ = 0;
  group.userData.phase = Math.random() * Math.PI * 2;
  group.userData.outfitMeshes = {};
  group.userData.charId = charDef.id;

  const skin = new THREE.Color(charDef.skin);
  const hair = new THREE.Color(charDef.hair);
  const eyes = new THREE.Color(charDef.eyes);

  const skinMat  = ()=>new THREE.MeshStandardMaterial({color:skin, roughness:.7});
  const hairMat  = ()=>new THREE.MeshStandardMaterial({color:hair, roughness:.9});
  const eyeWhite = ()=>new THREE.MeshStandardMaterial({color:0xffffff});
  const eyeIris  = ()=>new THREE.MeshStandardMaterial({color:eyes});

  // ── Legs ───────────────────────────────────────────────────────
  const legGeo = new THREE.CylinderGeometry(0.08, 0.07, 0.5);
  const lLeg = new THREE.Mesh(legGeo, skinMat());
  lLeg.position.set(-0.12, 0.25, 0);
  lLeg.castShadow = true;
  group.add(lLeg);
  const rLeg = new THREE.Mesh(legGeo, skinMat());
  rLeg.position.set( 0.12, 0.25, 0);
  rLeg.castShadow = true;
  group.add(rLeg);
  group.userData.leftLeg  = lLeg;
  group.userData.rightLeg = rLeg;

  // ── Torso ─────────────────────────────────────────────────────
  const torsoGeo = new THREE.BoxGeometry(0.35, 0.5, 0.2);
  const torso = new THREE.Mesh(torsoGeo, skinMat());
  torso.position.set(0, 0.75, 0);
  torso.castShadow = true;
  group.add(torso);

  // ── Arms ──────────────────────────────────────────────────────
  const armGeo = new THREE.CylinderGeometry(0.06, 0.06, 0.4);
  const lArm = new THREE.Mesh(armGeo, skinMat());
  lArm.position.set(-0.25, 0.75, 0); lArm.rotation.z = 0.2;
  lArm.castShadow = true;
  const rArm = new THREE.Mesh(armGeo, skinMat());
  rArm.position.set( 0.25, 0.75, 0); rArm.rotation.z = -0.2;
  rArm.castShadow = true;
  group.add(lArm); group.add(rArm);
  group.userData.leftArm  = lArm;
  group.userData.rightArm = rArm;

  // ── Head ──────────────────────────────────────────────────────
  const headGeo = new THREE.SphereGeometry(0.22, 16, 12);
  const head = new THREE.Mesh(headGeo, skinMat());
  head.position.set(0, 1.25, 0);
  head.castShadow = true;
  group.add(head);

  // Eyes
  const eyeScl = new THREE.SphereGeometry(0.045, 8, 8);
  const irisSc = new THREE.SphereGeometry(0.025, 8, 8);
  const lEyeW = new THREE.Mesh(eyeScl, eyeWhite()); lEyeW.position.set(-0.08, 1.28, 0.19);
  const lEyeI = new THREE.Mesh(irisSc, eyeIris());  lEyeI.position.set(-0.08, 1.28, 0.21);
  const rEyeW = new THREE.Mesh(eyeScl, eyeWhite()); rEyeW.position.set( 0.08, 1.28, 0.19);
  const rEyeI = new THREE.Mesh(irisSc, eyeIris());  rEyeI.position.set( 0.08, 1.28, 0.21);
  group.add(lEyeW); group.add(lEyeI); group.add(rEyeW); group.add(rEyeI);

  // ── Hair ──────────────────────────────────────────────────────
  const hairGeo = new THREE.SphereGeometry(0.24, 12, 8);
  hairGeo.scale(1, 0.7, 1);
  const hairMesh = new THREE.Mesh(hairGeo, hairMat());
  hairMesh.position.set(0, 1.36, 0);
  group.add(hairMesh);

  // ── Outfit layers ───────────────────────────────────────────────
  // Top
  const topGeo = new THREE.BoxGeometry(0.38, 0.26, 0.22);
  const topMesh = new THREE.Mesh(topGeo, new THREE.MeshStandardMaterial({color:0xff6b9d}));
  topMesh.position.set(0, 0.82, 0);
  group.add(topMesh);
  group.userData.outfitMeshes.top = topMesh;

  // Bottom
  const botGeo = new THREE.BoxGeometry(0.36, 0.26, 0.22);
  const botMesh = new THREE.Mesh(botGeo, new THREE.MeshStandardMaterial({color:0x9c27b0}));
  botMesh.position.set(0, 0.50, 0);
  group.add(botMesh);
  group.userData.outfitMeshes.bottom = botMesh;

  // Shoes (2)
  const shoeGeo = new THREE.BoxGeometry(0.12, 0.06, 0.18);
  const lShoe = new THREE.Mesh(shoeGeo, new THREE.MeshStandardMaterial({color:0x212121}));
  lShoe.position.set(-0.12, 0.03, 0.02);
  const rShoe = new THREE.Mesh(shoeGeo, new THREE.MeshStandardMaterial({color:0x212121}));
  rShoe.position.set( 0.12, 0.03, 0.02);
  group.add(lShoe); group.add(rShoe);
  group.userData.outfitMeshes.shoes = lShoe;
  group.userData.rShoe = rShoe;

  // Accessory (necklace torus)
  const accGeo = new THREE.TorusGeometry(0.10, 0.016, 8, 20);
  const accMesh = new THREE.Mesh(accGeo, new THREE.MeshStandardMaterial({color:0xffd700}));
  accMesh.position.set(0, 1.05, 0.10);
  accMesh.rotation.x = Math.PI/2;
  group.add(accMesh);
  group.userData.outfitMeshes.accessory = accMesh;

  // Outerwear (slightly larger box over torso)
  const outerGeo = new THREE.BoxGeometry(0.42, 0.48, 0.26);
  const outerMesh = new THREE.Mesh(outerGeo, new THREE.MeshStandardMaterial({
    color:0x1b1b1b, transparent:true, opacity:0.0
  }));
  outerMesh.position.set(0, 0.78, 0);
  group.add(outerMesh);
  group.userData.outfitMeshes.outerwear = outerMesh;

  return group;
}

// Update a single outfit slot color on a character group
export function applyOutfitColor(group, slot, hexColor){
  const mesh = group.userData.outfitMeshes[slot];
  if(!mesh) return;
  mesh.material.color.setStyle(hexColor);
  if(slot === 'outerwear'){
    mesh.material.opacity = hexColor ? 0.92 : 0;
    mesh.material.transparent = true;
  }
  // mirror shoes
  if(slot === 'shoes' && group.userData.rShoe){
    group.userData.rShoe.material.color.setStyle(hexColor);
  }
}

