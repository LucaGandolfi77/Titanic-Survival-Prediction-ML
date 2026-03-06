// js/scene3d.js — Three.js scene: characters, stage, camera
import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.169.0/build/three.module.js';
// lightweight custom orbital controls (avoid importing OrbitControls which uses bare 'three' specifier)
import { buildCharacter } from './character3d.js';
import { CHARACTERS } from './characters.js';
import { lerp } from './utils.js';

let renderer, camera, scene, controls;
let charGroups = {};     // charId -> THREE.Group
let slots = [];          // world positions for 8 character slots
const _targetCamPos = new THREE.Vector3();
const _lerpFactor = 0.06;
let _animCallbacks = []; // per-frame callbacks

export function initScene(canvas){
  renderer = new THREE.WebGLRenderer({canvas, antialias:true, alpha:false});
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.setPixelRatio(Math.min(window.devicePixelRatio,2));

  const w = canvas.clientWidth || canvas.offsetWidth || window.innerWidth * 0.65;
  const h = canvas.clientHeight || canvas.offsetHeight || window.innerHeight - 96;
  renderer.setSize(w, h, false);

  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0d0d1a);
  scene.fog = new THREE.Fog(0x0d0d1a, 20, 55);

  // Camera
  camera = new THREE.PerspectiveCamera(60, w/h, 0.1, 100);
  camera.position.set(0, 3.5, 10);
  camera.lookAt(0, 1, 0);

  // Lightweight custom controls (pointer drag to orbit, wheel to zoom)
  controls = {
    enableDamping: true,
    dampingFactor: 0.08,
    target: new THREE.Vector3(0, 1, 0),
    minDistance: 3,
    maxDistance: 22,
    maxPolarAngle: Math.PI * 0.55,
    _theta: Math.PI * 0.5,
    _phi: 0.35,
    _distance: 10,
    update() {
      const x = this.target.x + this._distance * Math.cos(this._phi) * Math.sin(this._theta);
      const y = this.target.y + this._distance * Math.sin(this._phi);
      const z = this.target.z + this._distance * Math.cos(this._phi) * Math.cos(this._theta);
      camera.position.lerp(new THREE.Vector3(x, y, z), this.enableDamping ? this.dampingFactor : 1);
      camera.lookAt(this.target);
    }
  };

  // Pointer handlers
  (function attachPointerHandlers(){
    const el = renderer.domElement;
    let isDown = false, lastX = 0, lastY = 0;
    el.addEventListener('pointerdown', e => { isDown = true; lastX = e.clientX; lastY = e.clientY; el.setPointerCapture?.(e.pointerId); });
    el.addEventListener('pointermove', e => {
      if(!isDown) return;
      const dx = (e.clientX - lastX) / 200;
      const dy = (e.clientY - lastY) / 200;
      controls._theta -= dx;
      controls._phi = Math.max(-controls.maxPolarAngle, Math.min(controls.maxPolarAngle, controls._phi + dy));
      lastX = e.clientX; lastY = e.clientY;
    });
    el.addEventListener('pointerup', e => { isDown = false; try{ el.releasePointerCapture?.(e.pointerId); }catch(_){} });
    el.addEventListener('wheel', e => { controls._distance = Math.max(controls.minDistance, Math.min(controls.maxDistance, controls._distance + e.deltaY * 0.02)); e.preventDefault(); }, { passive: false });
  })();

  // Lighting
  const ambient = new THREE.AmbientLight(0xffeedd, 0.8);
  scene.add(ambient);
  const sun = new THREE.DirectionalLight(0xffffff, 1.2);
  sun.position.set(5, 10, 8);
  sun.castShadow = true;
  sun.shadow.mapSize.set(1024, 1024);
  scene.add(sun);

  // Slot positions (8 chars in a semicircle)
  const R = 4.5;
  for(let i=0;i<8;i++){
    const a = Math.PI * (i / 7);
    slots.push(new THREE.Vector3(Math.cos(a)*R - R*0.5, 0, Math.sin(a)*1.8 - 1));
  }
  // Per-slot spotlight
  slots.forEach(pos=>{
    const spot = new THREE.SpotLight(0xff6b9d, 0.6, 8, Math.PI/8, 0.4);
    spot.position.set(pos.x, 5, pos.z);
    spot.target.position.copy(pos);
    scene.add(spot);
    scene.add(spot.target);
  });

  // Runway platform
  const runwayGeo = new THREE.BoxGeometry(12, 0.15, 5);
  const runwayMat = new THREE.MeshStandardMaterial({color:0x2a1f3d, roughness:.8});
  const runway = new THREE.Mesh(runwayGeo, runwayMat);
  runway.position.set(0, -0.075, 0);
  runway.receiveShadow = true;
  scene.add(runway);

  // Edge lights on runway
  [-5.5, 5.5].forEach(x=>{
    const edge = new THREE.PointLight(0xff6b9d, 0.4, 6);
    edge.position.set(x, 0.1, 0);
    scene.add(edge);
  });

  // City skyline (background boxes)
  _buildSkyline();

  // Ground plane
  const groundGeo = new THREE.PlaneGeometry(60, 60);
  const groundMat = new THREE.MeshStandardMaterial({color:0x0a0a14, roughness:1});
  const ground = new THREE.Mesh(groundGeo, groundMat);
  ground.rotation.x = -Math.PI/2;
  ground.position.y = -0.15;
  ground.receiveShadow = true;
  scene.add(ground);

  // Build characters
  CHARACTERS.forEach((c, i)=>{
    const group = buildCharacter(c);
    group.position.copy(slots[i]);
    scene.add(group);
    charGroups[c.id] = group;
  });

  // Resize observer
  const ro = new ResizeObserver(()=>{
    const w2 = canvas.clientWidth||canvas.offsetWidth;
    const h2 = canvas.clientHeight||canvas.offsetHeight;
    if(!w2||!h2) return;
    renderer.setSize(w2,h2,false);
    camera.aspect = w2/h2;
    camera.updateProjectionMatrix();
  });
  ro.observe(canvas.parentElement||document.body);

  // Animation loop
  let t = 0;
  function loop(){
    requestAnimationFrame(loop);
    t += 0.016;
    _animateChars(t);
    _animCallbacks.forEach(fn=>fn(t));
    // Smooth camera target lerp
    camera.position.lerp(_targetCamPos, _lerpFactor);
    controls.update();
    renderer.render(scene, camera);
  }
  _targetCamPos.copy(camera.position);
  loop();
}

function _buildSkyline(){
  const colors = [0x1a1a3e, 0x2a1a4e, 0x16213e, 0x0f3460];
  const accents= [0xff6b9d, 0x9c27b0, 0x00bcd4, 0xff9800, 0x4caf50];
  for(let i=0;i<20;i++){
    const w=0.5+Math.random()*1.2, h=1+Math.random()*5, d=0.4+Math.random()*0.8;
    const geo = new THREE.BoxGeometry(w,h,d);
    const mat = new THREE.MeshStandardMaterial({
      color: colors[Math.floor(Math.random()*colors.length)],
      emissive: accents[Math.floor(Math.random()*accents.length)],
      emissiveIntensity: 0.12,
      roughness:1
    });
    const m = new THREE.Mesh(geo, mat);
    m.position.set(-9+i*0.95, h/2-0.15, -6 - Math.random()*4);
    scene.add(m);
  }
}

function _animateChars(t){
  Object.values(charGroups).forEach(g=>{
    const anim = g.userData.animation || 'idle';
    const base = g.userData.baseY || 0;
    if(anim === 'idle'){
      g.position.y = base + Math.sin(t*1.2 + g.userData.phase)*0.02;
    } else if(anim === 'happy'){
      g.position.y = base + Math.abs(Math.sin(t*4))*0.08;
      if(g.userData.leftArm) g.userData.leftArm.rotation.z  = Math.sin(t*4)*0.5;
      if(g.userData.rightArm) g.userData.rightArm.rotation.z = -Math.sin(t*4)*0.5;
    } else if(anim === 'sad'){
      g.rotation.x = 0.15;
      g.position.y = base + Math.sin(t*0.5)*0.01;
    } else if(anim === 'flirt'){
      g.position.y = base + Math.sin(t*2)*0.03;
      g.rotation.y = Math.sin(t*1.5)*0.15;
    } else if(anim === 'fight'){
      g.position.z = g.userData.homeZ + Math.sin(t*8)*0.12;
    } else if(anim === 'dance'){
      g.position.y = base + Math.abs(Math.sin(t*3))*0.06;
      g.rotation.y += 0.02;
    } else if(anim === 'kiss'){
      g.position.y = base + Math.sin(t*0.8)*0.01;
    }
  });
}

export function setCharAnimation(charId, anim){
  const g = charGroups[charId];
  if(!g) return;
  g.userData.animation = anim;
  if(anim !== 'fight' && anim !== 'dance') g.rotation.set(0,0,0);
}

export function focusCharacter(charId){
  const g = charGroups[charId];
  if(!g) return;
  const p = g.position.clone();
  _targetCamPos.set(p.x, p.y+2, p.z+5);
  controls.target.lerp(new THREE.Vector3(p.x, p.y+1, p.z), 0.3);
}

export function resetCamera(){
  _targetCamPos.set(0, 3.5, 10);
  controls.target.set(0, 1, 0);
}

export function updateOutfitColor(charId, slot, hexColor){
  const g = charGroups[charId];
  if(!g) return;
  const mesh = g.userData.outfitMeshes && g.userData.outfitMeshes[slot];
  if(mesh) mesh.material.color.setStyle(hexColor);
}

export function spawnHeartParticles(charId){
  const g = charGroups[charId];
  if(!g) return;
  const h = document.createElement('div');
  h.className = 'particle-heart';
  h.textContent = '❤️';
  // position relative to 3d
  const canvas = renderer.domElement;
  const rect = canvas.getBoundingClientRect();
  const v = g.position.clone().project(camera);
  h.style.left = ((v.x+1)/2*rect.width + rect.left) + 'px';
  h.style.top  = ((-v.y+1)/2*rect.height + rect.top) + 'px';
  document.body.appendChild(h);
  setTimeout(()=>h.remove(), 1300);
}

export function showWeddingDecor(visible){
  if(!scene) return;
  // Remove/add arch
  scene.children.filter(c=>c.userData.wedding).forEach(c=>scene.remove(c));
  if(!visible) return;
  // Arch of tori
  const arc = new THREE.Group();
  arc.userData.wedding = true;
  for(let i=0;i<7;i++){
    const a = Math.PI * i / 6;
    const geo = new THREE.TorusGeometry(0.18, 0.04, 8, 16);
    const mat = new THREE.MeshStandardMaterial({color:0xffd700, emissive:0xffd700, emissiveIntensity:.4});
    const m = new THREE.Mesh(geo, mat);
    m.position.set(Math.cos(a)*1.4, Math.sin(a)*1.4 + 0.5, -1);
    arc.add(m);
  }
  scene.add(arc);
}

export function getScene(){ return scene; }
export function getCamera(){ return camera; }
export function getCharGroups(){ return charGroups; }

