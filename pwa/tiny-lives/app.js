/* =================================================================
   🏡 TINY LIVES — app.js
   Complete 3D social life-simulation engine
   ================================================================= */

// ── Constants ──────────────────────────────────────────────────────
const WORLD_SIZE = 40;
const TICK_MS = 2000;          // simulation tick interval
const TALK_RANGE = 3.5;
const TALK_DURATION = 4000;
const WANDER_SPEED = 0.04;
const NEED_DECAY = 0.35;       // per tick, base
const NEED_TALK_BOOST = 6;
const REL_TALK_BOOST = 4;

const MOODS = [
    { min: 80, emoji: '😄', label: 'Ecstatic' },
    { min: 60, emoji: '😊', label: 'Happy' },
    { min: 40, emoji: '😐', label: 'Fine' },
    { min: 20, emoji: '😟', label: 'Sad' },
    { min: 0,  emoji: '😫', label: 'Miserable' }
];

const SPEECH_LINES = [
    'Hey there!', 'Nice day!', 'How are you?', 'Ha ha ha!',
    'Tell me more!', 'Really?', 'No way!', 'That\'s great!',
    'Totally!', 'Interesting…', 'Oh wow!', 'Same here!',
    'Cool!', 'See ya!', 'Let\'s hang out!', 'Haha, yes!',

    // Funny / silly
    'I once trained a goldfish to fetch.',
    'I put pineapple on my pizza and it waved back.',
    'My llama sings in the shower.',
    'Danger: spontaneous dance attack incoming!',
    'Beware of invisible spaghetti.',
    'I can juggle one sock.',
    'That cloud looks like my ex\'s haircut.',
    'Free hugs for a limited time!',
    'I forgot why I walked here... but hi!',
    'If I were a sandwich, I\'d be an adventure.',

    // Crazy / mad
    'The pigeons have formed a union!',
    'We should build a rocket out of teapots.',
    'Aliens write my grocery list.',
    'My shoes are plotting a slow rebellion.',
    'I swear that bench winked at me.',
    'I\'m training to be a professional cloud inspector.',
    'Quick—hide the moon!',
    'I once argued with a lamppost and lost.',

    // Snarky / sassy
    'Oh wow, another boring day... said no one ever.',
    'I brought snacks. You\'re welcome.',
    'If sarcasm burned calories, I\'d be invisible.',
    'I\'m not lazy, I\'m on energy-saving mode.',

    // Short exclamations
    'Yippee!', 'Whoa!', 'Eek!', 'Yawn…', 'Boop!', 'Zing!','Pfft!',

    // Friendly / oddball
    'Fancy a duel? With water balloons.',
    'I collect rare dust bunnies.',
    'Let\'s swap secrets about the trees.',
    'I know a joke about a squirrel and a spreadsheet.',
    'If you listen closely the fountain hums off-key.'
];

const TRAIT_EFFECTS = {
    Outgoing:  { socialDecay: 1.4, talkChance: 1.5 },
    Shy:       { socialDecay: 0.7, talkChance: 0.4 },
    Lazy:      { moveChance: 0.4, energyDecay: 0.6 },
    Cheerful:  { moodRecover: 1.8 },
    Neat:      { hygieneDecay: 0.5 },
    Energetic: { moveChance: 1.4, energyDecay: 1.3 },
    Foodie:    { hungerDecay: 1.4 },
    Creative:  { funDecay: 0.6 },
    Grumpy:    { moodRecover: 0.5 },
    Athletic:  { energyDecay: 0.7, moveChance: 1.3 }
};

const RELATIONSHIP_STAGES = [
    { min: 0,  label: 'Stranger',     css: 'rel-stranger' },
    { min: 15, label: 'Acquaintance', css: 'rel-acquaintance' },
    { min: 40, label: 'Friend',       css: 'rel-friend' },
    { min: 70, label: 'Close Friend', css: 'rel-closefriend' }
];

// ── Character Definitions ──────────────────────────────────────────
const CHARACTER_DEFS = [
    { id: 'luna',   name: 'Luna',   age: 24, traits: ['Outgoing','Cheerful'],  bio: 'A social butterfly who lights up every room she enters.', avatar: '🌙', color: 0xff7eb3 },
    { id: 'max',    name: 'Max',    age: 31, traits: ['Athletic','Energetic'], bio: 'Loves jogging through the park at sunrise.', avatar: '🏃', color: 0x00b894 },
    { id: 'iris',   name: 'Iris',   age: 27, traits: ['Creative','Shy'],       bio: 'An artist who speaks through her paintings.', avatar: '🎨', color: 0xa29bfe },
    { id: 'oscar',  name: 'Oscar',  age: 45, traits: ['Grumpy','Neat'],        bio: 'The neighborhood\'s meticulous gardener.', avatar: '🌿', color: 0x636e72 },
    { id: 'daisy',  name: 'Daisy',  age: 22, traits: ['Cheerful','Foodie'],    bio: 'Always looking for the next great recipe.', avatar: '🌼', color: 0xfdcb6e },
    { id: 'felix',  name: 'Felix',  age: 35, traits: ['Lazy','Creative'],      bio: 'A dreamer who writes novels from his couch.', avatar: '📚', color: 0x74b9ff },
    { id: 'nova',   name: 'Nova',   age: 28, traits: ['Outgoing','Athletic'],  bio: 'Runs the local community center with big energy.', avatar: '⭐', color: 0xe17055 },
    { id: 'milo',   name: 'Milo',   age: 52, traits: ['Shy','Neat'],           bio: 'A quiet librarian who knows every book by heart.', avatar: '📖', color: 0x55a8b4 }
];

// ── State ──────────────────────────────────────────────────────────
let characters = [];
let paused = false;
let simSpeed = 1;
let simDay = 1;
let simHour = 8;
let simMinute = 0;
let selectedId = null;
let tickTimer = null;

// ── Three.js globals ───────────────────────────────────────────────
let scene, camera, renderer, raycaster, mouse;
let charMeshes = {};      // id → THREE.Group
let speechSprites = {};   // id → { div, timeout }
let groundGroup;
const bubbleContainer = document.createElement('div');
bubbleContainer.style.cssText = 'position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none;overflow:hidden;';
document.getElementById('scene-container').appendChild(bubbleContainer);

// ── Initialization ─────────────────────────────────────────────────
function init() {
    loadState();
    setupScene();
    buildWorld();
    createCharacterMeshes();
    setupInteraction();
    setupToolbar();
    startSimulation();
    animate();
    registerSW();
}

// ── Three.js Scene Setup ───────────────────────────────────────────
function setupScene() {
    const container = document.getElementById('scene-container');
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x87ceeb);
    scene.fog = new THREE.Fog(0x87ceeb, 50, 90);

    camera = new THREE.PerspectiveCamera(50, container.clientWidth / container.clientHeight, 0.1, 200);
    camera.position.set(0, 30, 35);
    camera.lookAt(0, 0, 0);

    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    container.appendChild(renderer.domElement);

    // Lights
    const ambient = new THREE.AmbientLight(0xffffff, 0.55);
    scene.add(ambient);

    const sun = new THREE.DirectionalLight(0xfff4e0, 0.9);
    sun.position.set(15, 25, 10);
    sun.castShadow = true;
    sun.shadow.mapSize.set(1024, 1024);
    sun.shadow.camera.left = -30;
    sun.shadow.camera.right = 30;
    sun.shadow.camera.top = 30;
    sun.shadow.camera.bottom = -30;
    scene.add(sun);

    const fill = new THREE.DirectionalLight(0xb4d8f0, 0.3);
    fill.position.set(-10, 10, -10);
    scene.add(fill);

    raycaster = new THREE.Raycaster();
    mouse = new THREE.Vector2();

    window.addEventListener('resize', () => {
        camera.aspect = container.clientWidth / container.clientHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(container.clientWidth, container.clientHeight);
    });
}

// ── World Building ─────────────────────────────────────────────────
function buildWorld() {
    groundGroup = new THREE.Group();
    scene.add(groundGroup);

    // Ground plane
    const groundGeo = new THREE.PlaneGeometry(WORLD_SIZE * 2, WORLD_SIZE * 2);
    const groundMat = new THREE.MeshLambertMaterial({ color: 0x7ec850 });
    const ground = new THREE.Mesh(groundGeo, groundMat);
    ground.rotation.x = -Math.PI / 2;
    ground.receiveShadow = true;
    groundGroup.add(ground);

    // Paths (cross shape)
    const pathMat = new THREE.MeshLambertMaterial({ color: 0xd4c4a8 });
    const pathH = new THREE.Mesh(new THREE.PlaneGeometry(WORLD_SIZE * 1.5, 3), pathMat);
    pathH.rotation.x = -Math.PI / 2;
    pathH.position.y = 0.01;
    groundGroup.add(pathH);

    const pathV = new THREE.Mesh(new THREE.PlaneGeometry(3, WORLD_SIZE * 1.5), pathMat);
    pathV.rotation.x = -Math.PI / 2;
    pathV.position.y = 0.01;
    groundGroup.add(pathV);

    // Plaza center
    const plazaGeo = new THREE.CircleGeometry(5, 32);
    const plazaMat = new THREE.MeshLambertMaterial({ color: 0xc8b896 });
    const plaza = new THREE.Mesh(plazaGeo, plazaMat);
    plaza.rotation.x = -Math.PI / 2;
    plaza.position.y = 0.02;
    groundGroup.add(plaza);

    // Fountain in center
    addFountain(0, 0);

    // Trees
    const treePositions = [
        [-12, -8], [-14, 6], [10, -12], [13, 8], [-8, 14],
        [6, -16], [-16, -14], [15, -6], [-5, -18], [18, 12],
        [-10, 18], [8, 18], [-18, 2], [20, -2], [-6, 10]
    ];
    treePositions.forEach(([x, z]) => addTree(x, z));

    // Benches
    addBench(-3, 6, 0);
    addBench(4, -5, Math.PI / 2);
    addBench(-7, -3, Math.PI);
    addBench(8, 7, -Math.PI / 4);

    // Lamp posts
    addLamp(-6, 6);
    addLamp(6, -6);
    addLamp(-6, -6);
    addLamp(6, 6);

    // Flower beds
    addFlowerBed(10, 4);
    addFlowerBed(-10, -5);
    addFlowerBed(3, 12);
}

function addTree(x, z) {
    const group = new THREE.Group();
    const trunkGeo = new THREE.CylinderGeometry(0.25, 0.35, 2, 6);
    const trunkMat = new THREE.MeshLambertMaterial({ color: 0x8B6914 });
    const trunk = new THREE.Mesh(trunkGeo, trunkMat);
    trunk.position.y = 1;
    trunk.castShadow = true;
    group.add(trunk);

    const leavesGeo = new THREE.SphereGeometry(1.8, 8, 6);
    const shade = 0x2d8a4e + Math.floor(Math.random() * 0x1a3a1a);
    const leavesMat = new THREE.MeshLambertMaterial({ color: shade });
    const leaves = new THREE.Mesh(leavesGeo, leavesMat);
    leaves.position.y = 3;
    leaves.castShadow = true;
    group.add(leaves);

    group.position.set(x, 0, z);
    groundGroup.add(group);
}

function addBench(x, z, rot) {
    const group = new THREE.Group();
    const seatGeo = new THREE.BoxGeometry(2.2, 0.15, 0.7);
    const seatMat = new THREE.MeshLambertMaterial({ color: 0x8B5E3C });
    const seat = new THREE.Mesh(seatGeo, seatMat);
    seat.position.y = 0.55;
    seat.castShadow = true;
    group.add(seat);

    const backGeo = new THREE.BoxGeometry(2.2, 0.6, 0.1);
    const back = new THREE.Mesh(backGeo, seatMat);
    back.position.set(0, 0.95, -0.3);
    group.add(back);

    for (const lx of [-0.85, 0.85]) {
        const legGeo = new THREE.BoxGeometry(0.12, 0.55, 0.5);
        const legMat = new THREE.MeshLambertMaterial({ color: 0x555555 });
        const leg = new THREE.Mesh(legGeo, legMat);
        leg.position.set(lx, 0.27, 0);
        group.add(leg);
    }

    group.position.set(x, 0, z);
    group.rotation.y = rot;
    groundGroup.add(group);
}

function addLamp(x, z) {
    const group = new THREE.Group();
    const poleGeo = new THREE.CylinderGeometry(0.08, 0.08, 3.5, 6);
    const poleMat = new THREE.MeshLambertMaterial({ color: 0x444444 });
    const pole = new THREE.Mesh(poleGeo, poleMat);
    pole.position.y = 1.75;
    group.add(pole);

    const bulbGeo = new THREE.SphereGeometry(0.25, 8, 8);
    const bulbMat = new THREE.MeshBasicMaterial({ color: 0xfff4c0 });
    const bulb = new THREE.Mesh(bulbGeo, bulbMat);
    bulb.position.y = 3.6;
    group.add(bulb);

    const light = new THREE.PointLight(0xfff4c0, 0.4, 8);
    light.position.y = 3.6;
    group.add(light);

    group.position.set(x, 0, z);
    groundGroup.add(group);
}

function addFountain(x, z) {
    const group = new THREE.Group();
    const baseGeo = new THREE.CylinderGeometry(1.8, 2, 0.6, 16);
    const baseMat = new THREE.MeshLambertMaterial({ color: 0xb0b0b0 });
    const base = new THREE.Mesh(baseGeo, baseMat);
    base.position.y = 0.3;
    base.castShadow = true;
    group.add(base);

    const waterGeo = new THREE.CylinderGeometry(1.5, 1.5, 0.1, 16);
    const waterMat = new THREE.MeshLambertMaterial({ color: 0x74b9ff, transparent: true, opacity: 0.7 });
    const water = new THREE.Mesh(waterGeo, waterMat);
    water.position.y = 0.6;
    group.add(water);

    const pillarGeo = new THREE.CylinderGeometry(0.2, 0.25, 1.5, 8);
    const pillar = new THREE.Mesh(pillarGeo, baseMat);
    pillar.position.y = 1.3;
    group.add(pillar);

    const topGeo = new THREE.SphereGeometry(0.35, 8, 8);
    const top = new THREE.Mesh(topGeo, baseMat);
    top.position.y = 2.1;
    group.add(top);

    group.position.set(x, 0, z);
    groundGroup.add(group);
}

function addFlowerBed(x, z) {
    const group = new THREE.Group();
    const bedGeo = new THREE.CylinderGeometry(1.2, 1.3, 0.3, 8);
    const bedMat = new THREE.MeshLambertMaterial({ color: 0x8B6914 });
    const bed = new THREE.Mesh(bedGeo, bedMat);
    bed.position.y = 0.15;
    group.add(bed);

    const flowerColors = [0xff6b6b, 0xffd93d, 0xff8fd4, 0xb8e0ff, 0xc8ff8f];
    for (let i = 0; i < 7; i++) {
        const fGeo = new THREE.SphereGeometry(0.2, 6, 4);
        const fMat = new THREE.MeshLambertMaterial({ color: flowerColors[i % flowerColors.length] });
        const f = new THREE.Mesh(fGeo, fMat);
        const angle = (i / 7) * Math.PI * 2;
        f.position.set(Math.cos(angle) * 0.7, 0.45, Math.sin(angle) * 0.7);
        group.add(f);
    }

    group.position.set(x, 0, z);
    groundGroup.add(group);
}

// ── Character Mesh Creation ────────────────────────────────────────
function createCharacterMeshes() {
    characters.forEach(c => {
        const group = new THREE.Group();
        group.userData.charId = c.id;

        // Body (capsule-like)
        const bodyGeo = new THREE.CylinderGeometry(0.35, 0.4, 1.2, 8);
        const bodyMat = new THREE.MeshLambertMaterial({ color: c.color });
        const body = new THREE.Mesh(bodyGeo, bodyMat);
        body.position.y = 1.0;
        body.castShadow = true;
        group.add(body);

        // Head
        const headGeo = new THREE.SphereGeometry(0.35, 10, 8);
        const headMat = new THREE.MeshLambertMaterial({ color: 0xfdd9b5 });
        const head = new THREE.Mesh(headGeo, headMat);
        head.position.y = 1.95;
        head.castShadow = true;
        group.add(head);

        // Eyes
        const eyeGeo = new THREE.SphereGeometry(0.06, 6, 4);
        const eyeMat = new THREE.MeshBasicMaterial({ color: 0x2c3e50 });
        for (const side of [-0.12, 0.12]) {
            const eye = new THREE.Mesh(eyeGeo, eyeMat);
            eye.position.set(side, 2.0, 0.28);
            group.add(eye);
        }

        // Name label (simple sprite)
        const canvas = document.createElement('canvas');
        canvas.width = 256;
        canvas.height = 64;
        const ctx = canvas.getContext('2d');
        ctx.font = 'bold 28px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillStyle = '#ffffff';
        ctx.strokeStyle = '#00000088';
        ctx.lineWidth = 4;
        ctx.strokeText(c.name, 128, 38);
        ctx.fillText(c.name, 128, 38);

        const tex = new THREE.CanvasTexture(canvas);
        const spriteMat = new THREE.SpriteMaterial({ map: tex, transparent: true });
        const sprite = new THREE.Sprite(spriteMat);
        sprite.position.y = 2.7;
        sprite.scale.set(2, 0.5, 1);
        group.add(sprite);

        // Selection ring (hidden by default)
        const ringGeo = new THREE.RingGeometry(0.55, 0.7, 24);
        const ringMat = new THREE.MeshBasicMaterial({ color: 0xffdd59, side: THREE.DoubleSide, transparent: true, opacity: 0.8 });
        const ring = new THREE.Mesh(ringGeo, ringMat);
        ring.rotation.x = -Math.PI / 2;
        ring.position.y = 0.05;
        ring.visible = false;
        ring.name = 'selectionRing';
        group.add(ring);

        // Position
        group.position.set(c.x, 0, c.z);
        scene.add(group);
        charMeshes[c.id] = group;
    });
}

// ── Character Data ─────────────────────────────────────────────────
function createCharacters() {
    return CHARACTER_DEFS.map((def, i) => {
        const angle = (i / CHARACTER_DEFS.length) * Math.PI * 2;
        const radius = 6 + Math.random() * 6;
        const relationships = {};
        CHARACTER_DEFS.forEach(other => {
            if (other.id !== def.id) relationships[other.id] = Math.floor(Math.random() * 10);
        });
        return {
            ...def,
            mood: 60 + Math.random() * 30,
            energy: 50 + Math.random() * 40,
            hunger: 50 + Math.random() * 40,
            social: 40 + Math.random() * 40,
            fun: 50 + Math.random() * 30,
            hygiene: 60 + Math.random() * 30,
            money: 100 + Math.floor(Math.random() * 400),
            currentAction: 'idle',
            relationships,
            x: Math.cos(angle) * radius,
            z: Math.sin(angle) * radius,
            targetX: 0,
            targetZ: 0,
            talkPartner: null,
            talkTimer: 0,
            idleTimer: 0,
            bobPhase: Math.random() * Math.PI * 2
        };
    });
}

// ── Trait Helpers ───────────────────────────────────────────────────
function traitFactor(char, key, fallback = 1) {
    for (const t of char.traits) {
        const fx = TRAIT_EFFECTS[t];
        if (fx && fx[key] !== undefined) return fx[key];
    }
    return fallback;
}

// ── Simulation ─────────────────────────────────────────────────────
function startSimulation() {
    if (tickTimer) clearInterval(tickTimer);
    tickTimer = setInterval(() => {
        if (paused) return;
        for (let s = 0; s < simSpeed; s++) simulationTick();
    }, TICK_MS);
}

function simulationTick() {
    advanceClock();
    characters.forEach(c => {
        decayNeeds(c);
        updateMood(c);
        behaviorUpdate(c);
    });
    checkProximityInteractions();
    saveState();
    updatePanelIfOpen();
}

function advanceClock() {
    simMinute += 15;
    if (simMinute >= 60) { simMinute = 0; simHour++; }
    if (simHour >= 24) { simHour = 0; simDay++; }
    const ampm = simHour >= 12 ? 'PM' : 'AM';
    const h = simHour % 12 || 12;
    const m = simMinute.toString().padStart(2, '0');
    document.getElementById('clock').textContent = `Day ${simDay} — ${h}:${m} ${ampm}`;
}

function decayNeeds(c) {
    const base = NEED_DECAY;
    c.energy  = clamp(c.energy  - base * traitFactor(c, 'energyDecay'));
    c.hunger  = clamp(c.hunger  - base * traitFactor(c, 'hungerDecay'));
    c.social  = clamp(c.social  - base * traitFactor(c, 'socialDecay'));
    c.fun     = clamp(c.fun     - base * 0.9);
    c.hygiene = clamp(c.hygiene - base * traitFactor(c, 'hygieneDecay') * 0.5);

    // Low hunger drags mood
    if (c.hunger < 20) c.mood = clamp(c.mood - 0.4);
    // Low energy drags mood
    if (c.energy < 15) c.mood = clamp(c.mood - 0.3);
}

function updateMood(c) {
    const avg = (c.energy + c.hunger + c.social + c.fun + c.hygiene) / 5;
    const target = avg;
    const speed = 0.08 * traitFactor(c, 'moodRecover');
    c.mood += (target - c.mood) * speed;
    c.mood = clamp(c.mood);
}

function behaviorUpdate(c) {
    if (c.currentAction === 'talking') {
        c.talkTimer -= TICK_MS;
        if (c.talkTimer <= 0) {
            c.currentAction = 'idle';
            c.talkPartner = null;
            c.idleTimer = 1500 + Math.random() * 2000;
            removeSpeechBubble(c.id);
        }
        return;
    }

    if (c.currentAction === 'resting') {
        c.energy = clamp(c.energy + 3);
        if (c.energy > 60) {
            c.currentAction = 'idle';
            c.idleTimer = 0;
        }
        return;
    }

    // Decide next action
    if (c.energy < 15) {
        c.currentAction = 'resting';
        return;
    }

    if (c.currentAction === 'idle') {
        c.idleTimer -= TICK_MS;
        if (c.idleTimer <= 0) {
            // Pick a new target to walk toward
            const moveChance = traitFactor(c, 'moveChance');
            if (Math.random() < 0.7 * moveChance) {
                c.targetX = (Math.random() - 0.5) * WORLD_SIZE * 0.8;
                c.targetZ = (Math.random() - 0.5) * WORLD_SIZE * 0.8;
                c.currentAction = 'walking';
            } else {
                c.idleTimer = 1500 + Math.random() * 2500;
            }
        }
    }
}

function checkProximityInteractions() {
    for (let i = 0; i < characters.length; i++) {
        for (let j = i + 1; j < characters.length; j++) {
            const a = characters[i];
            const b = characters[j];
            if (a.currentAction === 'talking' || b.currentAction === 'talking') continue;
            if (a.currentAction === 'resting' || b.currentAction === 'resting') continue;

            const dist = Math.hypot(a.x - b.x, a.z - b.z);
            if (dist < TALK_RANGE) {
                let chance = 0.25;
                chance *= traitFactor(a, 'talkChance');
                chance *= traitFactor(b, 'talkChance');
                if (a.social < 30) chance += 0.2;
                if (b.social < 30) chance += 0.2;

                if (Math.random() < chance) startConversation(a, b);
            }
        }
    }
}

function startConversation(a, b) {
    a.currentAction = 'talking';
    b.currentAction = 'talking';
    a.talkPartner = b.id;
    b.talkPartner = a.id;
    a.talkTimer = TALK_DURATION;
    b.talkTimer = TALK_DURATION;

    // Boost social + relationship
    a.social = clamp(a.social + NEED_TALK_BOOST);
    b.social = clamp(b.social + NEED_TALK_BOOST);
    a.fun = clamp(a.fun + 2);
    b.fun = clamp(b.fun + 2);

    const boost = REL_TALK_BOOST + Math.random() * 3;
    a.relationships[b.id] = clamp(a.relationships[b.id] + boost);
    b.relationships[a.id] = clamp(b.relationships[a.id] + boost);

    // Speech bubbles
    const lineA = SPEECH_LINES[Math.floor(Math.random() * SPEECH_LINES.length)];
    const lineB = SPEECH_LINES[Math.floor(Math.random() * SPEECH_LINES.length)];
    showSpeechBubble(a.id, lineA);
    setTimeout(() => showSpeechBubble(b.id, lineB), 600);

    // Face each other
    const meshA = charMeshes[a.id];
    const meshB = charMeshes[b.id];
    if (meshA && meshB) {
        meshA.lookAt(meshB.position.x, 0, meshB.position.z);
        meshB.lookAt(meshA.position.x, 0, meshA.position.z);
    }
}

// ── Movement (in render loop) ──────────────────────────────────────
function updateMovement(delta) {
    characters.forEach(c => {
        const mesh = charMeshes[c.id];
        if (!mesh) return;

        if (c.currentAction === 'walking') {
            const dx = c.targetX - c.x;
            const dz = c.targetZ - c.z;
            const dist = Math.hypot(dx, dz);

            if (dist < 0.3) {
                c.currentAction = 'idle';
                c.idleTimer = 1000 + Math.random() * 3000;
            } else {
                const speed = WANDER_SPEED * (c.energy > 20 ? 1 : 0.4) * simSpeed;
                c.x += (dx / dist) * speed * delta;
                c.z += (dz / dist) * speed * delta;

                // Face movement direction
                mesh.lookAt(c.targetX, 0, c.targetZ);
            }
        }

        // Bob animation
        c.bobPhase += delta * 0.004;
        const bobY = c.currentAction === 'walking' ? Math.sin(c.bobPhase * 3) * 0.08 : 0;
        mesh.position.set(c.x, bobY, c.z);

        // Selection ring
        const ring = mesh.getObjectByName('selectionRing');
        if (ring) ring.visible = (c.id === selectedId);
    });
}

// ── Speech Bubbles (screen-space overlay) ──────────────────────────
function showSpeechBubble(charId, text) {
    removeSpeechBubble(charId);
    const div = document.createElement('div');
    div.className = 'speech-bubble';
    div.textContent = text;
    bubbleContainer.appendChild(div);

    const timeout = setTimeout(() => removeSpeechBubble(charId), TALK_DURATION - 500);
    speechSprites[charId] = { div, timeout };
}

function removeSpeechBubble(charId) {
    const entry = speechSprites[charId];
    if (entry) {
        clearTimeout(entry.timeout);
        if (entry.div.parentNode) entry.div.parentNode.removeChild(entry.div);
        delete speechSprites[charId];
    }
}

function updateBubblePositions() {
    Object.keys(speechSprites).forEach(id => {
        const mesh = charMeshes[id];
        const entry = speechSprites[id];
        if (!mesh || !entry) return;

        const pos = new THREE.Vector3();
        pos.copy(mesh.position);
        pos.y += 3;
        pos.project(camera);

        const hw = renderer.domElement.clientWidth / 2;
        const hh = renderer.domElement.clientHeight / 2;
        const sx = pos.x * hw + hw;
        const sy = -pos.y * hh + hh;

        if (pos.z > 1) {
            entry.div.style.display = 'none';
        } else {
            entry.div.style.display = '';
            entry.div.style.left = sx + 'px';
            entry.div.style.top = sy + 'px';
        }
    });
}

// ── Interaction (raycasting + UI) ──────────────────────────────────
function setupInteraction() {
    const container = document.getElementById('scene-container');

    container.addEventListener('click', (e) => {
        const rect = renderer.domElement.getBoundingClientRect();
        mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
        mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

        raycaster.setFromCamera(mouse, camera);
        const meshList = Object.values(charMeshes).flatMap(g =>
            g.children.filter(ch => ch.isMesh)
        );
        const hits = raycaster.intersectObjects(meshList, false);

        if (hits.length > 0) {
            let parent = hits[0].object;
            while (parent && !parent.userData.charId) parent = parent.parent;
            if (parent && parent.userData.charId) {
                selectCharacter(parent.userData.charId);
                return;
            }
        }
        deselectCharacter();
    });

    document.getElementById('closePanel').addEventListener('click', deselectCharacter);

    // Hover cursor
    container.addEventListener('mousemove', (e) => {
        const rect = renderer.domElement.getBoundingClientRect();
        mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
        mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
        raycaster.setFromCamera(mouse, camera);
        const meshList = Object.values(charMeshes).flatMap(g =>
            g.children.filter(ch => ch.isMesh)
        );
        const hits = raycaster.intersectObjects(meshList, false);
        container.style.cursor = hits.length > 0 ? 'pointer' : 'default';
    });
}

function selectCharacter(id) {
    selectedId = id;
    updatePanel();
    document.getElementById('profile-panel').classList.remove('hidden');
}

function deselectCharacter() {
    selectedId = null;
    document.getElementById('profile-panel').classList.add('hidden');
}

// ── Panel Rendering ────────────────────────────────────────────────
function updatePanel() {
    const c = characters.find(ch => ch.id === selectedId);
    if (!c) return;

    document.getElementById('panelAvatar').textContent = c.avatar;
    document.getElementById('panelName').textContent = c.name;
    document.getElementById('panelAge').textContent = `Age ${c.age}`;
    document.getElementById('panelAction').textContent = actionLabel(c);
    document.getElementById('panelBio').textContent = c.bio;

    // Mood
    const moodInfo = getMoodInfo(c.mood);
    document.getElementById('panelMood').textContent = `${moodInfo.emoji} ${moodInfo.label} (${Math.round(c.mood)})`;

    // Traits
    const traitsEl = document.getElementById('panelTraits');
    traitsEl.innerHTML = c.traits.map(t => `<span class="trait-badge">${t}</span>`).join('');

    // Needs
    const needsEl = document.getElementById('panelNeeds');
    const needs = [
        { key: 'energy',  label: 'Energy' },
        { key: 'hunger',  label: 'Hunger' },
        { key: 'social',  label: 'Social' },
        { key: 'fun',     label: 'Fun' },
        { key: 'hygiene', label: 'Hygiene' },
    ];
    needsEl.innerHTML = needs.map(n => {
        const val = Math.round(c[n.key]);
        const color = val > 60 ? '#00b894' : val > 30 ? '#fdcb6e' : '#e17055';
        return `
            <div class="need-row">
                <span class="need-label">${n.label}</span>
                <div class="need-bar">
                    <div class="need-fill" style="width:${val}%;background:${color}"></div>
                </div>
                <span class="need-value">${val}</span>
            </div>`;
    }).join('');

    // Relationships
    const relsEl = document.getElementById('panelRels');
    const relEntries = Object.entries(c.relationships)
        .map(([id, score]) => {
            const other = characters.find(ch => ch.id === id);
            if (!other) return null;
            const stage = getRelStage(score);
            return `
                <div class="rel-row">
                    <span class="rel-name">${other.avatar} ${other.name}</span>
                    <span class="rel-stage ${stage.css}">${stage.label} (${Math.round(score)})</span>
                </div>`;
        })
        .filter(Boolean);
    relsEl.innerHTML = relEntries.join('');
}

function updatePanelIfOpen() {
    if (selectedId && !document.getElementById('profile-panel').classList.contains('hidden')) {
        updatePanel();
    }
}

function actionLabel(c) {
    switch (c.currentAction) {
        case 'walking': return '🚶 Walking';
        case 'talking': {
            const partner = characters.find(ch => ch.id === c.talkPartner);
            return `💬 Talking to ${partner ? partner.name : '…'}`;
        }
        case 'resting': return '😴 Resting';
        default: return '🧍 Idle';
    }
}

function getMoodInfo(mood) {
    for (const m of MOODS) {
        if (mood >= m.min) return m;
    }
    return MOODS[MOODS.length - 1];
}

function getRelStage(score) {
    for (let i = RELATIONSHIP_STAGES.length - 1; i >= 0; i--) {
        if (score >= RELATIONSHIP_STAGES[i].min) return RELATIONSHIP_STAGES[i];
    }
    return RELATIONSHIP_STAGES[0];
}

// ── Toolbar ────────────────────────────────────────────────────────
function setupToolbar() {
    document.getElementById('btnPause').addEventListener('click', () => {
        paused = true;
        document.getElementById('btnPause').disabled = true;
        document.getElementById('btnResume').disabled = false;
        showToast('Simulation paused');
    });

    document.getElementById('btnResume').addEventListener('click', () => {
        paused = false;
        document.getElementById('btnPause').disabled = false;
        document.getElementById('btnResume').disabled = true;
        showToast('Simulation resumed');
    });

    document.getElementById('btnSpeed').addEventListener('click', () => {
        const speeds = [1, 2, 4];
        const idx = (speeds.indexOf(simSpeed) + 1) % speeds.length;
        simSpeed = speeds[idx];
        document.getElementById('btnSpeed').textContent = `${simSpeed}×`;
        showToast(`Speed: ${simSpeed}×`);
    });

    document.getElementById('btnReset').addEventListener('click', () => {
        if (!confirm('Reset all progress?')) return;
        localStorage.removeItem('tinyLivesSave');
        Object.values(charMeshes).forEach(m => scene.remove(m));
        charMeshes = {};
        characters = createCharacters();
        createCharacterMeshes();
        simDay = 1; simHour = 8; simMinute = 0;
        selectedId = null;
        deselectCharacter();
        showToast('World reset!');
    });
}

// ── Toast ──────────────────────────────────────────────────────────
function showToast(msg) {
    const el = document.getElementById('toast');
    el.textContent = msg;
    el.classList.remove('hidden');
    clearTimeout(el._t);
    el._t = setTimeout(() => el.classList.add('hidden'), 2200);
}

// ── Persistence ────────────────────────────────────────────────────
function saveState() {
    const data = {
        characters: characters.map(c => ({
            id: c.id, mood: c.mood, energy: c.energy, hunger: c.hunger,
            social: c.social, fun: c.fun, hygiene: c.hygiene, money: c.money,
            currentAction: c.currentAction === 'talking' ? 'idle' : c.currentAction,
            relationships: c.relationships,
            x: c.x, z: c.z, talkPartner: null, talkTimer: 0
        })),
        simDay, simHour, simMinute
    };
    try { localStorage.setItem('tinyLivesSave', JSON.stringify(data)); } catch (e) { /* quota */ }
}

function loadState() {
    const raw = localStorage.getItem('tinyLivesSave');
    if (raw) {
        try {
            const data = JSON.parse(raw);
            characters = CHARACTER_DEFS.map(def => {
                const saved = data.characters.find(s => s.id === def.id);
                const base = createCharacters().find(ch => ch.id === def.id);
                if (saved) {
                    return { ...base, ...saved, ...def, color: def.color, avatar: def.avatar,
                             traits: def.traits, bio: def.bio, name: def.name, age: def.age,
                             targetX: saved.x, targetZ: saved.z, idleTimer: 0, bobPhase: Math.random() * Math.PI * 2 };
                }
                return base;
            });
            simDay = data.simDay || 1;
            simHour = data.simHour || 8;
            simMinute = data.simMinute || 0;
            return;
        } catch (e) { /* corrupt save, recreate */ }
    }
    characters = createCharacters();
}

// ── Render Loop ────────────────────────────────────────────────────
let prevTime = performance.now();

function animate() {
    requestAnimationFrame(animate);
    const now = performance.now();
    const delta = Math.min(now - prevTime, 100);
    prevTime = now;

    if (!paused) {
        updateMovement(delta);
    }
    updateBubblePositions();

    // Gentle camera orbit
    const t = now * 0.00003;
    camera.position.x = Math.sin(t) * 3;
    camera.lookAt(0, 0, 0);

    renderer.render(scene, camera);
}

// ── Utility ────────────────────────────────────────────────────────
function clamp(v, min = 0, max = 100) {
    return Math.max(min, Math.min(max, v));
}

// ── PWA Registration ───────────────────────────────────────────────
function registerSW() {
    if ('serviceWorker' in navigator) {
        navigator.serviceWorker.register('./service-worker.js')
            .then(() => console.log('SW registered'))
            .catch(err => console.warn('SW registration failed', err));
    }
}

// ── Boot ───────────────────────────────────────────────────────────
init();
