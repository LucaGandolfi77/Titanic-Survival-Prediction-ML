import * as THREE from 'three';
import { Analytics } from './analytics.js';

export class LevelEditor {
    constructor(sceneManager, cityGenerator, onLevelChange) {
        this.sceneMgr = sceneManager;
        this.city = cityGenerator;
        this.onLevelChange = onLevelChange;
        this.isActive = false;

        this.gridSize = 200;
        this.cellSize = 10;
        this.gridDivisions = this.gridSize / this.cellSize;

        this.buildings = [];
        this.selectedType = 'normal';
        this.selectedHeight = 10;
        this.ecoCount = 0;
        this.levelName = 'My Level';
        this.trashCount = 40;

        this.analytics = new Analytics();
        this.analytics.track('editor_opened');

        this._setupInteraction();
    }

    activate() {
        this.isActive = true;
        this.analytics.track('editor_activated');
        this._clearCity();
        this._drawGrid();
        this._drawButtons();
    }

    deactivate() {
        this.isActive = false;
        this._clearOverlays();
    }

    _clearCity() {
        while (this.city.cityGroup.children.length > 0) {
            const child = this.city.cityGroup.children[0];
            this.city.cityGroup.remove(child);
            if (child.geometry) child.geometry.dispose();
            if (child.material) child.material.dispose();
        }
        this.city.buildings = [];
        this.city.roadTiles = [];
        this.buildings = [];
    }

    _drawGrid() {
        const gridGeo = new THREE.PlaneGeometry(this.gridSize, this.gridSize, this.gridDivisions, this.gridDivisions);
        const gridMat = new THREE.MeshBasicMaterial({
            color: 0x4a7c59,
            wireframe: true,
            transparent: true,
            opacity: 0.15
        });
        const grid = new THREE.Mesh(gridGeo, gridMat);
        grid.rotation.x = -Math.PI / 2;
        grid.position.y = 0.05;
        grid.userData.isGrid = true;
        this.sceneMgr.scene.add(grid);

        const groundGeo = new THREE.PlaneGeometry(this.gridSize, this.gridSize);
        const groundMat = new THREE.MeshLambertMaterial({ color: 0x2a2a2a });
        const ground = new THREE.Mesh(groundGeo, groundMat);
        ground.rotation.x = -Math.PI / 2;
        ground.userData.isGrid = true;
        this.sceneMgr.scene.add(ground);
    }

    _setupInteraction() {
        this._clickHandler = (event) => {
            if (!this.isActive) return;
            this._placeAtMouse(event);
        };
        this._contextHandler = (event) => {
            if (!this.isActive) return;
            event.preventDefault();
            this._removeAtMouse(event);
        };
        window.addEventListener('click', this._clickHandler);
        window.addEventListener('contextmenu', this._contextHandler);
    }

    _placeAtMouse(event) {
        const rect = this.sceneMgr.canvas.getBoundingClientRect();
        const mouse = new THREE.Vector2(
            ((event.clientX - rect.left) / rect.width) * 2 - 1,
            -((event.clientY - rect.top) / rect.height) * 2 + 1
        );

        const raycaster = new THREE.Raycaster();
        raycaster.setFromCamera(mouse, this.sceneMgr.camera);

        const groundIntersect = raycaster.intersectObjects(
            this.sceneMgr.scene.children.filter(c => c.userData.isGrid)
        );

        if (groundIntersect.length > 0) {
            const point = groundIntersect[0].point;
            this._placeBuilding(point.x, point.z);
        }
    }

    _removeAtMouse(event) {
        const rect = this.sceneMgr.canvas.getBoundingClientRect();
        const mouse = new THREE.Vector2(
            ((event.clientX - rect.left) / rect.width) * 2 - 1,
            -((event.clientY - rect.top) / rect.height) * 2 + 1
        );

        const raycaster = new THREE.Raycaster();
        raycaster.setFromCamera(mouse, this.sceneMgr.camera);

        const intersects = raycaster.intersectObjects(
            this.sceneMgr.scene.children.filter(c => c.userData.isBuilding)
        );

        if (intersects.length > 0) {
            const hit = intersects[0].object;
            const buildingData = this.buildings.find(b => b.mesh === hit);
            if (buildingData) {
                this.sceneMgr.scene.remove(hit);
                if (hit.geometry) hit.geometry.dispose();
                if (hit.material) hit.material.dispose();
                const idx = this.buildings.indexOf(buildingData);
                if (idx >= 0) {
                    if (buildingData.type === 'eco') this.ecoCount--;
                    this.buildings.splice(idx, 1);
                    this.analytics.track('building_removed', { type: buildingData.type });
                }
            }
        }
    }

    _placeBuilding(x, z) {
        const w = 5, d = 5, h = this.selectedHeight;
        const type = this.selectedType === 'trash_spawner' ? 'normal' : this.selectedType;

        let mat;
        if (type === 'eco') {
            mat = new THREE.MeshLambertMaterial({ color: 0x22c55e });
            this.ecoCount++;
        } else if (type === 'trash_spawner') {
            mat = new THREE.MeshLambertMaterial({ color: 0x999999 });
        } else {
            const palettes = [0xe8d5b0, 0xc4a882, 0xc8c8d4, 0xb8a090, 0xd4c4b0];
            mat = new THREE.MeshLambertMaterial({ color: palettes[Math.floor(Math.random() * palettes.length)] });
        }

        const geo = new THREE.BoxGeometry(w, h, d);
        const mesh = new THREE.Mesh(geo, mat);
        mesh.position.set(x, h / 2, z);
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        mesh.userData.isBuilding = true;
        this.sceneMgr.scene.add(mesh);

        const building = {
            mesh, type, w, h, d, x, z,
            eco: type === 'eco'
        };
        this.buildings.push(building);
        this.city.buildings.push({
            mesh: mesh,
            type: type,
            health: h < 10 ? 1 : (h < 18 ? 2 : 3),
            pos: mesh.position.clone(),
            size: new THREE.Vector3(w, h, d),
            active: true,
            isEco: type === 'eco'
        });

        if (type === 'eco') {
            this._addEcoDetails(mesh, w, h, d);
        }

        this.analytics.track('building_placed', { type, x, z, height: h });
    }

    _addEcoDetails(parent, w, h, d) {
        const solarGeo = new THREE.PlaneGeometry(w * 0.6, d * 0.6);
        const solarMat = new THREE.MeshBasicMaterial({ color: 0x003366, side: THREE.DoubleSide });
        const solar = new THREE.Mesh(solarGeo, solarMat);
        solar.position.set(0, h / 2 + 0.1, 0);
        solar.rotation.x = -Math.PI / 2;
        parent.add(solar);

        const light = new THREE.PointLight(0x22c55e, 1.0, 10);
        light.position.set(0, h / 2 + 1, 0);
        parent.add(light);
    }

    _drawButtons() {
        const container = document.createElement('div');
        container.id = 'editor-toolbar';
        container.style.cssText = `
            position: absolute; top: 10px; left: 10px; z-index: 200;
            background: rgba(13,17,23,0.9); border: 1px solid #4a7c59;
            border-radius: 8px; padding: 10px; display: flex; flex-direction: column;
            gap: 6px; font-family: 'Share Tech Mono', monospace;
        `;
        container.innerHTML = `
            <div style="color:#f0f6fc;font-size:0.9rem;margin-bottom:4px;">EDITOR TOOLBAR</div>
            <button data-type="normal" class="editor-btn" style="background:${this.selectedType==='normal'?'#4a7c59':'#161b22'};color:#fff;border:1px solid #6b7280;padding:4px 8px;border-radius:4px;cursor:pointer;">🏢 Normal</button>
            <button data-type="eco" class="editor-btn" style="background:${this.selectedType==='eco'?'#4a7c59':'#161b22'};color:#fff;border:1px solid #6b7280;padding:4px 8px;border-radius:4px;cursor:pointer;">🌿 Eco</button>
            <button data-type="trash_spawner" class="editor-btn" style="background:${this.selectedType==='trash_spawner'?'#4a7c59':'#161b22'};color:#fff;border:1px solid #6b7280;padding:4px 8px;border-radius:4px;cursor:pointer;">🗑️ Trash</button>
            <div style="color:#6b7280;font-size:0.8rem;margin-top:4px;">Height: <input type="range" id="editor-height" min="2" max="30" value="${this.selectedHeight}" style="width:80px;"></div>
            <div style="color:#6b7280;font-size:0.8rem;">Eco: ${this.ecoCount}</div>
            <button id="editor-save" style="background:#4a7c59;color:#fff;border:none;padding:6px;border-radius:4px;cursor:pointer;font-weight:bold;">💾 SAVE</button>
            <button id="editor-load" style="background:#161b22;color:#fff;border:1px solid #6b7280;padding:6px;border-radius:4px;cursor:pointer;">📂 LOAD</button>
            <button id="editor-play" style="background:#f97316;color:#fff;border:none;padding:6px;border-radius:4px;cursor:pointer;font-weight:bold;">▶ PLAY</button>
            <button id="editor-back" style="background:#ef4444;color:#fff;border:none;padding:6px;border-radius:4px;cursor:pointer;">← BACK</button>
        `;
        document.body.appendChild(container);

        container.querySelectorAll('.editor-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                container.querySelectorAll('.editor-btn').forEach(b => {
                    b.style.background = '#161b22';
                });
                btn.style.background = '#4a7c59';
                this.selectedType = btn.dataset.type;
            });
        });

        const heightInput = container.querySelector('#editor-height');
        if (heightInput) {
            heightInput.addEventListener('input', (e) => {
                this.selectedHeight = parseInt(e.target.value);
            });
        }

        const saveBtn = container.querySelector('#editor-save');
        if (saveBtn) saveBtn.addEventListener('click', () => this._saveLevel());
        const loadBtn = container.querySelector('#editor-load');
        if (loadBtn) loadBtn.addEventListener('click', () => this._loadLevel());
        const playBtn = container.querySelector('#editor-play');
        if (playBtn) playBtn.addEventListener('click', () => this._playLevel());
        const backBtn = container.querySelector('#editor-back');
        if (backBtn) backBtn.addEventListener('click', () => this._exitEditor());
    }

    _saveLevel() {
        const levelData = {
            version: 1,
            name: this.levelName,
            settings: {
                ecoCount: this.ecoCount,
                maxHeight: this.selectedHeight * 2,
                trashCount: this.trashCount
            },
            buildings: this.buildings.map(b => ({
                x: b.x, z: b.z, w: b.w, h: b.h, d: b.d, type: b.type
            }))
        };

        try {
            localStorage.setItem('lazykaiju_level', JSON.stringify(levelData));
            const encoded = btoa(unescape(encodeURIComponent(JSON.stringify(levelData))));
            navigator.clipboard.writeText(encoded).then(() => {
                this.analytics.track('level_saved', { buildings: this.buildings.length });
                alert('Level saved & copied to clipboard!');
            }).catch(() => {
                alert('Level saved locally! (clipboard blocked)');
            });
        } catch (e) {
            alert('Error saving: ' + e.message);
        }
    }

    _loadLevel() {
        try {
            const data = JSON.parse(localStorage.getItem('lazykaiju_level') || 'null');
            if (!data) { alert('No saved level found'); return; }
            this._clearCity();
            this.buildings = [];
            this.ecoCount = 0;

            for (const b of (data.buildings || [])) {
                this._placeBuilding(b.x, b.z);
                this.selectedHeight = b.h;
            }
            this.analytics.track('level_loaded', { buildings: this.buildings.length });
        } catch (e) {
            alert('Error loading: ' + e.message);
        }
    }

    _playLevel() {
        this.analytics.track('level_play', { buildings: this.buildings.length });
        this.deactivate();
        if (this.onLevelChange) {
            const cityConfig = {
                ecoCount: this.ecoCount,
                maxHeight: this.selectedHeight * 2,
                customBuildings: this.buildings.map(b => ({ x: b.x, z: b.z, w: b.w, h: b.h, d: b.d, type: b.type })),
                trashCount: this.trashCount
            };
            this.onLevelChange(cityConfig);
        }
    }

    _exitEditor() {
        this.analytics.track('editor_exited');
        this.deactivate();
    }

    _clearOverlays() {
        const toolbar = document.getElementById('editor-toolbar');
        if (toolbar) toolbar.remove();

        this.sceneMgr.scene.children.forEach(child => {
            if (child.userData.isGrid) {
                this.sceneMgr.scene.remove(child);
                if (child.geometry) child.geometry.dispose();
                if (child.material) child.material.dispose();
            }
        });
    }
}
