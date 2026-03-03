import * as THREE from 'three';

export class SceneManager {
    constructor() {
        this.canvas = document.getElementById('game-canvas');
        this.renderer = new THREE.WebGLRenderer({ canvas: this.canvas, antialias: true });
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;

        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x8fb4c8);
        this.scene.fog = new THREE.FogExp2(0x8fb4c8, 0.015);

        // Lighting
        this.ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
        this.scene.add(this.ambientLight);

        this.dirLight = new THREE.DirectionalLight(0xfff0dd, 0.8);
        this.dirLight.position.set(50, 100, 50);
        this.dirLight.castShadow = true;
        this.dirLight.shadow.mapSize.width = 2048;
        this.dirLight.shadow.mapSize.height = 2048;
        this.dirLight.shadow.camera.near = 0.5;
        this.dirLight.shadow.camera.far = 300;
        this.dirLight.shadow.camera.left = -100;
        this.dirLight.shadow.camera.right = 100;
        this.dirLight.shadow.camera.top = 100;
        this.dirLight.shadow.camera.bottom = -100;
        this.scene.add(this.dirLight);

        // Add menu canvas setup
        this.initMenuScene();

        window.addEventListener('resize', this.onResize.bind(this));
    }

    initMenuScene() {
        // Will be used for UI background attract mode
    }

    onResize() {
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        if(this.camera) {
            this.camera.aspect = window.innerWidth / window.innerHeight;
            this.camera.updateProjectionMatrix();
        }
    }

    renderTargetCamera(camera) {
        this.camera = camera;
        this.renderer.render(this.scene, camera);
    }
    
    add(obj) { this.scene.add(obj); }
    remove(obj) { this.scene.remove(obj); }
}