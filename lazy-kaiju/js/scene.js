import * as THREE from 'three';

export class SceneManager {
    constructor() {
        this.canvas = document.getElementById('game-canvas');
        this.renderer = new THREE.WebGLRenderer({ canvas: this.canvas, antialias: true });
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;

        this.isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(
            navigator.userAgent
        ) || (window.innerWidth < 768);

        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x8fb4c8);
        this.scene.fog = new THREE.FogExp2(0x8fb4c8, this.isMobile ? 0.02 : 0.015);

        this.camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000);
        this.camera.position.set(0, 20, 40);
        this.camera.lookAt(0, 0, 0);

        this.ambientLight = new THREE.AmbientLight(0xffffff, this.isMobile ? 0.8 : 0.6);
        this.scene.add(this.ambientLight);

        this.dirLight = new THREE.DirectionalLight(0xfff0dd, this.isMobile ? 0.6 : 0.8);
        this.dirLight.position.set(50, 100, 50);
        this.dirLight.castShadow = true;
        this.dirLight.shadow.mapSize.width = this.isMobile ? 1024 : 2048;
        this.dirLight.shadow.mapSize.height = this.isMobile ? 1024 : 2048;
        this.dirLight.shadow.camera.near = 0.5;
        this.dirLight.shadow.camera.far = 300;
        this.dirLight.shadow.camera.left = -100;
        this.dirLight.shadow.camera.right = 100;
        this.dirLight.shadow.camera.top = 100;
        this.dirLight.shadow.camera.bottom = -100;
        this.dirLight.shadow.bias = -0.001;
        this.scene.add(this.dirLight);

        this.initMenuScene();
        window.addEventListener('resize', this.onResize.bind(this));
    }

    initMenuScene() {}

    onResize() {
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        if (this.camera) {
            this.camera.aspect = window.innerWidth / window.innerHeight;
            this.camera.updateProjectionMatrix();
        }
    }

    renderTargetCamera(camera) {
        this.camera = camera;
        this.renderer.render(this.scene, camera);
    }

    render() {
        if (!this.renderer) return;
        if (!this.camera) {
            this.camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000);
            this.camera.position.set(0, 20, 40);
            this.camera.lookAt(0, 0, 0);
        }
        this.renderer.render(this.scene, this.camera);
    }

    add(obj) { this.scene.add(obj); }
    remove(obj) { this.scene.remove(obj); }
}