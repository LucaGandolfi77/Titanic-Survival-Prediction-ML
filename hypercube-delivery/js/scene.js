import * as THREE from 'three';

// Three.js renderer, camera, lights setup

export class SceneManager {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        
        // Renderer setup
        this.renderer = new THREE.WebGLRenderer({ 
            canvas: this.canvas, 
            antialias: true,
            powerPreference: "high-performance"
        });
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2)); // limit to 2 for perf
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
        this.renderer.toneMappingExposure = 1.2;

        // Scene setup
        this.scene = new THREE.Scene();
        this.scene.fog = new THREE.FogExp2('#1a237e', 0.015);
        
        // World Container - this is rotated during transitions, NOT the scene
        this.worldContainer = new THREE.Group();
        this.scene.add(this.worldContainer);
        
        // Camera setup
        this.camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        
        // Lighting
        this.ambientLight = new THREE.AmbientLight(0xffffff, 0.4);
        this.scene.add(this.ambientLight);
        
        this.directionalLight = new THREE.DirectionalLight(0xffffff, 1.0);
        this.directionalLight.position.set(50, 100, 50);
        this.directionalLight.castShadow = true;
        this.directionalLight.shadow.mapSize.width = 2048;
        this.directionalLight.shadow.mapSize.height = 2048;
        this.directionalLight.shadow.camera.near = 0.5;
        this.directionalLight.shadow.camera.far = 200;
        this.directionalLight.shadow.camera.left = -50;
        this.directionalLight.shadow.camera.right = 50;
        this.directionalLight.shadow.camera.top = 50;
        this.directionalLight.shadow.camera.bottom = -50;
        this.directionalLight.shadow.bias = -0.0005;
        this.scene.add(this.directionalLight);

        this.isContextLost = false;

        // WebGL Context Loss handling
        this.canvas.addEventListener('webglcontextlost', (e) => {
            e.preventDefault();
            this.isContextLost = true;
            console.warn('WebGL context lost. Attempting recovery...');
        });
        this.canvas.addEventListener('webglcontextrestored', () => {
            console.log('WebGL context restored.');
            this.isContextLost = false;
            this.handleContextRestore();
        });

        // Window resize handler
        this._resizeHandler = this.onWindowResize.bind(this);
        window.addEventListener('resize', this._resizeHandler);
    }

    dispose() {
        window.removeEventListener('resize', this._resizeHandler);
        this.renderer.dispose();
        this.renderer.forceContextLoss();
    }

    handleContextLoss() {
        this.isContextLost = true;
        if (this._contextLostHandler) {
            this._contextLostHandler();
        }
    }

    handleContextRestore() {
        this.isContextLost = false;
    }

    onWindowResize() {
        this.camera.aspect = window.innerWidth / window.innerHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(window.innerWidth, window.innerHeight);
    }
    
    setEnvironmentColor(skyColor, fogColor, fogDensity) {
        this.scene.background = new THREE.Color(skyColor);
        this.scene.fog.color = new THREE.Color(fogColor);
        this.scene.fog.density = fogDensity;
        this.directionalLight.color = new THREE.Color(skyColor).lerp(new THREE.Color(0xffffff), 0.7);
    }

    render() {
        if (this.isContextLost) return;
        this.renderer.render(this.scene, this.camera);
    }
}