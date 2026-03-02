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

        // Window resize handler
        window.addEventListener('resize', this.onWindowResize.bind(this));
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
        this.renderer.render(this.scene, this.camera);
    }
}