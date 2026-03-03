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
        this.scene.background = new THREE.Color(0x02020d);

        // Core camera
        this.camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 100);
        this.cameraObj = new THREE.Group();
        this.cameraObj.position.set(0, 1.7, 0); // Eye height
        this.cameraObj.add(this.camera);
        this.scene.add(this.cameraObj);
        
        // Base lights
        const ambientLight = new THREE.HemisphereLight(0x0a0a1f, 0x1a0a00, 0.4);
        this.scene.add(ambientLight);

        this.particles = [];
        this.decals = [];

        window.addEventListener('resize', this.onWindowResize.bind(this));
    }

    onWindowResize() {
        this.camera.aspect = window.innerWidth / window.innerHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(window.innerWidth, window.innerHeight);
    }

    addParticle(mesh, velocity, lifeTime) {
        this.scene.add(mesh);
        this.particles.push({ mesh, velocity, lifeTime, age: 0 });
    }

    addDecal(mesh) {
        this.scene.add(mesh);
        this.decals.push(mesh);
        if(this.decals.length > 50) {
            const oldDecal = this.decals.shift();
            this.scene.remove(oldDecal);
            oldDecal.geometry.dispose();
            oldDecal.material.dispose();
        }
    }

    updateParticles(dt, gravityVec) {
        for (let i = this.particles.length - 1; i >= 0; i--) {
            const p = this.particles[i];
            p.age += dt;
            if (p.age >= p.lifeTime) {
                this.scene.remove(p.mesh);
                if (p.mesh.geometry) p.mesh.geometry.dispose();
                if (p.mesh.material) p.mesh.material.dispose();
                this.particles.splice(i, 1);
                continue;
            }
            
            p.velocity.addScaledVector(gravityVec, dt);
            p.mesh.position.addScaledVector(p.velocity, dt);
            
            // Fade out
            if (p.mesh.material && p.mesh.material.opacity !== undefined) {
                p.mesh.material.opacity = 1.0 - (p.age / p.lifeTime);
            }
            if (p.mesh.material && p.mesh.material.size !== undefined) {
               // shrink points if needed
            }
        }
    }

    render() {
        this.renderer.render(this.scene, this.camera);
    }
}
