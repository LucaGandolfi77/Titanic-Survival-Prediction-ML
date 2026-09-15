import * as THREE from 'three';

export class SceneManager {
    constructor() {
        this.canvas = document.getElementById('game-canvas');
        this.postfxCanvas = document.getElementById('postfx-canvas');
        this.renderer = new THREE.WebGLRenderer({ canvas: this.canvas, antialias: true });
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
        this.renderer.toneMappingExposure = 1.2;
        this.initBloom();
        this.initSSAO();

        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x02020d);
        this.scene.fog = new THREE.FogExp2(0x02020d, 0.035);

        this.postfxCtx = this.postfxCanvas ? this.postfxCanvas.getContext('2d') : null;
        if(this.postfxCanvas) {
            this.postfxCanvas.width = window.innerWidth;
            this.postfxCanvas.height = window.innerHeight;
        }

        this.camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 100);
        this.cameraObj = new THREE.Group();
        this.cameraObj.position.set(0, 1.7, 0);
        this.cameraObj.add(this.camera);
        this.scene.add(this.cameraObj);

        const ambientLight = new THREE.HemisphereLight(0x0a0a1f, 0x1a0a00, 0.4);
        this.scene.add(ambientLight);

        this.barLight1 = new THREE.PointLight(0xff8c00, 2.5, 8);
        this.barLight1.position.set(-3, 1, -1);
        this.scene.add(this.barLight1);

        this.barLight2 = new THREE.PointLight(0xff8c00, 2.5, 8);
        this.barLight2.position.set(3, 1, -1);
        this.scene.add(this.barLight2);

        this.accentLight = new THREE.PointLight(0x00e5ff, 1.5, 12);
        this.accentLight.position.set(0, 3, -5);
        this.scene.add(this.accentLight);

        this.createDustParticles();

        this.particles = [];
        this.decals = [];

        // Menu background cup
        this.menuCup = this.createMenuCup();

        // Confetti particles for menu
        this.confetti = this.createConfetti(60);

        window.addEventListener('resize', this.onWindowResize.bind(this));
    }

    createMenuCup() {
        const group = new THREE.Group();
        
        // Cup body
        const cupGeo = new THREE.CylinderGeometry(0.5, 0.4, 1.2, 32);
        const cupMat = new THREE.MeshStandardMaterial({
            color: 0xffffff, roughness: 0.1, metalness: 0.1,
            transparent: true, opacity: 0.4
        });
        const cup = new THREE.Mesh(cupGeo, cupMat);
        group.add(cup);

        // Coffee liquid
        const liquidGeo = new THREE.CylinderGeometry(0.48, 0.48, 0.05, 32);
        const liquidMat = new THREE.MeshStandardMaterial({
            color: 0x3d1a00, roughness: 0.1,
            emissive: 0x3d1a00, emissiveIntensity: 0.2
        });
        const liquid = new THREE.Mesh(liquidGeo, liquidMat);
        liquid.position.y = 0.3;
        group.add(liquid);

        group.position.set(0, 1.5, -3);
        group.scale.set(2, 2, 2);
        this.scene.add(group);
        return group;
    }

    createConfetti(count) {
        const geo = new THREE.BoxGeometry(0.05, 0.05, 0.05);
        const mat = new THREE.MeshBasicMaterial({
            color: 0xff8c00, transparent: true, opacity: 0.6
        });
        const mesh = new THREE.InstancedMesh(geo, mat, count);
        const matrix = new THREE.Matrix4();
        this.confettiData = [];
        for(let i = 0; i < count; i++) {
            const x = (Math.random() - 0.5) * 10;
            const y = Math.random() * 8;
            const z = (Math.random() - 0.5) * 8 - 2;
            matrix.makeTranslation(x, y, z);
            mesh.setMatrixAt(i, matrix);
            this.confettiData.push({
                x, y, z,
                vx: (Math.random() - 0.5) * 0.02,
                vy: -Math.random() * 0.03 - 0.01,
                vz: (Math.random() - 0.5) * 0.02,
                rot: Math.random() * Math.PI * 2
            });
        }
        mesh.instanceMatrix.needsUpdate = true;
        this.scene.add(mesh);
        return mesh;
    }

    updateMenuConfetti(dt) {
        if(!this.confetti || !this.confettiData) return;
        const matrix = new THREE.Matrix4();
        for(let i = 0; i < this.confettiData.length; i++) {
            const c = this.confettiData[i];
            c.x += c.vx;
            c.y += c.vy;
            c.z += c.vz;
            c.rot += dt;
            if(c.y < -4) {
                c.y = 8;
                c.x = (Math.random() - 0.5) * 10;
                c.z = (Math.random() - 0.5) * 8 - 2;
            }
            matrix.makeTranslation(c.x, c.y, c.z);
            matrix.rotateY(c.rot);
            this.confetti.setMatrixAt(i, matrix);
        }
        this.confetti.instanceMatrix.needsUpdate = true;
    }

    updateMenuCup(time) {
        if(!this.menuCup) return;
        this.menuCup.rotation.y = time * 0.3;
        this.menuCup.position.y = 1.5 + Math.sin(time * 0.5) * 0.2;
    }

    onWindowResize() {
        this.camera.aspect = window.innerWidth / window.innerHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        if(this.postfxCanvas) {
            this.postfxCanvas.width = window.innerWidth;
            this.postfxCanvas.height = window.innerHeight;
        }
    }

    createDustParticles() {
        const count = 200;
        const geo = new THREE.BufferGeometry();
        const positions = new Float32Array(count * 3);
        for(let i = 0; i < count; i++) {
            positions[i*3] = (Math.random() - 0.5) * 12;
            positions[i*3+1] = Math.random() * 8;
            positions[i*3+2] = (Math.random() - 0.5) * 10;
        }
        geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        const mat = new THREE.PointsMaterial({
            color: 0xff8c00,
            size: 0.02,
            transparent: true,
            opacity: 0.3,
            sizeAttenuation: true
        });
        this.dust = new THREE.Points(geo, mat);
        this.scene.add(this.dust);
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

            if (p.mesh.material && p.mesh.material.opacity !== undefined) {
                p.mesh.material.opacity = 1.0 - (p.age / p.lifeTime);
            }
        }
    }

    updateDust(time) {
        if(!this.dust) return;
        this.dust.rotation.y = time * 0.00005;
        const pos = this.dust.geometry.attributes.position;
        for(let i = 0; i < pos.count; i++) {
            pos.array[i*3+1] += Math.sin(time * 0.001 + i) * 0.001;
            if(pos.array[i*3+1] > 8) pos.array[i*3+1] = 0;
        }
        pos.needsUpdate = true;
    }

    initBloom() {
        const w = window.innerWidth;
        const h = window.innerHeight;

        // Scene render target
        this.sceneTarget = new THREE.WebGLRenderTarget(w, h, {
            minFilter: THREE.LinearFilter,
            magFilter: THREE.LinearFilter,
            format: THREE.RGBAFormat,
            type: THREE.UnsignedByteType
        });

        // Bloom render targets (half resolution)
        const bw = Math.floor(w / 2);
        const bh = Math.floor(h / 2);
        this.bloomTargetA = new THREE.WebGLRenderTarget(bw, bh, {
            minFilter: THREE.LinearFilter,
            magFilter: THREE.LinearFilter,
            format: THREE.RGBAFormat
        });
        this.bloomTargetB = new THREE.WebGLRenderTarget(bw, bh, {
            minFilter: THREE.LinearFilter,
            magFilter: THREE.LinearFilter,
            format: THREE.RGBAFormat
        });

        // Bloom-only material (threshold)
        this.bloomMaterial = new THREE.ShaderMaterial({
            uniforms: {
                tDiffuse: { value: null },
                uThreshold: { value: 0.8 },
                uResolution: { value: new THREE.Vector2(bw, bh) }
            },
            vertexShader: `
                varying vec2 vUv;
                void main() {
                    vUv = uv;
                    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
                }
            `,
            fragmentShader: `
                uniform sampler2D tDiffuse;
                uniform float uThreshold;
                varying vec2 vUv;
                void main() {
                    vec4 color = texture2D(tDiffuse, vUv);
                    float brightness = dot(color.rgb, vec3(0.2126, 0.7152, 0.0722));
                    float contribution = max(0.0, brightness - uThreshold) / max(brightness, 0.001);
                    gl_FragColor = vec4(color.rgb * contribution, 1.0);
                }
            `
        });

        // Bloom blur (two-pass separable Gaussian)
        this.blurHMaterial = new THREE.ShaderMaterial({
            uniforms: {
                tDiffuse: { value: null },
                uResolution: { value: new THREE.Vector2(bw, bh) }
            },
            vertexShader: `
                varying vec2 vUv;
                void main() {
                    vUv = uv;
                    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
                }
            `,
            fragmentShader: `
                uniform sampler2D tDiffuse;
                uniform vec2 uResolution;
                varying vec2 vUv;
                void main() {
                    vec2 texel = 1.0 / uResolution;
                    float weight[5];
                    weight[0] = 0.227027;
                    weight[1] = 0.1945946;
                    weight[2] = 0.1216216;
                    weight[3] = 0.054054;
                    weight[4] = 0.016216;
                    vec2 offset[5];
                    offset[0] = vec2(0.0);
                    offset[1] = vec2(texel.x * 1.0, 0.0);
                    offset[2] = vec2(texel.x * 2.0, 0.0);
                    offset[3] = vec2(texel.x * 3.0, 0.0);
                    offset[4] = vec2(texel.x * 4.0, 0.0);
                    vec4 result = texture2D(tDiffuse, vUv) * weight[0];
                    for(int i = 1; i < 5; i++) {
                        result += texture2D(tDiffuse, vUv + offset[i]) * weight[i];
                        result += texture2D(tDiffuse, vUv - offset[i]) * weight[i];
                    }
                    gl_FragColor = result;
                }
            `
        });

        this.blurVMaterial = new THREE.ShaderMaterial({
            uniforms: {
                tDiffuse: { value: null },
                uResolution: { value: new THREE.Vector2(bw, bh) }
            },
            vertexShader: `
                varying vec2 vUv;
                void main() {
                    vUv = uv;
                    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
                }
            `,
            fragmentShader: `
                uniform sampler2D tDiffuse;
                uniform vec2 uResolution;
                varying vec2 vUv;
                void main() {
                    vec2 texel = 1.0 / uResolution;
                    float weight[5];
                    weight[0] = 0.227027;
                    weight[1] = 0.1945946;
                    weight[2] = 0.1216216;
                    weight[3] = 0.054054;
                    weight[4] = 0.016216;
                    vec2 offset[5];
                    offset[0] = vec2(0.0);
                    offset[1] = vec2(0.0, texel.y * 1.0);
                    offset[2] = vec2(0.0, texel.y * 2.0);
                    offset[3] = vec2(0.0, texel.y * 3.0);
                    offset[4] = vec2(0.0, texel.y * 4.0);
                    vec4 result = texture2D(tDiffuse, vUv) * weight[0];
                    for(int i = 1; i < 5; i++) {
                        result += texture2D(tDiffuse, vUv + offset[i]) * weight[i];
                        result += texture2D(tDiffuse, vUv - offset[i]) * weight[i];
                    }
                    gl_FragColor = result;
                }
            `
        });

        // Final composite material (scene + bloom + tone mapping)
        this.bloomCompositeMaterial = new THREE.ShaderMaterial({
            uniforms: {
                tScene: { value: null },
                tBloom: { value: null },
                uBloomStrength: { value: 1.2 },
                uExposure: { value: 1.2 }
            },
            vertexShader: `
                varying vec2 vUv;
                void main() {
                    vUv = uv;
                    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
                }
            `,
            fragmentShader: `
                uniform sampler2D tScene;
                uniform sampler2D tBloom;
                uniform float uBloomStrength;
                uniform float uExposure;
                varying vec2 vUv;
                void main() {
                    vec3 sceneColor = texture2D(tScene, vUv).rgb;
                    vec3 bloomColor = texture2D(tBloom, vUv).rgb;
                    vec3 color = sceneColor + bloomColor * uBloomStrength;
                    color = pow(color, vec3(1.0 / 2.2));
                    gl_FragColor = vec4(color, 1.0);
                }
            `
        });

        // Fullscreen quad for post-processing
        this.postfxQuad = new THREE.Mesh(
            new THREE.PlaneGeometry(2, 2),
            this.bloomCompositeMaterial
        );
        this.postfxScene = new THREE.Scene();
        this.postfxScene.add(this.postfxQuad);
        this.postfxCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
    }


    initSSAO() {
        const w = window.innerWidth;
        const h = window.innerHeight;

        this.ssaoTarget = new THREE.WebGLRenderTarget(w, h, {
            minFilter: THREE.LinearFilter,
            magFilter: THREE.LinearFilter,
            format: THREE.RGBAFormat
        });

        // Depth texture for SSAO sampling
        this.depthTexture = new THREE.DepthTexture(w, h);
        this.depthTexture.type = THREE.UnsignedShortType;
        this.sceneTarget.depthTexture = this.depthTexture;

        this.ssaoMaterial = new THREE.ShaderMaterial({
            uniforms: {
                tDiffuse: { value: null },
                tDepth: { value: null },
                uResolution: { value: new THREE.Vector2(w, h) },
                uRadius: { value: 0.5 },
                uStrength: { value: 0.5 },
                uNear: { value: 0.1 },
                uFar: { value: 100.0 }
            },
            vertexShader: `
                varying vec2 vUv;
                void main() {
                    vUv = uv;
                    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
                }
            `,
            fragmentShader: `
                uniform sampler2D tDiffuse;
                uniform sampler2D tDepth;
                uniform vec2 uResolution;
                uniform float uRadius;
                uniform float uStrength;
                uniform float uNear;
                uniform float uFar;
                varying vec2 vUv;
                
                float readDepth(vec2 uv) {
                    float z = texture2D(tDepth, uv).r;
                    return uNear + z * (uFar - uNear);
                }
                
                void main() {
                    vec4 color = texture2D(tDiffuse, vUv);
                    float depth = readDepth(vUv);
                    vec2 texel = 1.0 / uResolution;
                    
                    float ao = 0.0;
                    float totalWeight = 0.0;
                    
                    // Sample in a disk pattern
                    for(int x = -3; x <= 3; x++) {
                        for(int y = -3; y <= 3; y++) {
                            vec2 offset = vec2(float(x), float(y)) * texel * uRadius;
                            float sampleDepth = readDepth(vUv + offset);
                            float diff = depth - sampleDepth;
                            float weight = max(0.0, 1.0 - diff * 10.0);
                            ao += weight * step(0.001, abs(diff));
                            totalWeight += weight;
                        }
                    }
                    
                    ao = 1.0 - (ao / max(totalWeight, 0.001)) * uStrength;
                    gl_FragColor = vec4(color.rgb * ao, color.a);
                }
            `
        });

        this.ssaoQuad = new THREE.Mesh(
            new THREE.PlaneGeometry(2, 2),
            this.ssaoMaterial
        );
        this.ssaoScene = new THREE.Scene();
        this.ssaoScene.add(this.ssaoQuad);
        this.ssaoCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
    }

    renderPostFX() {
        if(!this.renderer || !this.sceneTarget) return;
        const w = window.innerWidth;
        const h = window.innerHeight;
        const bw = Math.floor(w / 2);
        const bh = Math.floor(h / 2);

        // Reinit on resize if needed
        if(this.sceneTarget.width !== w || this.sceneTarget.height !== h) {
            this.sceneTarget.setSize(w, h);
            const newBw = Math.floor(w / 2);
            const newBh = Math.floor(h / 2);
            this.bloomTargetA.setSize(newBw, newBh);
            this.bloomTargetB.setSize(newBw, newBh);
            this.bloomMaterial.uniforms.uResolution.value.set(newBw, newBh);
            this.blurHMaterial.uniforms.uResolution.value.set(newBw, newBh);
            this.blurVMaterial.uniforms.uResolution.value.set(newBw, newBh);
        }

        // 1. Render scene to sceneTarget
        this.renderer.setRenderTarget(this.sceneTarget);
        this.renderer.render(this.scene, this.camera);

        // 1.5 SSAO: apply ambient occlusion
        this.ssaoMaterial.uniforms.tDiffuse.value = this.sceneTarget.texture;
        this.renderer.setRenderTarget(this.ssaoTarget);
        this.renderer.render(this.ssaoScene, this.ssaoCamera);

        // 2. Render bloom pass (threshold) to bloomTargetA
        this.bloomMaterial.uniforms.tDiffuse.value = this.sceneTarget.texture;
        this.renderer.setRenderTarget(this.bloomTargetA);
        this.renderer.render(this.postfxScene, this.postfxCamera);

        // 3. Horizontal blur: bloomTargetA -> bloomTargetB
        this.blurHMaterial.uniforms.tDiffuse.value = this.bloomTargetA.texture;
        this.renderer.setRenderTarget(this.bloomTargetB);
        this.renderer.render(this.postfxScene, this.postfxCamera);

        // 4. Vertical blur: bloomTargetB -> bloomTargetA
        this.blurVMaterial.uniforms.tDiffuse.value = this.bloomTargetB.texture;
        this.renderer.setRenderTarget(this.bloomTargetA);
        this.renderer.render(this.postfxScene, this.postfxCamera);

        // 5. Composite: scene + bloom + SSAO to screen
        this.bloomCompositeMaterial.uniforms.tScene.value = this.ssaoTarget.texture;
        this.bloomCompositeMaterial.uniforms.tBloom.value = this.bloomTargetA.texture;
        this.renderer.setRenderTarget(null);
        this.renderer.render(this.postfxScene, this.postfxCamera);
    }

    render() {
        this.renderer.render(this.scene, this.camera);
        this.renderPostFX();
    }
}
