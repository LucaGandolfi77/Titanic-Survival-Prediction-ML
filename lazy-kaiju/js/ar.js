import * as THREE from 'three';

export class ARMode {
    constructor(sceneManager) {
        this.sceneMgr = sceneManager;
        this.isActive = false;
        this.kaijuScale = 1;
        this.tiltSensitivity = 2;
        this._deviceOrientation = null;
        this._originalBg = null;
        this._tablePlane = null;
    }

    async checkSupport() {
        const support = {
            xr: 'xr' in navigator,
            deviceOrientation: 'DeviceOrientationEvent' in window,
            webxr: false,
            ar: false
        };

        if (support.xr) {
            try {
                support.webxr = await navigator.xr.isSessionSupported('immersive-ar');
                support.ar = support.webxr;
            } catch {
                support.webxr = false;
                support.ar = false;
            }
        }

        return support;
    }

    activate() {
        this.isActive = true;
        this._originalBg = this.sceneMgr.scene.background;
        this.sceneMgr.scene.background = new THREE.Color(0xc4a882);

        this.sceneMgr.renderer.xr.enabled = true;

        window.addEventListener('deviceorientation', this._onOrientation);

        this._createTablePlane();
    }

    _createTablePlane() {
        const planeGeo = new THREE.PlaneGeometry(200, 200);
        const planeMat = new THREE.MeshBasicMaterial({
            color: 0x8b7355,
            transparent: true,
            opacity: 0.3
        });
        this._tablePlane = new THREE.Mesh(planeGeo, planeMat);
        this._tablePlane.rotation.x = -Math.PI / 2;
        this._tablePlane.position.y = 0;
        this.sceneMgr.scene.add(this._tablePlane);
    }

    _onOrientation = (e) => {
        this._deviceOrientation = {
            alpha: e.alpha || 0,
            beta: e.beta || 0,
            gamma: e.gamma || 0
        };

        if (this.kaiju) {
            const tiltZ = (e.gamma || 0) * this.tiltSensitivity * (Math.PI / 180);
            const tiltX = (e.beta || 0) * this.tiltSensitivity * (Math.PI / 180) * 0.5;
            this.kaiju.group.rotation.z = THREE.MathUtils.lerp(this.kaiju.group.rotation.z, tiltX, 0.1);
            this.kaiju.group.rotation.x = THREE.MathUtils.lerp(this.kaiju.group.rotation.x, tiltZ, 0.1);
        }
    };

    setKaijuRef(kaiju) {
        this.kaiju = kaiju;
    }

    getSupportMessage() {
        return {
            full: 'Il tuo dispositivo supporta WebXR AR!',
            orientation: 'Usando i sensori del dispositivo per il controllo.',
            unsupported: 'AR richiede un dispositivo mobile con sensore o WebXR.',
            fallback: 'Modalità AR semplificata attiva (tilt controls).'
        };
    }

    deactivate() {
        this.isActive = false;
        window.removeEventListener('deviceorientation', this._onOrientation);

        if (this._originalBg) {
            this.sceneMgr.scene.background = this._originalBg;
        }

        if (this._tablePlane) {
            this.sceneMgr.scene.remove(this._tablePlane);
            if (this._tablePlane.geometry) this._tablePlane.geometry.dispose();
            if (this._tablePlane.material) this._tablePlane.material.dispose();
            this._tablePlane = null;
        }

        this.sceneMgr.renderer.xr.enabled = false;
    }

    isSupported() {
        return 'DeviceOrientationEvent' in window || 'xr' in navigator;
    }
}