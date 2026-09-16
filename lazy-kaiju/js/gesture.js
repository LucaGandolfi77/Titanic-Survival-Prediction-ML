import * as THREE from 'three';
import { MathUtils } from './utils.js';

export class GestureRecognition {
    constructor() {
        this.enabled = false;
        this.lastGestureTime = 0;
        this.gestureCooldown = 800;
        this.onSlam = null;
        this.onYawn = null;
        this._video = null;
        this._activity = 0;
        this._prevActivity = 0;
        this._sampleCount = 0;
        this._motionThreshold = 30;
    }

    async activate() {
        try {
            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) return false;
            this._video = document.createElement('video');
            this._video.width = 320;
            this._video.height = 240;
            this._video.autoplay = true;

            const stream = await navigator.mediaDevices.getUserMedia({
                video: { width: 320, height: 240 },
                audio: false
            });
            this._video.srcObject = stream;
            this._stream = stream;
            this.enabled = true;
            this._startDetection();
            return true;
        } catch {
            return false;
        }
    }

    _startDetection() {
        if (!this._video) return;
        const canvas = document.createElement('canvas');
        canvas.width = 32;
        canvas.height = 24;
        const ctx = canvas.getContext('2d', { willReadFrequently: true });

        const detect = () => {
            if (!this.enabled) return;

            try {
                ctx.drawImage(this._video, 0, 0, 32, 24);
                const imageData = ctx.getImageData(0, 0, 32, 24);
                const data = imageData.data;

                let totalChange = 0;
                for (let i = 0; i < data.length; i += 4) {
                    const brightness = (data[i] + data[i+1] + data[i+2]) / 3;
                    if (i > 0) {
                        totalChange += Math.abs(brightness - this._prevActivity);
                    }
                    this._prevActivity = brightness;
                }

                this._activity = totalChange / (data.length / 4);
                this._sampleCount++;

                if (this._sampleCount % 10 === 0) {
                    this._classifyGesture();
                }
            } catch {}

            requestAnimationFrame(detect);
        };

        this._video.onloadedmetadata = () => {
            this._video.play();
            detect();
        };
    }

    _classifyGesture() {
        const now = Date.now();
        if (now - this.lastGestureTime < this.gestureCooldown) return;

        const acceleration = Math.abs(this._activity - this._prevActivity);

        if (acceleration > this._motionThreshold) {
            this.lastGestureTime = now;
            const gesture = Math.random() > 0.5 ? 'slam' : 'yawn';

            if (gesture === 'slam' && this.onSlam) {
                this.onSlam();
            } else if (gesture === 'yawn' && this.onYawn) {
                this.onYawn();
            }
        }
    }

    deactivate() {
        this.enabled = false;
        if (this._stream) {
            this._stream.getTracks().forEach(t => t.stop());
        }
    }
}