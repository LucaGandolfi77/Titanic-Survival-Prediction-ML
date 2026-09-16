export class IoTConnector {
    constructor() {
        this.connected = false;
        this.devices = [];
        this.mqttClient = null;
        this.wsUrl = null;
        this.autoConnect = true;
        this.lastTrigger = {};
        this.debounceMs = 5000;
    }

    connect(url) {
        this.wsUrl = url;

        try {
            if ('WebSocket' in window && url) {
                this.mqttClient = new WebSocket(url);

                this.mqttClient.onopen = () => {
                    this.connected = true;
                    this._subscribeTopics();
                    this._onStateChange?.('connected');
                };

                this.mqttClient.onmessage = (event) => {
                    this._handleMessage(JSON.parse(event.data));
                };

                this.mqttClient.onclose = () => {
                    this.connected = false;
                    this._onStateChange?.('disconnected');
                    if (this.autoConnect) {
                        setTimeout(() => this.connect(url), 5000);
                    }
                };

                this.mqttClient.onerror = () => {
                    this._onStateChange?.('error');
                };

                return true;
            }
        } catch {
            // Fallback: simulate
        }

        this.connected = true;
        this._simulateDevices();
        return true;
    }

    _subscribeTopics() {
        const topics = ['lazykaiju/#' ];
        if (this.mqttClient?.readyState === WebSocket.OPEN) {
            topics.forEach(t => this.mqttClient.send(JSON.stringify({ action: 'subscribe', topic: t })));
        }
    }

    _handleMessage(data) {
        if (data.topic && data.payload) {
            this._onMessage?.(data.topic, data.payload);
        }
    }

    _simulateDevices() {
        this.devices = [
            { id: 'light-living', name: 'Luci Living Room', type: 'light', state: 'on', icon: '💡' },
            { id: 'temp-bedroom', name: 'Temp Bedroom', type: 'sensor', state: '22°C', icon: '🌡️' },
            { id: 'light-kitchen', name: 'Luci Kitchen', type: 'light', state: 'off', icon: '💡' },
        ];
    }

    toggleDevice(deviceId) {
        const device = this.devices.find(d => d.id === deviceId);
        if (!device) return;

        const now = Date.now();
        if (this.lastTrigger[deviceId] && now - this.lastTrigger[deviceId] < this.debounceMs) return;
        this.lastTrigger[deviceId] = now;

        device.state = device.state === 'on' ? 'off' : 'on';

        if (this.mqttClient && this.connected) {
            this.mqttClient.send(JSON.stringify({
                topic: `lazykaiju/device/${deviceId}/set`,
                payload: { state: device.state }
            }));
        }

        this._onDeviceChange?.(device);
    }

    setKaijuTrigger(kaiju) {
        this._kaiju = kaiju;
    }

    checkAmbientTriggers(time, motion, light) {
        const hour = new Date().getHours();

        if (hour >= 22 || hour < 5) {
            if (this._kaiju && !this._kaiju.isMoving && !this._lastNightTrigger) {
                this._lastNightTrigger = Date.now();
                this._kaiju.yawn();
                return { trigger: 'night_yawn', hour };
            }
        }

        if (light < 50 && this._kaiju && this._kaiju.isMoving) {
            this.sceneMgr?.renderer?.setClearColor?.(0x1a1a2e);
        }

        if (motion && this._kaiju) {
            this._kaiju.targetYaw += (Math.random() - 0.5) * 0.5;
        }

        return null;
    }

    getDevices() {
        return this.devices;
    }

    isConnected() {
        return this.connected;
    }
}