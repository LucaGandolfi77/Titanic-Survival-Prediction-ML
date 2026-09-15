import * as THREE from 'three';
import { bus } from './event-bus.js';

const ALIEN_TYPES = [
    { type: 'blob', color: 0x39ff14, desc: "Blob" },
    { type: 'tentacle', color: 0x9c27b0, desc: "Tentacle" },
    { type: 'crystal', color: 0x00e5ff, desc: "Crystal" },
    { type: 'jellyfish', color: 0x00ff88, desc: "Jellyfish" },
    { type: 'robot', color: 0xaaaaaa, desc: "Robot" },
    { type: 'star', color: 0xffdd00, desc: "Star" }
];

export class CustomerSystem {
    constructor(scene, hud) {
        this.scene = scene;
        this.hud = hud;
        this.customers = [];
        this.orders = [];
        
        this.spawnTimer = 5;
        
        // Spawn positions next to the bar
        this.seats = [
            new THREE.Vector3(-2, 0, 0),
            new THREE.Vector3(0, 0, 0),
            new THREE.Vector3(2, 0, 0)
        ];
    }
    
    update(dt, stressMode = false) {
        const maxCustomers = stressMode ? 10 : 2;
        this.spawnTimer -= dt;
        if(this.spawnTimer <= 0 && this.customers.length < maxCustomers) {
            this.spawnCustomer();
            this.spawnTimer = 15 + Math.random() * 15;
        }

        for (let i = this.customers.length - 1; i >= 0; i--) {
            const cust = this.customers[i];
            
            // Animation bob
            cust.time += dt;
            cust.mesh.position.y = cust.baseY + Math.sin(cust.time * 2) * 0.1;
            
            // Update order time
            if (cust.order) {
                cust.order.timeLeft -= dt;
                
                if (cust.order.timeLeft <= 0) {
                    // Fail
                    this.removeCustomer(i, false);
                } else {
                    this.hud.updateOrder(cust.order);
                }
            }
        }
    }
    
    spawnCustomer() {
        // Find empty seat
        const occupied = this.customers.map(c => c.seatIndex);
        let seatIndex = -1;
        for(let i=0; i<this.seats.length; i++) {
            if(!occupied.includes(i)) {
                seatIndex = i;
                break;
            }
        }
        
        if(seatIndex === -1) return; // Full
        
        const template = ALIEN_TYPES[Math.floor(Math.random() * ALIEN_TYPES.length)];
        
        // Geometry based on type
        let geo;
        if(template.type === 'blob') geo = new THREE.SphereGeometry(0.35, 16, 16);
        else if(template.type === 'tentacle') geo = new THREE.CylinderGeometry(0.2, 0.4, 0.8);
        else if(template.type === 'crystal') geo = new THREE.IcosahedronGeometry(0.4, 0);
        else if(template.type === 'jellyfish') geo = new THREE.SphereGeometry(0.3, 12, 8);
        else if(template.type === 'robot') geo = new THREE.BoxGeometry(0.5, 0.55, 0.5);
        else geo = new THREE.OctahedronGeometry(0.45, 0);
        
        let mat = new THREE.MeshStandardMaterial({
            color: template.color,
            emissive: template.color,
            emissiveIntensity: 0.3,
            roughness: 0.3,
            metalness: 0.5
        });

        // Type-specific material upgrades
        if(template.type === 'robot') {
            mat.metalness = 0.9;
            mat.roughness = 0.2;
        } else if(template.type === 'crystal') {
            mat.metalness = 0.3;
            mat.roughness = 0.1;
            mat.emissiveIntensity = 0.5;
        } else if(template.type === 'jellyfish') {
            mat.transparent = true;
            mat.opacity = 0.6;
            mat.emissiveIntensity = 0.5;
        } else if(template.type === 'star') {
            mat.emissiveIntensity = 0.6;
            mat.metalness = 0.7;
        }
        const mesh = new THREE.Mesh(geo, mat);

        const glowLight = new THREE.PointLight(template.color, 0.8, 3);
        mesh.add(glowLight);
        
        const pos = this.seats[seatIndex].clone();
        pos.y = 0; // Seat height at floor level
        
        mesh.position.copy(pos);
        this.scene.add(mesh);
        
        const order = {
            id: Date.now(),
            icon: template.desc[0], // just a letter for now in HUD
            targetFill: 0.6 + Math.random() * 0.3, // 0.6 to 0.9
            sugarCount: Math.floor(Math.random() * 3),
            timeLeft: 45,
            maxTime: 45
        };
        
        this.hud.addOrder(order);
        
        this.customers.push({
            mesh,
            seatIndex,
            template,
            baseY: pos.y,
            time: Math.random() * 10,
            order
        });
        
        bus.emit("audio:playShift"); // Play generic spawn sound
    }
    
    checkDelivery(cup) {
        // Find if cup is near any customer
        for (let i = 0; i < this.customers.length; i++) {
            const cust = this.customers[i];
            const dist = cust.mesh.position.distanceTo(cup.position);
            
            if (dist < 1.5) {
                // Check if order matches
                const fillDiff = Math.abs(cup.fillLevel - cust.order.targetFill);
                if (fillDiff < 0.15) {
                    // Success!
                    let score = 100;
                    if(cup.scoreMultiplier) score = Math.round(100 * cup.scoreMultiplier);
                    if(fillDiff < 0.05) score += Math.round(50 * cup.scoreMultiplier);
                    
                    this.removeCustomer(i, true, score);
                    return true;
                }
            }
        }
        return false;
    }
    
    removeCustomer(index, success, score = 0) {
        const cust = this.customers[index];
        this.scene.remove(cust.mesh);
        cust.mesh.geometry.dispose();
        cust.mesh.material.dispose();
        
        this.hud.removeOrder(cust.order.id);
        
        if(success) {
            if(window.gameEngine) window.gameEngine.addScore(score);
            bus.emit("audio:playHappy");
        } else {
            if(window.gameEngine) window.gameEngine.addSpill(); // mark as fail
            bus.emit("audio:playAngry");
        }
        
        this.customers.splice(index, 1);
    }
}