// Package logic, delivery zones, rewards
export class PackageManager {
    constructor(sceneContainer) {
        this.container = sceneContainer;
        this.activePackages = [];
        this.pickups = [];   // Physical items in world
        this.deliveries = [];// Physical delivery zones
        
        this.packageGeo = new THREE.BoxGeometry(2, 2, 2);
        this.packageMat = new THREE.MeshStandardMaterial({ color: 0xffaa00 });
        this.zoneGeo = new THREE.RingGeometry(4, 5, 32);
        this.zoneGeo.rotateX(-Math.PI/2);
    }
    
    generatePackageForLevel(level, availableCells) {
        // Pick random start and end from available cells
        let startCell = availableCells[Math.floor(Math.random() * availableCells.length)];
        let endCell = availableCells[Math.floor(Math.random() * availableCells.length)];
        // Ensure they aren't the same
        while (startCell === endCell && availableCells.length > 1) {
            endCell = availableCells[Math.floor(Math.random() * availableCells.length)];
        }
        
        const pkg = {
            id: Date.now() + Math.random(),
            pickupCell: startCell,
            destinationCell: endCell,
            state: 'available', // available -> carried -> delivered
            timeLimit: Math.max(30, 90 - (level * 5)),
            timeRemaining: Math.max(30, 90 - (level * 5)),
            type: 'standard', // could expand to fragile, frozen
            points: 100 * level
        };
        
        this.activePackages.push(pkg);
        return pkg;
    }

    createVisuals(currentCell) {
        // Clear old
        this.pickups.forEach(p => this.container.remove(p));
        this.deliveries.forEach(d => this.container.remove(d));
        this.pickups = [];
        this.deliveries = [];

        this.activePackages.forEach(pkg => {
            if (pkg.state === 'available' && pkg.pickupCell === currentCell) {
                // Spawn pickup box
                const mesh = new THREE.Mesh(this.packageGeo, this.packageMat);
                mesh.position.set((Math.random()-0.5)*40, 1, (Math.random()-0.5)*40);
                mesh.userData = { isPickup: true, packageId: pkg.id };
                
                // Add glowing ring
                const zoneMat = new THREE.MeshBasicMaterial({color: 0x00ff00, side: THREE.DoubleSide});
                const zone = new THREE.Mesh(this.zoneGeo, zoneMat);
                zone.position.copy(mesh.position);
                zone.position.y = 0.1;
                
                this.container.add(mesh);
                this.container.add(zone);
                this.pickups.push({mesh: mesh, zone: zone, pkg: pkg});
            }
            
            if (pkg.state === 'carried' && pkg.destinationCell === currentCell) {
                // Spawn delivery zone
                const zoneMat = new THREE.MeshBasicMaterial({color: 0x00e5ff, side: THREE.DoubleSide});
                const zone = new THREE.Mesh(this.zoneGeo, zoneMat);
                zone.position.set((Math.random()-0.5)*40, 0.1, (Math.random()-0.5)*40);
                zone.userData = { isDelivery: true, packageId: pkg.id };
                
                this.container.add(zone);
                this.deliveries.push({zone: zone, pkg: pkg});
            }
        });
    }

    update(dt, vanPosition) {
        let stateChanged = false;
        let deliveredCount = 0;
        let scoreGained = 0;

        // Tick timers
        this.activePackages.forEach(pkg => {
            if (pkg.state === 'carried') {
                pkg.timeRemaining -= dt;
                if(pkg.timeRemaining <= 0) {
                    pkg.state = 'failed';
                    stateChanged = true;
                }
            }
        });

        // Remove failed
        if (stateChanged) {
            this.activePackages = this.activePackages.filter(p => p.state !== 'failed');
        }

        // Check pickups
        for (let i = this.pickups.length - 1; i >= 0; i--) {
            const p = this.pickups[i];
            if (vanPosition.distanceTo(p.mesh.position) < 6) {
                p.pkg.state = 'carried';
                this.container.remove(p.mesh);
                this.container.remove(p.zone);
                this.pickups.splice(i, 1);
                stateChanged = true;
            }
        }

        // Check deliveries
        for (let i = this.deliveries.length - 1; i >= 0; i--) {
            const d = this.deliveries[i];
            if (d.pkg.state === 'carried' && vanPosition.distanceTo(d.zone.position) < 6) {
                d.pkg.state = 'delivered';
                deliveredCount++;
                scoreGained += d.pkg.points + Math.floor(d.pkg.timeRemaining * 10);
                this.container.remove(d.zone);
                this.deliveries.splice(i, 1);
                stateChanged = true;
            }
        }
        
        // Remove delivered mathematically
        this.activePackages = this.activePackages.filter(p => p.state !== 'delivered');

        return { stateChanged, deliveredCount, scoreGained };
    }
}