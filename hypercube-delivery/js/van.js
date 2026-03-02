// Van mesh and physics
export class Van {
    constructor() {
        this.group = new THREE.Group();
        this.velocity = new THREE.Vector3();
        this.speed = 0;
        this.maxSpeed = 18;
        this.acceleration = 12;
        this.braking = 25;
        this.friction = 5;
        this.steering = 0;
        this.maxSteering = 2.5;
        
        this.createMesh();
    }

    createMesh() {
        const bodyMat = new THREE.MeshStandardMaterial({color: 0xffffff});
        const trimMat = new THREE.MeshStandardMaterial({color: 0x222222});
        
        // Main body
        const bodyGeo = new THREE.BoxGeometry(2.5, 1.5, 4.5);
        this.body = new THREE.Mesh(bodyGeo, bodyMat);
        this.body.position.y = 1.0;
        this.body.castShadow = true;
        this.group.add(this.body);

        // Cab
        const cabGeo = new THREE.BoxGeometry(2.2, 1.0, 1.8);
        const cab = new THREE.Mesh(cabGeo, bodyMat);
        cab.position.set(0, 1.25, 1.0);
        cab.castShadow = true;
        this.body.add(cab);

        // Wheels
        const wheelGeo = new THREE.CylinderGeometry(0.4, 0.4, 0.3, 16);
        wheelGeo.rotateZ(Math.PI/2);
        
        this.wheels = [];
        const wheelPositions = [
            [-1.3, 0.4, 1.5],
            [ 1.3, 0.4, 1.5],
            [-1.3, 0.4, -1.5],
            [ 1.3, 0.4, -1.5]
        ];

        wheelPositions.forEach(pos => {
            const wheel = new THREE.Mesh(wheelGeo, trimMat);
            wheel.position.set(...pos);
            this.wheels.push(wheel);
            this.group.add(wheel);
        });

        // Headlights
        const lightL = new THREE.PointLight(0xfff0dd, 1, 30);
        lightL.position.set(-0.8, 1.0, 2.3);
        const lightR = new THREE.PointLight(0xfff0dd, 1, 30);
        lightR.position.set(0.8, 1.0, 2.3);
        this.group.add(lightL);
        this.group.add(lightR);
    }

    update(dt, controls) {
        // Controls: controls.accel (0-1), controls.brake (bool), controls.steer (-1 to 1)
        
        // Acceleration
        if (controls.accel > 0) {
            this.speed += this.acceleration * controls.accel * dt;
        } else if (controls.reverse > 0) {
            this.speed -= this.acceleration * controls.reverse * dt;
        } else {
            // friction
            if (this.speed > 0) this.speed -= this.friction * dt;
            if (this.speed < 0) this.speed += this.friction * dt;
            if (Math.abs(this.speed) < 0.1) this.speed = 0;
        }

        if (controls.brake) {
            if (this.speed > 0) this.speed -= this.braking * dt;
            if (this.speed < 0) this.speed += this.braking * dt;
        }

        // Clamp speed
        this.speed = Math.max(-this.maxSpeed / 2, Math.min(this.speed, this.maxSpeed));

        // Steering (only if moving)
        const moveRatio = Math.abs(this.speed) / this.maxSpeed;
        if (moveRatio > 0.01) {
            const steerAmt = controls.steer * this.maxSteering * dt;
            // invert steering if reversing
            const dir = Math.sign(this.speed);
            this.group.rotateY(steerAmt * dir);
            
            // Visual steering for front wheels
            this.wheels[0].rotation.y = controls.steer * 0.5;
            this.wheels[1].rotation.y = controls.steer * 0.5;
        } else {
            this.wheels[0].rotation.y = 0;
            this.wheels[1].rotation.y = 0;
        }

        // Move forward
        const moveDir = new THREE.Vector3(0, 0, 1).applyQuaternion(this.group.quaternion);
        this.velocity.copy(moveDir).multiplyScalar(this.speed);
        
        this.group.position.add(this.velocity.clone().multiplyScalar(dt));

        // Wheel spin animation
        const spin = (this.speed * dt) / 0.4; // dist = speed*dt, radius = 0.4
        this.wheels.forEach(w => w.rotateX(spin));

        // Body roll based on turning and speed
        this.body.rotation.z = THREE.MathUtils.lerp(this.body.rotation.z, -controls.steer * moveRatio * 0.1, 0.1);
        
        // Simple bounds constraint (don't fall off the 100x100 tile for now, except portals)
        if (Math.abs(this.group.position.x) > 48) this.group.position.x = Math.sign(this.group.position.x) * 48;
        if (Math.abs(this.group.position.z) > 48) this.group.position.z = Math.sign(this.group.position.z) * 48;
    }
}