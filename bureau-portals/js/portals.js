import * as THREE from 'three';
import { MathUtils } from './utils.js';

// Portal types
export const PORTAL_TYPES = {
  NORMAL: 0,
  UPSIDE_DOWN: 1,
  SIDEWAYS_L: 2,
  SIDEWAYS_R: 3,
  FORWARD_DOWN: 4,
  ROTATED_45: 5,
  LOOP: 6,
  MIRROR: 7,
  SCALED: 8,
  TIME_LAG: 9,
  VOID: 10,
  CORRECT: 11
};

const PORTAL_COLORS = {
  [PORTAL_TYPES.NORMAL]: 0x4488ff,
  [PORTAL_TYPES.UPSIDE_DOWN]: 0xff2244,
  [PORTAL_TYPES.SIDEWAYS_L]: 0xff8800,
  [PORTAL_TYPES.SIDEWAYS_R]: 0xff8800,
  [PORTAL_TYPES.FORWARD_DOWN]: 0xff8800,
  [PORTAL_TYPES.ROTATED_45]: 0x22ff88,
  [PORTAL_TYPES.LOOP]: 0x22ff88,
  [PORTAL_TYPES.MIRROR]: 0x4488ff,
  [PORTAL_TYPES.SCALED]: 0xaa44ff,
  [PORTAL_TYPES.TIME_LAG]: 0xaa44ff,
  [PORTAL_TYPES.VOID]: 0x000000,
  [PORTAL_TYPES.CORRECT]: 0x4488ff
};

export class Portal {
  constructor(scene, position, rotation, type, destinationRoom, destinationPortal) {
    this.scene = scene;
    this.position = position.clone();
    this.rotation = rotation.clone();
    this.type = type;
    this.destinationRoom = destinationRoom;
    this.destinationPortal = destinationPortal;
    
    this.stencilID = 0;
    this.group = new THREE.Group();
    this.group.position.copy(position);
    this.group.rotation.copy(rotation);
    
    this.buildFrame();
    this.buildSurface();
    
    scene.add(this.group);
  }

  buildFrame() {
    const matFrame = new THREE.MeshLambertMaterial({ color: 0x8a8a8a });
    
    // Left pillar
    const leftPillar = new THREE.Mesh(
      new THREE.BoxGeometry(0.3, 3.5, 0.3),
      matFrame
    );
    leftPillar.position.x = -1.0;
    leftPillar.castShadow = true;
    this.group.add(leftPillar);
    
    // Right pillar
    const rightPillar = leftPillar.clone();
    rightPillar.position.x = 1.0;
    rightPillar.castShadow = true;
    this.group.add(rightPillar);
    
    // Top beam
    const topBeam = new THREE.Mesh(
      new THREE.BoxGeometry(2.6, 0.3, 0.3),
      matFrame
    );
    topBeam.position.y = 1.75;
    topBeam.castShadow = true;
    this.group.add(topBeam);
  }

  buildSurface() {
    const geo = new THREE.PlaneGeometry(2.0, 3.0);
    const material = this.createPortalMaterial();
    
    this.surfaceMesh = new THREE.Mesh(geo, material);
    this.surfaceMesh.position.z = 0.05;
    this.surfaceMesh.userData.isPortal = true;
    this.surfaceMesh.userData.portalType = this.type;
    this.group.add(this.surfaceMesh);
  }

  createPortalMaterial() {
    return new THREE.ShaderMaterial({
      uniforms: {
        uTime: { value: 0 },
        uColor: { value: new THREE.Color(PORTAL_COLORS[this.type]) },
        uType: { value: this.type }
      },
      vertexShader: `
        varying vec2 vUv;
        void main() {
          vUv = uv;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      fragmentShader: `
        uniform float uTime;
        uniform vec3 uColor;
        uniform int uType;
        varying vec2 vUv;
        
        void main() {
          vec2 uv = vUv;
          
          // Animated edge glow
          float edgeGlow = 0.0;
          edgeGlow += step(0.95, uv.x) + step(0.05, 1.0 - uv.x);
          edgeGlow += step(0.95, uv.y) + step(0.05, 1.0 - uv.y);
          
          // Distortion ripple
          float dist = length(uv - 0.5);
          float wave = sin(dist * 10.0 - uTime * 3.0) * 0.1;
          uv += normalize(uv - 0.5) * wave;
          
          // Void is pure black
          if (uType == 10) {
            gl_FragColor = vec4(0.0, 0.0, 0.0, 0.8);
            return;
          }
          
          // Chromatic aberration at edges
          float aberration = abs(edgeGlow) * 0.2;
          
          // Base color with edge glow
          float alpha = 0.6 + edgeGlow * 0.3;
          gl_FragColor = vec4(uColor * (1.0 + edgeGlow * 0.5), alpha);
        }
      `
    });
  }

  getDestinationCamera(mainCamera) {
    // Transform main camera through portal to destination
    const virtualCam = new THREE.PerspectiveCamera(
      mainCamera.fov,
      mainCamera.aspect,
      mainCamera.near,
      mainCamera.far
    );

    // Convert camera position to portal's local space
    const localPos = new THREE.Vector3().copy(mainCamera.position);
    this.group.worldToLocal(localPos);

    // Apply portal transformation based on type
    const transformedPos = this.applyPortalTransform(localPos);

    // Convert back to world space from destination portal
    if (this.destinationPortal) {
      this.destinationPortal.group.localToWorld(transformedPos);
      virtualCam.position.copy(transformedPos);

      // Calculate orientation
      const camQuat = new THREE.Quaternion();
      mainCamera.getWorldQuaternion(camQuat);
      
      const portalQuat = new THREE.Quaternion();
      this.group.getWorldQuaternion(portalQuat);
      
      const destQuat = new THREE.Quaternion();
      if (this.destinationPortal.group) {
        this.destinationPortal.group.getWorldQuaternion(destQuat);
      }

      const relQuat = new THREE.Quaternion();
      relQuat.multiplyQuaternions(destQuat, portalQuat.invert());
      relQuat.multiplyQuaternions(relQuat, camQuat);

      virtualCam.quaternion.copy(relQuat);
    }

    return virtualCam;
  }

  applyPortalTransform(localPos) {
    const result = localPos.clone();
    
    switch (this.type) {
      case PORTAL_TYPES.UPSIDE_DOWN:
        result.y = -result.y;
        break;
      case PORTAL_TYPES.SIDEWAYS_L:
        [result.x, result.z] = [-result.z, result.x];
        break;
      case PORTAL_TYPES.SIDEWAYS_R:
        [result.x, result.z] = [result.z, -result.x];
        break;
      case PORTAL_TYPES.FORWARD_DOWN:
        [result.y, result.z] = [result.z, result.y];
        break;
      case PORTAL_TYPES.MIRROR:
        result.x = -result.x;
        break;
      case PORTAL_TYPES.SCALED:
        result.multiplyScalar(0.5);
        break;
    }
    
    return result;
  }

  update(time) {
    if (this.surfaceMesh && this.surfaceMesh.material.uniforms) {
      this.surfaceMesh.material.uniforms.uTime.value = time;
    }
  }

  checkPlayerCrossing(prevPos, currPos) {
    // Check if player line segment crosses portal plane
    const planePoint = this.group.position.clone();
    const planeNormal = new THREE.Vector3(0, 0, 1).applyMatrix4(this.group.matrixWorld);
    planeNormal.normalize();

    const prevDist = MathUtils.pointToPlaneDistance(prevPos, planePoint, planeNormal);
    const currDist = MathUtils.pointToPlaneDistance(currPos, planePoint, planeNormal);

    // Crossed if signs differ
    return prevDist * currDist < 0;
  }

  getGravityAfterTransit() {
    switch (this.type) {
      case PORTAL_TYPES.UPSIDE_DOWN:
        return new THREE.Vector3(0, -1, 0);
      case PORTAL_TYPES.SIDEWAYS_L:
        return new THREE.Vector3(1, 0, 0);
      case PORTAL_TYPES.SIDEWAYS_R:
        return new THREE.Vector3(-1, 0, 0);
      case PORTAL_TYPES.FORWARD_DOWN:
        return new THREE.Vector3(0, 0, 1);
      default:
        return new THREE.Vector3(0, -1, 0);
    }
  }
}

export class PortalManager {
  constructor(scene, world) {
    this.scene = scene;
    this.world = world;
    this.portals = [];
    this.nextStencilID = 1;
  }

  createPortal(position, rotation, type, srcRoomID, dstRoomID) {
    const portal = new Portal(
      this.scene,
      position,
      rotation,
      type,
      this.world.getRoomByID(dstRoomID),
      null
    );

    portal.stencilID = this.nextStencilID++;
    this.portals.push(portal);
    return portal;
  }

  linkPortals(portal1ID, portal2ID) {
    const p1 = this.portals[portal1ID];
    const p2 = this.portals[portal2ID];
    
    if (p1 && p2) {
      p1.destinationPortal = p2;
      p2.destinationPortal = p1;
    }
  }

  update(time) {
    this.portals.forEach(p => p.update(time));
  }

  // Render scene through portal using stencil buffer
  renderPortalView(renderer, mainCamera, destinationScene) {
    // This is called per-portal from main render loop
    // Uses stencil technique
  }
}