export class PortalRenderer {
  constructor(sceneManager, portalManager) {
    this.renderer = sceneManager.renderer;
    this.camera = sceneManager.camera;
    this.scene = sceneManager.scene;
    this.portals = portalManager;
  }

  renderVisiblePortals() {
    const visiblePortals = [];

    this.portals.portals.forEach((portal) => {
      const distToCamera = this.camera.position.distanceTo(portal.position);
      if (distToCamera < 50) {
        visiblePortals.push(portal);
      }
    });

    visiblePortals.forEach((portal) => {
      this.renderPortalStencil(portal);
    });
  }

  renderPortalStencil(portal) {
    const gl = this.renderer.getContext();

    gl.enable(gl.STENCIL_TEST);
    gl.stencilMask(0xff);
    gl.colorMask(false, false, false, false);
    gl.stencilFunc(gl.ALWAYS, portal.stencilID, 0xff);
    gl.stencilOp(gl.KEEP, gl.KEEP, gl.REPLACE);

    this.renderer.render(portal.frameScene, this.camera);

    gl.colorMask(true, true, true, true);
    gl.stencilFunc(gl.EQUAL, portal.stencilID, 0xff);
    gl.stencilOp(gl.KEEP, gl.KEEP, gl.KEEP);

    const virtualCamera = portal.getDestinationCamera(this.camera);
    this.renderer.render(this.scene, virtualCamera);

    gl.colorMask(false, false, false, false);
    gl.depthFunc(gl.ALWAYS);
    this.renderer.render(portal.surfaceScene, this.camera);
    gl.depthFunc(gl.LESS);

    gl.stencilFunc(gl.ALWAYS, 0, 0);
    gl.stencilOp(gl.KEEP, gl.KEEP, gl.KEEP);
    gl.stencilMask(0);
    gl.depthFunc(gl.LESS);
  }
}
