export const fog = {
  revealed: new Set(),
  visible: new Set(),

  reveal(q, r) {
    this.revealed.add(`${q},${r}`);
  },

  revealArea(centerQ, centerR, radius) {
    for (let dr = -radius; dr <= radius; dr++) {
      for (let dq = -radius; dq <= radius; dq++) {
        if (Math.abs(dq) + Math.abs(dr) + Math.abs(dq + dr) > radius * 2) continue;
        this.reveal(centerQ + dq, centerR + dr);
      }
    }
  },

  isRevealed(q, r) {
    return this.revealed.has(`${q},${r}`);
  },

  setVisible(q, r) {
    this.visible.add(`${q},${r}`);
  },

  isVisible(q, r) {
    return this.visible.has(`${q},${r}`);
  },

  clear() {
    this.revealed.clear();
    this.visible.clear();
  },
};
