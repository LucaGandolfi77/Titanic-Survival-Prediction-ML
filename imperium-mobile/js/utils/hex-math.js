export const HEX_SIZE = 24;
export const HEX_WIDTH = HEX_SIZE * 2;
export const HEX_HEIGHT = Math.sqrt(3) * HEX_SIZE;

export function hexDistance(q1, r1, q2, r2) {
  return (Math.abs(q1 - q2) + Math.abs(q1 + r1 - q2 - r2) + Math.abs(r1 - r2)) / 2;
}

export function hexNeighbor(q, r, direction) {
  const dirs = [
    [+1, 0], [-1, 0], [0, +1], [0, -1], [+1, -1], [-1, +1]
  ];
  return [q + dirs[direction][0], r + dirs[direction][1]];
}

export function hexNeighbors(q, r) {
  return [0, 1, 2, 3, 4, 5].map(d => hexNeighbor(q, r, d));
}

export function hexToPixel(q, r) {
  const x = HEX_SIZE * (3/2 * q);
  const y = HEX_SIZE * (Math.sqrt(3)/2 * q + Math.sqrt(3) * r);
  return { x, y };
}

export function pixelToHex(x, y) {
  const q = (2/3 * x) / HEX_SIZE;
  const r = (-1/3 * x + Math.sqrt(3)/3 * y) / HEX_SIZE;
  return hexRound(q, r);
}

function hexRound(q, r) {
  const s = -q - r;
  let rq = Math.round(q);
  let rr = Math.round(r);
  let rs = Math.round(s);
  const dq = Math.abs(rq - q);
  const dr = Math.abs(rr - r);
  const ds = Math.abs(rs - s);
  if (dq > dr && dq > ds) {
    rq = -rr - rs;
  } else if (dr > ds) {
    rr = -rq - rs;
  }
  return { q: rq, r: rr };
}

export function hexToPixelCenter(q, r) {
  const { x, y } = hexToPixel(q, r);
  return { x: x + HEX_SIZE, y: y + HEX_SIZE };
}

export function hexcorners(cx, cy) {
  const corners = [];
  for (let i = 0; i < 6; i++) {
    const angle = Math.PI / 180 * (60 * i - 30);
    corners.push({
      x: cx + HEX_SIZE * Math.cos(angle),
      y: cy + HEX_SIZE * Math.sin(angle)
    });
  }
  return corners;
}
