const UNRAVELERS = {
  ember: { name: 'Ember', icon: '🔥', rarity: 'common', desc: 'Burns all Red threads', cooldown: 6, color: '#FFB3BA' },
  ripple: { name: 'Ripple', icon: '💧', rarity: 'common', desc: 'Clears all Blue threads', cooldown: 6, color: '#BAE1FF' },
  spark: { name: 'Spark', icon: '✨', rarity: 'common', desc: 'Blasts a 3×3 area', cooldown: 5, color: '#FFF5BA' },
  flamy: { name: 'Flamy', icon: '🔥', rarity: 'rare', desc: 'Burns a full row + column', cooldown: 4, color: '#FF8FAB' },
  aquara: { name: 'Aquara', icon: '💧', rarity: 'rare', desc: 'Floods a row + column Blue', cooldown: 4, color: '#7BBFFF' },
  lux: { name: 'Lux', icon: '⭐', rarity: 'rare', desc: 'Cuts any single thread', cooldown: 3, color: '#FFE599' },
  shade: { name: 'Shade', icon: '🌑', rarity: 'epic', desc: 'Reshuffles the board', cooldown: 3, color: '#C9A9E0' },
  voidweaver: { name: 'Voidweaver', icon: '🌀', rarity: 'epic', desc: 'Clears all Chaos threads', cooldown: 3, color: '#99D6B8' },
  chaosUnraveler: { name: 'Chaos Unraveler', icon: '💫', rarity: 'legendary', desc: 'Nuclear option — clears all', cooldown: 2, color: '#FFD700' },
};

const RARITY_COST = {
  common: 50,
  rare: 150,
  epic: 400,
  legendary: 1000,
};

const RARITY_ORDER = ['common', 'rare', 'epic', 'legendary'];

function canDeploy(state, unravelerId) {
  const unraveler = UNRAVELERS[unravelerId];
  if (!unraveler) return false;
  if (!state.roster.includes(unravelerId)) return false;
  const cooldown = state.cooldowns?.[unravelerId] || 0;
  return cooldown <= 0;
}

function deployUnraveler(state, unravelerId, board, r, c) {
  const unraveler = UNRAVELERS[unravelerId];
  const level = state.unravelerLevels[unravelerId] || 1;
  const cellsToRemove = [];

  switch (unravelerId) {
    case 'ember':
      for (let rr = 0; rr < ROWS; rr++) {
        for (let cc = 0; cc < COLS; cc++) {
          if (board[rr][cc] && board[rr][cc].type === 'red') {
            cellsToRemove.push({ r: rr, c: cc });
          }
        }
      }
      break;
    case 'ripple':
      for (let rr = 0; rr < ROWS; rr++) {
        for (let cc = 0; cc < COLS; cc++) {
          if (board[rr][cc] && board[rr][cc].type === 'blue') {
            cellsToRemove.push({ r: rr, c: cc });
          }
        }
      }
      break;
    case 'spark':
      for (let dr = -1; dr <= 1; dr++) {
        for (let dc = -1; dc <= 1; dc++) {
          const nr = r + dr, nc = c + dc;
          if (nr >= 0 && nr < ROWS && nc >= 0 && nc < COLS && board[nr][nc]) {
            cellsToRemove.push({ r: nr, c: nc });
          }
        }
      }
      break;
    case 'flamy':
      for (let cc = 0; cc < COLS; cc++) cellsToRemove.push({ r, c: cc });
      for (let rr = 0; rr < ROWS; rr++) {
        if (rr !== r) cellsToRemove.push({ r: rr, c });
      }
      break;
    case 'aquara':
      for (let cc = 0; cc < COLS; cc++) {
        if (board[r][cc] && board[r][cc].type === 'blue') cellsToRemove.push({ r, c: cc });
      }
      for (let rr = 0; rr < ROWS; rr++) {
        if (rr !== r && board[rr][c] && board[rr][c].type === 'blue') cellsToRemove.push({ r: rr, c });
      }
      break;
    case 'lux':
      if (board[r][c]) cellsToRemove.push({ r, c });
      break;
    case 'shade':
      return null;
    case 'voidweaver':
      for (let rr = 0; rr < ROWS; rr++) {
        for (let cc = 0; cc < COLS; cc++) {
          if (board[rr][cc] && board[rr][cc].type === 'chaos') {
            cellsToRemove.push({ r: rr, c: cc });
          }
        }
      }
      break;
    case 'chaosUnraveler':
      for (let rr = 0; rr < ROWS; rr++) {
        for (let cc = 0; cc < COLS; cc++) {
          if (board[rr][cc]) cellsToRemove.push({ r: rr, c: cc });
        }
      }
      break;
  }

  return { cellsToRemove, unraveler };
}

function bossDamage(board, positions) {
  let damage = 0;
  positions.forEach(({ r, c }) => {
    if (board[r][c] && board[r][c].isBoss) {
      board[r][c].hp--;
      damage++;
      if (board[r][c].hp <= 0) {
        board[r][c] = null;
      }
    }
  });
  return damage;
}

function getAnchorNeighbors(board, positions) {
  const anchors = [];
  const posSet = new Set(positions.map((p) => `${p.r},${p.c}`));
  for (const { r, c } of positions) {
    const dirs = [[-1, 0], [1, 0], [0, -1], [0, 1]];
    for (const [dr, dc] of dirs) {
      const nr = r + dr, nc = c + dc;
      if (nr >= 0 && nr < ROWS && nc >= 0 && nc < COLS) {
        if (board[nr][nc] && board[nr][nc].isAnchor && !posSet.has(`${nr},${nc}`)) {
          anchors.push({ r: nr, c: nc });
        }
      }
    }
  }
  return anchors;
}
