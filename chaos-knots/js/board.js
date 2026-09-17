const COLS = 6;
const ROWS = 8;
const THREAD_TYPES = ['red', 'blue', 'gold', 'shadow', 'chaos'];

const THREAD_COLORS = {
  red: '#FFB3BA',
  blue: '#BAE1FF',
  gold: '#FFF5BA',
  shadow: '#D4BAE1',
  chaos: '#B8E6D0',
};

const THREAD_EMOJI = {
  red: '🔴',
  blue: '🔵',
  gold: '🟡',
  shadow: '🟣',
  chaos: '🌀',
};

function createCell(type, isAnchor, isBoss) {
  return { type, isAnchor: !!isAnchor, isBoss: !!isBoss, hp: isBoss ? 3 : 1 };
}

function generateBoard(level) {
  const board = [];
  const numAnchors = Math.min(1 + Math.floor(level / 5), 4);
  const numBosses = level % 10 === 0 ? 1 : 0;
  const anchorPositions = new Set();
  const bossPositions = new Set();

  while (anchorPositions.size < numAnchors) {
    const pos = Math.floor(Math.random() * COLS * ROWS);
    anchorPositions.add(pos);
  }
  while (bossPositions.size < numBosses) {
    const pos = Math.floor(Math.random() * COLS * ROWS);
    if (!anchorPositions.has(pos)) bossPositions.add(pos);
  }

  for (let r = 0; r < ROWS; r++) {
    const row = [];
    for (let c = 0; c < COLS; c++) {
      const idx = r * COLS + c;
      if (bossPositions.has(idx)) {
        row.push(createCell('chaos', false, true));
      } else if (anchorPositions.has(idx)) {
        row.push(createCell('chaos', true, false));
      } else {
        const maxType = Math.min(2 + Math.floor(level / 3), THREAD_TYPES.length);
        const type = THREAD_TYPES[Math.floor(Math.random() * maxType)];
        row.push(createCell(type, false, false));
      }
    }
    board.push(row);
  }
  return board;
}

function findMatches(board) {
  const matched = new Set();

  for (let r = 0; r < ROWS; r++) {
    let c = 0;
    while (c < COLS) {
      if (!board[r][c]) { c++; continue; }
      let type = board[r][c].type;
      let count = 1;
      while (c + count < COLS && board[r][c + count] && board[r][c + count].type === type) {
        count++;
      }
      if (count >= 3) {
        for (let i = 0; i < count; i++) {
          matched.add(`${r},${c + i}`);
        }
      }
      c += Math.max(count, 1);
    }
  }

  for (let c = 0; c < COLS; c++) {
    let r = 0;
    while (r < ROWS) {
      if (!board[r][c]) { r++; continue; }
      let type = board[r][c].type;
      let count = 1;
      while (r + count < ROWS && board[r + count][c] && board[r + count][c].type === type) {
        count++;
      }
      if (count >= 3) {
        for (let i = 0; i < count; i++) {
          matched.add(`${r + i},${c}`);
        }
      }
      r += Math.max(count, 1);
    }
  }

  return [...matched].map((s) => {
    const [r, c] = s.split(',').map(Number);
    return { r, c };
  });
}

function removeCells(board, positions) {
  positions.forEach(({ r, c }) => {
    board[r][c] = null;
  });
}

function applyGravity(board) {
  for (let c = 0; c < COLS; c++) {
    let writeRow = ROWS - 1;
    for (let r = ROWS - 1; r >= 0; r--) {
      if (board[r][c]) {
        if (writeRow !== r) {
          board[writeRow][c] = board[r][c];
          board[r][c] = null;
        }
        writeRow--;
      }
    }
    for (let r = writeRow; r >= 0; r--) {
      board[r][c] = createCell(THREAD_TYPES[Math.floor(Math.random() * THREAD_TYPES.length)], false, false);
    }
  }
}

function hasValidMoves(board) {
  for (let r = 0; r < ROWS; r++) {
    for (let c = 0; c < COLS - 1; c++) {
      if (board[r][c] && board[r][c + 1]) {
        const temp = board[r][c];
        board[r][c] = board[r][c + 1];
        board[r][c + 1] = temp;
        const matches = findMatches(board);
        temp = board[r][c];
        board[r][c] = board[r][c + 1];
        board[r][c + 1] = temp;
        if (matches.length > 0) return true;
      }
    }
  }
  for (let r = 0; r < ROWS - 1; r++) {
    for (let c = 0; c < COLS; c++) {
      if (board[r][c] && board[r + 1][c]) {
        const temp = board[r][c];
        board[r][c] = board[r + 1][c];
        board[r + 1][c] = temp;
        const matches = findMatches(board);
        temp = board[r][c];
        board[r][c] = board[r + 1][c];
        board[r + 1][c] = temp;
        if (matches.length > 0) return true;
      }
    }
  }
  return false;
}

function shuffleBoard(board) {
  const cells = [];
  for (let r = 0; r < ROWS; r++) {
    for (let c = 0; c < COLS; c++) {
      cells.push(board[r][c]);
    }
  }
  for (let i = cells.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [cells[i], cells[j]] = [cells[j], cells[i]];
  }
  let idx = 0;
  for (let r = 0; r < ROWS; r++) {
    for (let c = 0; c < COLS; c++) {
      board[r][c] = cells[idx++];
    }
  }
}

function processBoard(board, onCascade) {
  let cascadeLevel = 0;
  const results = { matches: 0, cascades: 0 };

  let matches = findMatches(board);
  while (matches.length > 0) {
    results.matches += matches.length;
    removeCells(board, matches);
    applyGravity(board);
    cascadeLevel++;
    results.cascades = cascadeLevel - 1;
    if (onCascade) onCascade(cascadeLevel);
    matches = findMatches(board);
  }

  if (!hasValidMoves(board)) {
    shuffleBoard(board);
  }

  return results;
}
