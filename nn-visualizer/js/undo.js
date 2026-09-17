export function createUndoSystem() {
  const undoStack = [];
  const redoStack = [];
  const maxHistory = 50;

  function push(state) {
    undoStack.push(JSON.stringify(state));
    if (undoStack.length > maxHistory) undoStack.shift();
    redoStack.length = 0;
  }

  function undo(currentState) {
    if (undoStack.length === 0) return null;
    redoStack.push(JSON.stringify(currentState));
    const prev = JSON.parse(undoStack.pop());
    return prev;
  }

  function redo(currentState) {
    if (redoStack.length === 0) return null;
    undoStack.push(JSON.stringify(currentState));
    const next = JSON.parse(redoStack.pop());
    return next;
  }

  function canUndo() {
    return undoStack.length > 0;
  }

  function canRedo() {
    return redoStack.length > 0;
  }

  function clear() {
    undoStack.length = 0;
    redoStack.length = 0;
  }

  function size() {
    return undoStack.length;
  }

  return { push, undo, redo, canUndo, canRedo, clear, size };
}
