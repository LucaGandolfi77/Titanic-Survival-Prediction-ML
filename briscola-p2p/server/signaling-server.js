/**
 * @fileoverview Lightweight WebSocket signaling server for Briscola P2P.
 *
 * Relays SDP offers/answers and ICE candidates between two peers so they
 * can establish a direct WebRTC DataChannel.  No game state is stored on
 * the server — it only routes connection-setup messages.
 *
 * Message protocol:
 *   CLIENT → SERVER
 *     create_room  { roomCode }
 *     join_room    { roomCode }
 *     rejoin_room  { roomCode, role }
 *     offer        { roomCode, sdp }
 *     answer       { roomCode, sdp }
 *     ice          { roomCode, candidate }
 *     leave        { roomCode }
 *
 *   SERVER → CLIENT
 *     room_created   { roomCode }
 *     room_joined    { roomCode, isHost }
 *     guest_joined
 *     peer_ready
 *     offer          { sdp }
 *     answer         { sdp }
 *     ice            { candidate }
 *     error          { message }
 *     peer_left
 */

'use strict';

const { WebSocketServer, WebSocket } = require('ws');

/* ─── Configuration ──────────────────────────────────────────── */

const PORT              = parseInt(process.env.PORT, 10) || 8080;
const ROOM_TIMEOUT_MS   = 10 * 60 * 1000;   // 10 minutes
const HEARTBEAT_MS      = 30 * 1000;         // 30 seconds
const RECONNECT_GRACE   = 120 * 1000;        // 2 minute grace on disconnect
const CLEANUP_INTERVAL  = 60 * 1000;         // check stale rooms every minute

/* ─── Room storage ───────────────────────────────────────────── */

/**
 * @typedef {Object} Room
 * @property {WebSocket|null} host
 * @property {WebSocket|null} guest
 * @property {number}         createdAt
 * @property {number}         lastActivity
 * @property {NodeJS.Timeout|null} hostTimer
 * @property {NodeJS.Timeout|null} guestTimer
 * @property {NodeJS.Timeout|null} expiryTimer
 */

/** @type {Map<string, Room>} */
const rooms = new Map();

/* ─── Server bootstrap ───────────────────────────────────────── */

const wss = new WebSocketServer({ port: PORT });
console.log(`🃏 Briscola signaling server listening on port ${PORT}`);

/* ─── Heartbeat (ping / pong) ────────────────────────────────── */

const heartbeat = setInterval(() => {
  for (const ws of wss.clients) {
    if (ws.isAlive === false) { ws.terminate(); continue; }
    ws.isAlive = false;
    ws.ping();
  }
}, HEARTBEAT_MS);

wss.on('close', () => clearInterval(heartbeat));

/* ─── Connection handler ─────────────────────────────────────── */

wss.on('connection', (ws) => {
  ws.isAlive = true;
  ws.on('pong', () => { ws.isAlive = true; });

  ws.on('message', (raw) => {
    let msg;
    try { msg = JSON.parse(raw); }
    catch { return sendTo(ws, { type: 'error', message: 'Invalid JSON' }); }
    handleMessage(ws, msg);
  });

  ws.on('close', () => handleDisconnect(ws));
});

/* ─── Message router ─────────────────────────────────────────── */

/**
 * Route an incoming message to the appropriate handler.
 * @param {WebSocket} ws
 * @param {Object}    msg
 */
function handleMessage(ws, msg) {
  const { type, roomCode } = msg;

  switch (type) {
    case 'create_room':  return handleCreateRoom(ws, roomCode);
    case 'join_room':    return handleJoinRoom(ws, roomCode);
    case 'rejoin_room':  return handleRejoinRoom(ws, roomCode, msg.role);
    case 'offer':        return relay(ws, roomCode, { type: 'offer',  sdp: msg.sdp });
    case 'answer':       return relay(ws, roomCode, { type: 'answer', sdp: msg.sdp });
    case 'ice':          return relay(ws, roomCode, { type: 'ice', candidate: msg.candidate });
    case 'leave':        return handleLeave(ws, roomCode);
    default:             return sendTo(ws, { type: 'error', message: `Unknown type: ${type}` });
  }
}

/* ─── Room lifecycle ─────────────────────────────────────────── */

/**
 * Create a new room with the caller as host.
 * @param {WebSocket} ws
 * @param {string}    roomCode
 */
function handleCreateRoom(ws, roomCode) {
  if (!isValidCode(roomCode)) {
    return sendTo(ws, { type: 'error', message: 'Invalid room code (need 6 alphanumeric chars)' });
  }
  if (rooms.has(roomCode)) {
    return sendTo(ws, { type: 'error', message: 'Room already exists' });
  }

  /** @type {Room} */
  const room = {
    host: ws,
    guest: null,
    createdAt: Date.now(),
    lastActivity: Date.now(),
    hostTimer: null,
    guestTimer: null,
    expiryTimer: setTimeout(() => destroyRoom(roomCode), ROOM_TIMEOUT_MS),
  };

  rooms.set(roomCode, room);
  ws.roomCode = roomCode;
  ws.role = 'host';

  sendTo(ws, { type: 'room_created', roomCode });
  log(`Room ${roomCode} created`);
}

/**
 * Join an existing room as guest.
 * @param {WebSocket} ws
 * @param {string}    roomCode
 */
function handleJoinRoom(ws, roomCode) {
  const room = rooms.get(roomCode);
  if (!room)       return sendTo(ws, { type: 'error', message: 'Room not found' });
  if (room.guest)  return sendTo(ws, { type: 'error', message: 'Room is full' });

  room.guest = ws;
  touchRoom(room);
  ws.roomCode = roomCode;
  ws.role = 'guest';

  sendTo(ws, { type: 'room_joined', roomCode, isHost: false });
  if (room.host && room.host.readyState === WebSocket.OPEN) {
    sendTo(room.host, { type: 'guest_joined' });
  }
  log(`Guest joined room ${roomCode}`);
}

/**
 * Re-join a room after a page transition (landing → game page).
 * @param {WebSocket} ws
 * @param {string}    roomCode
 * @param {string}    role
 */
function handleRejoinRoom(ws, roomCode, role) {
  const room = rooms.get(roomCode);
  if (!room) return sendTo(ws, { type: 'error', message: 'Room not found' });

  if (role !== 'host' && role !== 'guest') {
    return sendTo(ws, { type: 'error', message: 'Invalid role' });
  }

  /* Clear any pending disconnect timer for this slot */
  const timerKey = role === 'host' ? 'hostTimer' : 'guestTimer';
  if (room[timerKey]) { clearTimeout(room[timerKey]); room[timerKey] = null; }

  room[role] = ws;
  touchRoom(room);
  ws.roomCode = roomCode;
  ws.role = role;

  sendTo(ws, { type: 'room_joined', roomCode, isHost: role === 'host' });

  /* Notify both sides when the pair is complete */
  if (bothConnected(room)) {
    sendTo(room.host,  { type: 'peer_ready' });
    sendTo(room.guest, { type: 'peer_ready' });
    log(`Both players ready in room ${roomCode}`);
  }
}

/**
 * Explicit leave: notify opponent and destroy the room.
 * @param {WebSocket} ws
 * @param {string}    roomCode
 */
function handleLeave(ws, roomCode) {
  const room = rooms.get(roomCode);
  if (!room) return;

  const other = getOther(room, ws.role);
  if (other) sendTo(other, { type: 'peer_left' });
  destroyRoom(roomCode);
}

/**
 * Implicit disconnect (WebSocket close).  Start a grace-period timer so
 * the player can reconnect (e.g. after a page redirect).
 * @param {WebSocket} ws
 */
function handleDisconnect(ws) {
  const { roomCode, role } = ws;
  if (!roomCode || !role) return;
  const room = rooms.get(roomCode);
  if (!room) return;

  const timerKey = role === 'host' ? 'hostTimer' : 'guestTimer';
  room[timerKey] = setTimeout(() => {
    room[role] = null;
    const other = getOther(room, role);
    if (other) sendTo(other, { type: 'peer_left' });
    if (!room.host && !room.guest) destroyRoom(roomCode);
  }, RECONNECT_GRACE);

  log(`${role} disconnected from ${roomCode} (grace period started)`);
}

/* ─── Helpers ────────────────────────────────────────────────── */

/**
 * Relay a message from sender to the other player in the room.
 * @param {WebSocket} sender
 * @param {string}    roomCode
 * @param {Object}    message
 */
function relay(sender, roomCode, message) {
  const room = rooms.get(roomCode);
  if (!room) return;
  touchRoom(room);
  const target = getOther(room, sender.role);
  if (target) sendTo(target, message);
}

/**
 * Send JSON to a single WebSocket.
 * @param {WebSocket} ws
 * @param {Object}    data
 */
function sendTo(ws, data) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify(data));
  }
}

/**
 * Return the other player's socket for a given role.
 * @param {Room}   room
 * @param {string} role
 * @returns {WebSocket|null}
 */
function getOther(room, role) {
  const other = role === 'host' ? room.guest : room.host;
  return other && other.readyState === WebSocket.OPEN ? other : null;
}

/**
 * Check whether both players have an open WebSocket.
 * @param {Room} room
 * @returns {boolean}
 */
function bothConnected(room) {
  return room.host?.readyState === WebSocket.OPEN &&
         room.guest?.readyState === WebSocket.OPEN;
}

/**
 * Refresh the room's activity timestamp and expiry timer.
 * @param {Room} room
 */
function touchRoom(room) {
  room.lastActivity = Date.now();
}

/**
 * Tear down a room and clear all associated timers.
 * @param {string} roomCode
 */
function destroyRoom(roomCode) {
  const room = rooms.get(roomCode);
  if (!room) return;
  clearTimeout(room.expiryTimer);
  clearTimeout(room.hostTimer);
  clearTimeout(room.guestTimer);
  rooms.delete(roomCode);
  log(`Room ${roomCode} destroyed`);
}

/**
 * Validate a room code (6 alphanumeric characters).
 * @param {string} code
 * @returns {boolean}
 */
function isValidCode(code) {
  return typeof code === 'string' && /^[A-Z0-9]{6}$/i.test(code);
}

/** Simple timestamped logger. */
function log(msg) {
  const ts = new Date().toISOString().slice(11, 19);
  console.log(`[${ts}] ${msg}`);
}

/* ─── Periodic room cleanup ──────────────────────────────────── */

setInterval(() => {
  const now = Date.now();
  for (const [code, room] of rooms) {
    if (now - room.lastActivity > ROOM_TIMEOUT_MS) {
      destroyRoom(code);
    }
  }
}, CLEANUP_INTERVAL);
