/**
 * @fileoverview WebRTC peer connection manager for Briscola P2P.
 *
 * Manages the full WebRTC lifecycle:
 * - RTCPeerConnection creation with STUN servers
 * - DataChannel ("briscola-game", ordered, reliable)
 * - SDP offer/answer exchange via SignalingClient
 * - ICE candidate trickling
 * - Connection-state monitoring + auto-reconnect (up to 3 attempts)
 * - Message send/receive via the DataChannel
 *
 * @module webrtc
 */

import { SignalingClient } from './signaling.js';

/**
 * @typedef {Object} GameMessage
 * @property {string} type  Message type identifier.
 */

/** ICE server configuration (public Google STUN). */
const ICE_CONFIG = {
  iceServers: [
    { urls: 'stun:stun.l.google.com:19302' },
    { urls: 'stun:stun1.l.google.com:19302' },
  ],
};

const MAX_RECONNECT = 3;
const DC_LABEL = 'briscola-game';

export class BriscolaWebRTC {
  /**
   * @param {string}   signalingUrl       WebSocket URL of signaling server.
   * @param {function} onMessage          Called with each parsed GameMessage.
   * @param {function} onConnectionChange Called with state string.
   */
  constructor(signalingUrl, onMessage, onConnectionChange) {
    /** @type {SignalingClient} */
    this.signaling = new SignalingClient(signalingUrl);

    /** @type {function(GameMessage): void} */
    this._onMessage = onMessage;

    /** @type {function(string): void} */
    this._onConnectionChange = onConnectionChange;

    /** @type {RTCPeerConnection|null} */
    this._pc = null;

    /** @type {RTCDataChannel|null} */
    this._dc = null;

    /** @type {string|null} */
    this.roomCode = null;

    /** @type {"host"|"guest"|null} */
    this.role = null;

    /** @type {GameMessage[]} Messages queued while DC is not yet open. */
    this._outQueue = [];

    /** @type {number} */
    this._reconnectAttempts = 0;

    /** @type {string} */
    this._state = 'disconnected';
  }

  /* ═══════════════════════════════════════════════════════════
     Public API
     ═══════════════════════════════════════════════════════════ */

  /**
   * Create a room and wait for the guest.
   * Used on the **landing page** (index.html).
   * @param {string} roomCode
   * @returns {Promise<void>}
   */
  async createRoom(roomCode) {
    this.roomCode = roomCode;
    this.role = 'host';
    this._setState('connecting');

    await this.signaling.connect();

    this.signaling.on('guest_joined', () => {
      this._onConnectionChange('guest_joined');
    });

    this.signaling.on('error', (d) => {
      this._onConnectionChange(`error:${d.message}`);
    });

    this.signaling.send({ type: 'create_room', roomCode });
  }

  /**
   * Join an existing room.
   * Used on the **landing page** (index.html).
   * @param {string} roomCode
   * @returns {Promise<void>}
   */
  async joinRoom(roomCode) {
    this.roomCode = roomCode;
    this.role = 'guest';
    this._setState('connecting');

    await this.signaling.connect();

    this.signaling.on('error', (d) => {
      this._onConnectionChange(`error:${d.message}`);
    });

    this.signaling.send({ type: 'join_room', roomCode });
  }

  /**
   * Re-join a room on the **game page** and set up the full
   * WebRTC DataChannel connection.
   * @param {string}          roomCode
   * @param {"host"|"guest"}  role
   * @returns {Promise<void>}
   */
  async rejoinAndConnect(roomCode, role) {
    this.roomCode = roomCode;
    this.role = role;
    this._setState('connecting');

    await this.signaling.connect();

    /* Wire up signaling message handlers */
    this.signaling.on('peer_ready', () => {
      if (this.role === 'host') this._startWebRTC();
    });

    this.signaling.on('offer', async (data) => {
      if (this.role === 'guest') {
        await this._handleOffer(data.sdp);
      }
    });

    this.signaling.on('answer', async (data) => {
      try { await this._pc?.setRemoteDescription(data.sdp); }
      catch (e) { console.error('setRemoteDescription(answer):', e); }
    });

    this.signaling.on('ice', async (data) => {
      try { await this._pc?.addIceCandidate(data.candidate); }
      catch (e) { console.error('addIceCandidate:', e); }
    });

    this.signaling.on('peer_left', () => {
      this._setState('disconnected');
    });

    this.signaling.on('error', (d) => {
      this._onConnectionChange(`error:${d.message}`);
    });

    /* Ask the signaling server to put us back in the room */
    this.signaling.send({ type: 'rejoin_room', roomCode, role });
  }

  /**
   * Send a game message to the peer via the DataChannel.
   * If the channel is not yet open, the message is queued.
   * @param {GameMessage} msg
   */
  sendMessage(msg) {
    const json = JSON.stringify(msg);
    if (this._dc?.readyState === 'open') {
      this._dc.send(json);
    } else {
      this._outQueue.push(json);
    }
  }

  /**
   * Cleanly shut down everything.
   */
  disconnect() {
    this._reconnectAttempts = MAX_RECONNECT; // prevent auto-reconnect
    this._dc?.close();
    this._pc?.close();
    this.signaling.send({ type: 'leave', roomCode: this.roomCode });
    this.signaling.disconnect();
    this._setState('closed');
  }

  /* ═══════════════════════════════════════════════════════════
     WebRTC setup (host initiates)
     ═══════════════════════════════════════════════════════════ */

  /** Host creates the PeerConnection, DataChannel, and SDP offer. */
  _startWebRTC() {
    this._createPC();

    // Host creates the DataChannel
    this._dc = this._pc.createDataChannel(DC_LABEL, {
      ordered: true,
    });
    this._wireDataChannel(this._dc);

    // Create and send SDP offer
    this._pc.createOffer()
      .then((offer) => this._pc.setLocalDescription(offer))
      .then(() => {
        this.signaling.send({
          type: 'offer',
          roomCode: this.roomCode,
          sdp: this._pc.localDescription,
        });
      })
      .catch((err) => console.error('createOffer:', err));
  }

  /**
   * Guest handles an incoming SDP offer.
   * @param {RTCSessionDescriptionInit} sdp
   */
  async _handleOffer(sdp) {
    this._createPC();

    // Guest listens for the DataChannel created by host
    this._pc.addEventListener('datachannel', (e) => {
      this._dc = e.channel;
      this._wireDataChannel(this._dc);
    });

    try {
      await this._pc.setRemoteDescription(sdp);
      const answer = await this._pc.createAnswer();
      await this._pc.setLocalDescription(answer);
      this.signaling.send({
        type: 'answer',
        roomCode: this.roomCode,
        sdp: this._pc.localDescription,
      });
    } catch (err) {
      console.error('handleOffer:', err);
    }
  }

  /** Create (or re-create) the RTCPeerConnection. */
  _createPC() {
    if (this._pc) {
      this._pc.close();
    }

    this._pc = new RTCPeerConnection(ICE_CONFIG);

    this._pc.addEventListener('icecandidate', (e) => {
      if (e.candidate) {
        this.signaling.send({
          type: 'ice',
          roomCode: this.roomCode,
          candidate: e.candidate,
        });
      }
    });

    this._pc.addEventListener('connectionstatechange', () => {
      const s = this._pc?.connectionState ?? 'closed';
      switch (s) {
        case 'connected':
          this._reconnectAttempts = 0;
          this._setState('connected');
          break;
        case 'disconnected':
        case 'failed':
          this._handlePeerDisconnect();
          break;
        case 'closed':
          this._setState('closed');
          break;
      }
    });
  }

  /* ═══════════════════════════════════════════════════════════
     DataChannel wiring
     ═══════════════════════════════════════════════════════════ */

  /**
   * Attach event listeners to the DataChannel.
   * @param {RTCDataChannel} dc
   */
  _wireDataChannel(dc) {
    dc.addEventListener('open', () => {
      this._setState('connected');
      this._flushQueue();
    });

    dc.addEventListener('message', (event) => {
      try {
        const msg = JSON.parse(event.data);
        this._onMessage(msg);
      } catch { /* ignore malformed */ }
    });

    dc.addEventListener('close', () => {
      this._handlePeerDisconnect();
    });

    dc.addEventListener('error', (e) => {
      console.error('DataChannel error:', e);
    });
  }

  /** Send all queued messages. */
  _flushQueue() {
    while (this._outQueue.length > 0) {
      const json = this._outQueue.shift();
      if (this._dc?.readyState === 'open') {
        this._dc.send(json);
      }
    }
  }

  /* ═══════════════════════════════════════════════════════════
     Connection-state + auto-reconnect
     ═══════════════════════════════════════════════════════════ */

  /**
   * @param {string} s
   */
  _setState(s) {
    if (this._state === s) return;
    this._state = s;
    this._onConnectionChange(s);
  }

  /** Called when the peer connection drops unexpectedly. */
  _handlePeerDisconnect() {
    if (this._reconnectAttempts >= MAX_RECONNECT) {
      this._setState('failed');
      return;
    }
    this._reconnectAttempts++;
    this._setState('disconnected');
    console.log(`Reconnect attempt ${this._reconnectAttempts}/${MAX_RECONNECT}`);

    // Re-request the signaling server to trigger a new WebRTC handshake
    setTimeout(() => {
      if (this.role === 'host') {
        this._startWebRTC();
      } else {
        // Guest asks the server to nudge host into re-offering
        this.signaling.send({ type: 'rejoin_room', roomCode: this.roomCode, role: this.role });
      }
    }, 2000 * this._reconnectAttempts);
  }
}
