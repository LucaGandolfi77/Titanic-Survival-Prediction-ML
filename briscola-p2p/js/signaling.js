/**
 * @fileoverview WebSocket signaling client.
 *
 * Thin wrapper around the browser WebSocket API for communicating with
 * the Briscola signaling server.  Provides:
 *
 * - Automatic JSON serialisation / deserialisation
 * - Event-based message routing via `on(type, handler)`
 * - Connection-state callbacks (open / close / error)
 * - Message queuing while the socket is not yet open
 *
 * @module signaling
 */

/**
 * @callback MessageHandler
 * @param {Object} data  Parsed JSON message
 */

export class SignalingClient {
  /**
   * @param {string} url  WebSocket URL of the signaling server.
   */
  constructor(url) {
    /** @type {string} */
    this.url = url;

    /** @type {WebSocket|null} */
    this._ws = null;

    /** @type {Map<string, MessageHandler>} */
    this._handlers = new Map();

    /** @type {Object[]} Messages queued while the socket is opening. */
    this._queue = [];

    /** @type {boolean} */
    this._intentionallyClosed = false;
  }

  /* ────────────────────────────────────────────────────────── */
  /*  Connection                                                */
  /* ────────────────────────────────────────────────────────── */

  /**
   * Open the WebSocket connection.  Resolves when the socket is open.
   * @returns {Promise<void>}
   */
  connect() {
    return new Promise((resolve, reject) => {
      this._intentionallyClosed = false;
      this._ws = new WebSocket(this.url);

      this._ws.addEventListener('open', () => {
        this._flushQueue();
        this._emit('open', {});
        resolve();
      });

      this._ws.addEventListener('message', (event) => {
        try {
          const data = JSON.parse(event.data);
          this._emit(data.type, data);
        } catch { /* ignore malformed messages */ }
      });

      this._ws.addEventListener('close', () => {
        this._emit('close', {});
      });

      this._ws.addEventListener('error', (err) => {
        this._emit('error', { error: err });
        if (this._ws?.readyState !== WebSocket.OPEN) reject(err);
      });
    });
  }

  /**
   * Register a handler for a given message type or lifecycle event.
   *
   * Special types: `"open"`, `"close"`, `"error"`.
   *
   * @param {string}         type
   * @param {MessageHandler} handler
   */
  on(type, handler) {
    this._handlers.set(type, handler);
  }

  /**
   * Remove a registered handler.
   * @param {string} type
   */
  off(type) {
    this._handlers.delete(type);
  }

  /**
   * Send a JSON message to the server.
   * If the socket is still opening, the message is queued.
   * @param {Object} message
   */
  send(message) {
    if (this._ws?.readyState === WebSocket.OPEN) {
      this._ws.send(JSON.stringify(message));
    } else {
      this._queue.push(message);
    }
  }

  /**
   * Gracefully close the WebSocket.
   */
  disconnect() {
    this._intentionallyClosed = true;
    this._ws?.close();
    this._ws = null;
  }

  /** @returns {boolean} */
  get isOpen() {
    return this._ws?.readyState === WebSocket.OPEN;
  }

  /* ────────────────────────────────────────────────────────── */
  /*  Internal                                                  */
  /* ────────────────────────────────────────────────────────── */

  /**
   * Dispatch to the registered handler, if any.
   * @param {string} type
   * @param {Object} data
   */
  _emit(type, data) {
    const handler = this._handlers.get(type);
    if (handler) handler(data);
  }

  /** Flush any messages that were queued while the socket was opening. */
  _flushQueue() {
    while (this._queue.length > 0) {
      const msg = this._queue.shift();
      this._ws?.send(JSON.stringify(msg));
    }
  }
}
