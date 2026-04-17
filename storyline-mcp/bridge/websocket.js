// ============================================================
// websocket.js — Gestione connessione WebSocket con Storyline
// ============================================================

import { WebSocketServer } from "ws";

/** @type {import("ws").WebSocket | null} */
let storylineClient = null;

/** Map di richieste in attesa di risposta */
const pendingRequests = new Map();

/** Contatore auto-incrementante per gli ID richiesta */
let requestId = 0;

/**
 * Verifica se Storyline è attualmente connesso.
 * @returns {boolean}
 */
export function isConnected() {
  return storylineClient !== null && storylineClient.readyState === 1;
}

/**
 * Invia un comando a Storyline via WebSocket e attende la risposta.
 * @param {object} payload — Il comando da inviare (cmd, + parametri).
 * @param {number} [timeoutMs=5000] — Timeout in millisecondi.
 * @returns {Promise<object>} — La risposta di Storyline.
 */
export function sendCommand(payload, timeoutMs = 5000) {
  return new Promise((resolve, reject) => {
    if (!isConnected()) {
      return reject(
        new Error("Storyline non connesso. Apri il corso nel browser.")
      );
    }

    const id = ++requestId;
    payload.id = id;

    pendingRequests.set(id, resolve);

    // Timeout di sicurezza
    const timer = setTimeout(() => {
      if (pendingRequests.has(id)) {
        pendingRequests.delete(id);
        reject(new Error("Timeout: Storyline non ha risposto in 5 secondi"));
      }
    }, timeoutMs);

    // Cleanup timer quando la risposta arriva
    const originalResolve = resolve;
    pendingRequests.set(id, (msg) => {
      clearTimeout(timer);
      originalResolve(msg);
    });

    storylineClient.send(JSON.stringify(payload));
  });
}

/**
 * Inizializza il WebSocket server su un httpServer esistente.
 * @param {import("http").Server} httpServer
 */
export function initWebSocket(httpServer) {
  const wss = new WebSocketServer({
    server: httpServer,
    verifyClient: (info) => {
      // Accetta solo connessioni da localhost (sicurezza)
      const origin = info.origin || "";
      return (
        origin.includes("localhost") ||
        origin.includes("127.0.0.1") ||
        origin === ""
      );
    },
  });

  wss.on("connection", (ws) => {
    storylineClient = ws;
    console.log("[WS] Storyline 360 connesso via WebSocket");

    ws.on("message", (raw) => {
      try {
        const msg = JSON.parse(raw.toString());
        const resolve = pendingRequests.get(msg.id);
        if (resolve) {
          pendingRequests.delete(msg.id);
          resolve(msg);
        }
      } catch (err) {
        console.error("[WS] Errore parsing messaggio:", err.message);
      }
    });

    ws.on("close", () => {
      storylineClient = null;
      console.log("[WS] Storyline disconnesso");
    });

    ws.on("error", (err) => {
      console.error("[WS] Errore WebSocket:", err.message);
    });
  });

  return wss;
}
