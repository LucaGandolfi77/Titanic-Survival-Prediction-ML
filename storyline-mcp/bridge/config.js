// ============================================================
// config.js — Configurazione centralizzata del bridge server
// ============================================================

/** Porta HTTP + WebSocket del bridge */
export const PORT = parseInt(process.env.BRIDGE_PORT || "8765", 10);

/** Host di ascolto */
export const HOST = process.env.BRIDGE_HOST || "localhost";

/** Timeout risposta WebSocket (ms) */
export const WS_TIMEOUT = parseInt(process.env.WS_TIMEOUT || "5000", 10);

/** Origini CORS consentite (stringa o array) */
export const CORS_ORIGIN = process.env.CORS_ORIGIN || "*";
