// ============================================================
// server.js — Entry point del Bridge Server
// ============================================================
// Avvia con: node bridge/server.js
//
// Il bridge espone:
//   - REST API su http://localhost:8765 (per il MCP server Python)
//   - WebSocket su ws://localhost:8765  (per Storyline 360 nel browser)
// ============================================================

import express from "express";
import { createServer } from "http";
import cors from "cors";

import { PORT, HOST, CORS_ORIGIN } from "./config.js";
import { initWebSocket } from "./websocket.js";
import routes from "./routes.js";

// --- Express app ---
const app = express();
app.use(cors({ origin: CORS_ORIGIN }));
app.use(express.json());

// Monta le route REST
app.use("/", routes);

// --- HTTP + WebSocket server ---
const httpServer = createServer(app);
initWebSocket(httpServer);

// --- Avvio ---
httpServer.listen(PORT, HOST, () => {
  console.log(`[Bridge] REST  → http://${HOST}:${PORT}`);
  console.log(`[Bridge] WS    → ws://${HOST}:${PORT}`);
  console.log("[Bridge] In attesa della connessione Storyline 360...");
});
