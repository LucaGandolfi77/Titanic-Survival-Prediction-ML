// ============================================================
// routes.js — Endpoint REST esposti al MCP server Python
// ============================================================

import { Router } from "express";
import { sendCommand, isConnected } from "./websocket.js";

const router = Router();

/**
 * POST /set_var
 * Body: { name: string, value: string|number|boolean }
 * Imposta una variabile Storyline.
 */
router.post("/set_var", async (req, res) => {
  try {
    const { name, value } = req.body;
    if (!name) {
      return res.status(400).json({ error: "Parametro 'name' mancante" });
    }
    const result = await sendCommand({ cmd: "setVar", name, value });
    res.json(result);
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

/**
 * GET /get_var?name=<variabile>
 * Legge il valore di una variabile Storyline.
 */
router.get("/get_var", async (req, res) => {
  try {
    const { name } = req.query;
    if (!name) {
      return res.status(400).json({ error: "Parametro 'name' mancante" });
    }
    const result = await sendCommand({ cmd: "getVar", name });
    res.json(result);
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

/**
 * POST /jump_slide
 * Body: { slideIndex: number } — indice 1-based
 * Naviga a una slide specifica.
 */
router.post("/jump_slide", async (req, res) => {
  try {
    const { slideIndex } = req.body;
    if (slideIndex == null || slideIndex < 1) {
      return res
        .status(400)
        .json({ error: "Parametro 'slideIndex' mancante o non valido" });
    }
    const result = await sendCommand({ cmd: "jumpSlide", slideIndex });
    res.json(result);
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

/**
 * POST /animate
 * Body: { objectName: string, props: object }
 * Anima un oggetto Storyline via GSAP.
 */
router.post("/animate", async (req, res) => {
  try {
    const { objectName, props } = req.body;
    if (!objectName) {
      return res
        .status(400)
        .json({ error: "Parametro 'objectName' mancante" });
    }
    const result = await sendCommand({ cmd: "animate", objectName, props });
    res.json(result);
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

/**
 * POST /next_slide
 * Avanza alla slide successiva.
 */
router.post("/next_slide", async (req, res) => {
  try {
    const result = await sendCommand({ cmd: "nextSlide" });
    res.json(result);
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

/**
 * POST /prev_slide
 * Torna alla slide precedente.
 */
router.post("/prev_slide", async (req, res) => {
  try {
    const result = await sendCommand({ cmd: "prevSlide" });
    res.json(result);
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

/**
 * POST /execute_js
 * Body: { code: string }
 * Esegue codice JavaScript arbitrario nel contesto di Storyline.
 * ATTENZIONE: usare con cautela, solo in sviluppo locale.
 */
router.post("/execute_js", async (req, res) => {
  try {
    const { code } = req.body;
    if (!code) {
      return res.status(400).json({ error: "Parametro 'code' mancante" });
    }
    const result = await sendCommand({ cmd: "executeJs", code });
    res.json(result);
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

/**
 * GET /status
 * Verifica se Storyline è connesso.
 */
router.get("/status", (_req, res) => {
  res.json({ connected: isConnected() });
});

/**
 * GET /health
 * Health check del bridge server.
 */
router.get("/health", (_req, res) => {
  res.json({
    status: "ok",
    storylineConnected: isConnected(),
    uptime: process.uptime(),
  });
});

export default router;
