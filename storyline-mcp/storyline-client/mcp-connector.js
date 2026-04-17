// ============================================================
// mcp-connector.js — Codice da inserire in Storyline 360
// ============================================================
// ISTRUZIONI:
// 1. In Storyline 360, vai sulla prima slide (o slide "init")
// 2. Aggiungi trigger: "Execute JavaScript" su "Timeline starts"
// 3. Copia-incolla TUTTO il contenuto di questo file nel trigger
//
// REQUISITI:
// - Il bridge server deve essere attivo (node bridge/server.js)
// - Il corso deve essere aperto via http://localhost (NON file://)
// - Funziona solo in locale, non su Review 360 o LMS remoti
// ============================================================

(function () {
  // Evita inizializzazione doppia
  if (window._slMcpInitialized) return;
  window._slMcpInitialized = true;

  var player = GetPlayer();
  var ws;
  var reconnectDelay = 3000;
  var wsUrl = "ws://localhost:8765";

  function connect() {
    ws = new WebSocket(wsUrl);

    ws.onopen = function () {
      console.log("[SL-MCP] Connesso al bridge");
      try {
        player.SetVar("mcp_status", "connected");
      } catch (e) {
        /* variabile non definita — ok */
      }
    };

    ws.onclose = function () {
      console.log("[SL-MCP] Disconnesso, riconnessione in " + (reconnectDelay / 1000) + "s...");
      try {
        player.SetVar("mcp_status", "disconnected");
      } catch (e) {
        /* variabile non definita — ok */
      }
      setTimeout(connect, reconnectDelay);
    };

    ws.onerror = function (e) {
      console.error("[SL-MCP] Errore WebSocket:", e);
    };

    ws.onmessage = function (event) {
      var msg;
      try {
        msg = JSON.parse(event.data);
      } catch (e) {
        console.error("[SL-MCP] JSON non valido:", event.data);
        return;
      }

      var response = { id: msg.id, status: "ok" };

      try {
        switch (msg.cmd) {
          case "setVar":
            player.SetVar(msg.name, msg.value);
            break;

          case "getVar":
            response.value = player.GetVar(msg.name);
            break;

          case "jumpSlide":
            handleJumpSlide(msg, response);
            break;

          case "nextSlide":
            handleNextSlide(response);
            break;

          case "prevSlide":
            handlePrevSlide(response);
            break;

          case "animate":
            handleAnimate(msg, response);
            break;

          case "executeJs":
            handleExecuteJs(msg, response);
            break;

          default:
            response.status = "error";
            response.error = "Comando sconosciuto: " + msg.cmd;
        }
      } catch (e) {
        response.status = "error";
        response.error = e.message || String(e);
      }

      ws.send(JSON.stringify(response));
    };
  }

  // --- Handler per i comandi ---

  function handleJumpSlide(msg, response) {
    // Storyline 360 JS API avanzata (Build 3.98+)
    if (typeof player.slides === "function") {
      var slides = player.slides();
      if (slides && slides[msg.slideIndex - 1]) {
        slides[msg.slideIndex - 1].jumpTo();
      } else {
        response.status = "error";
        response.error = "Slide index non valido: " + msg.slideIndex;
      }
    } else {
      response.status = "error";
      response.error = "API slides() non disponibile — richiede Build 3.98+";
    }
  }

  function handleNextSlide(response) {
    if (typeof player.next === "function") {
      player.next();
    } else {
      response.status = "error";
      response.error = "API next() non disponibile";
    }
  }

  function handlePrevSlide(response) {
    if (typeof player.prev === "function") {
      player.prev();
    } else {
      response.status = "error";
      response.error = "API prev() non disponibile";
    }
  }

  function handleAnimate(msg, response) {
    var targetEl =
      document.querySelector('[data-acc-text="' + msg.objectName + '"]') ||
      document.getElementById(msg.objectName);

    if (targetEl && window.gsap) {
      gsap.to(targetEl, msg.props);
      response.status = "animated";
    } else if (!targetEl) {
      response.status = "error";
      response.error = "Oggetto non trovato: " + msg.objectName;
    } else {
      response.status = "error";
      response.error = "GSAP non disponibile";
    }
  }

  function handleExecuteJs(msg, response) {
    try {
      // Esegue il codice nel contesto globale
      var result = new Function("player", msg.code)(player);
      response.result = result !== undefined ? String(result) : "undefined";
    } catch (e) {
      response.status = "error";
      response.error = "Errore esecuzione JS: " + e.message;
    }
  }

  // Avvia la connessione
  connect();
})();
