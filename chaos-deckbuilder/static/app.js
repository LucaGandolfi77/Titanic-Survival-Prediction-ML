const root = document.getElementById("game-root");
if (!root) throw new Error("Missing game root.");

const socket = io();
const room = root.dataset.room;
const username = root.dataset.username;

let state = null;
let selected = [];
let lastLogLine = "";
let showTutorial = !localStorage.getItem("chaos_tutorial_seen");

// === A1: Web Audio API Sound Synthesis ===
let audioCtx = null;
function getAudioCtx() {
  if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  return audioCtx;
}

function synthBuy() {
  try {
    const ctx = getAudioCtx();
    if (ctx.state === "suspended") ctx.resume();
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = "triangle";
    osc.frequency.setValueAtTime(600, ctx.currentTime);
    osc.frequency.exponentialRampToValueAtTime(900, ctx.currentTime + 0.08);
    gain.gain.setValueAtTime(0.15, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.15);
    osc.connect(gain).connect(ctx.destination);
    osc.start(); osc.stop(ctx.currentTime + 0.15);
  } catch(e) {}
}

function synthLock() {
  try {
    const ctx = getAudioCtx();
    if (ctx.state === "suspended") ctx.resume();
    [523, 659, 784].forEach((freq, i) => {
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = "sine";
      osc.frequency.value = freq;
      gain.gain.setValueAtTime(0.12, ctx.currentTime + i * 0.06);
      gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + i * 0.06 + 0.2);
      osc.connect(gain).connect(ctx.destination);
      osc.start(ctx.currentTime + i * 0.06);
      osc.stop(ctx.currentTime + i * 0.06 + 0.2);
    });
  } catch(e) {}
}

function synthHit() {
  try {
    const ctx = getAudioCtx();
    if (ctx.state === "suspended") ctx.resume();
    const bufferSize = ctx.sampleRate * 0.12;
    const buffer = ctx.createBuffer(1, bufferSize, ctx.sampleRate);
    const data = buffer.getChannelData(0);
    for (let i = 0; i < bufferSize; i++) {
      data[i] = (Math.random() * 2 - 1) * (1 - i / bufferSize);
    }
    const src = ctx.createBufferSource();
    const filter = ctx.createBiquadFilter();
    const gain = ctx.createGain();
    src.buffer = buffer;
    filter.type = "lowpass";
    filter.frequency.value = 1200;
    gain.gain.setValueAtTime(0.2, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.12);
    src.connect(filter).connect(gain).connect(ctx.destination);
    src.start();
  } catch(e) {}
}

function synthWin() {
  try {
    const ctx = getAudioCtx();
    if (ctx.state === "suspended") ctx.resume();
    [523, 659, 784, 1047].forEach((freq, i) => {
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = "sine";
      osc.frequency.value = freq;
      gain.gain.setValueAtTime(0.15, ctx.currentTime + i * 0.12);
      gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + i * 0.12 + 0.4);
      osc.connect(gain).connect(ctx.destination);
      osc.start(ctx.currentTime + i * 0.12);
      osc.stop(ctx.currentTime + i * 0.12 + 0.4);
    });
  } catch(e) {}
}

function synthRare() {
  try {
    const ctx = getAudioCtx();
    if (ctx.state === "suspended") ctx.resume();
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = "sine";
    osc.frequency.setValueAtTime(1200, ctx.currentTime);
    osc.frequency.exponentialRampToValueAtTime(2400, ctx.currentTime + 0.15);
    osc.frequency.exponentialRampToValueAtTime(1800, ctx.currentTime + 0.3);
    gain.gain.setValueAtTime(0.12, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.35);
    const vibrato = ctx.createOscillator();
    const vibratoGain = ctx.createGain();
    vibrato.frequency.value = 6;
    vibratoGain.gain.value = 20;
    vibrato.connect(vibratoGain).connect(osc.frequency);
    vibrato.start(); osc.connect(gain).connect(ctx.destination);
    osc.start(); osc.stop(ctx.currentTime + 0.35);
    vibrato.stop(ctx.currentTime + 0.35);
  } catch(e) {}
}

const sfx = { buy: synthBuy, lock: synthLock, hit: synthHit, win: synthWin, rare: synthRare };
function playSound(name) { if (sfx[name]) sfx[name](); }

const els = {
  dailyName: document.getElementById("daily-name"),
  youStats: document.getElementById("you-stats"),
  opponents: document.getElementById("opponents"),
  missionBox: document.getElementById("mission-box"),
  achievements: document.getElementById("achievements"),
  market: document.getElementById("market"),
  hand: document.getElementById("hand"),
  battleLog: document.getElementById("battle-log"),
  submitBtn: document.getElementById("submit-btn"),
  phaseBadge: document.getElementById("phase-badge"),
  tablesOptions: document.getElementById("tables-options"),
  backsOptions: document.getElementById("backs-options"),
  particlesOptions: document.getElementById("particles-options"),
};

function cardHTML(card, showBuy = false) {
  const rarityGlow = ["rare","epic","legendary"].includes(card.rarity)
    ? `box-shadow:0 0 16px rgba(139,92,246,${card.rarity==="legendary"?.35:card.rarity==="epic"?.25:.18});`
    : "";
  return `
    <div class="game-card reveal-card" data-card-id="${card.id}" data-rarity="${card.rarity || "common"}" style="${rarityGlow}">
      <div class="meta">
        <span class="cost">Cost: ${card.effective_cost ?? card.cost}</span>
        <span class="rarity rarity-${card.rarity || "common"}">${card.rarity || "common"}</span>
      </div>
      <div>
        <h3>${card.name}</h3>
        <div class="tribe-line">${card.tribe || "neutral"} tribe</div>
        <div class="text">${card.text}</div>
      </div>
      <div class="meta">
        <span>&#9876; ${card.power}</span>
        <span>&#128737; ${card.shield}</span>
        <span>&#10084; ${card.heal}</span>
      </div>
      <div class="meta">
        <span>&#127911; +${card.draw}</span>
        <span>&#127853; +${card.gold}</span>
        <span>&#9760; ${card.poison}</span>
      </div>
      ${showBuy ? `<button class="buy-btn" data-buy-id="${card.id}">Buy</button>` : ``}
    </div>
  `;
}

function getDeckArchetype(hand) {
  if (!hand || !hand.length) return "";
  const tribeCounts = {};
  hand.forEach(c => { const t = c.tribe || "neutral"; tribeCounts[t] = (tribeCounts[t]||0)+1; });
  const max = Math.max(...Object.values(tribeCounts), 0);
  if (max >= 3) {
    const top = Object.entries(tribeCounts).sort((a,b)=>b[1]-a[1])[0];
    return `${top[0].toUpperCase()} ENGINE`;
  }
  if (tribeCounts.cursed >= 2) return "CURSED CHAOS";
  if (tribeCounts.machine >= 2) return "TECH ECONOMY";
  if (tribeCounts.ocean >= 2) return "TURTLE DEFENSE";
  if (tribeCounts.beast >= 2) return "BEAST AGGRO";
  if (tribeCounts.warrior >= 2) return "WARFARE";
  if (tribeCounts.cult >= 2) return "CULT RITUAL";
  return "";
}

function renderStats() {
  const you = state.you;
  const archetype = getDeckArchetype(you.hand);
  let html = `
    <div class="stat"><strong>${you.username}</strong></div>
    <div class="stat">HP: ${you.hp}</div>
    <div class="stat">Coins: ${you.coins}</div>
    <div class="stat">Deck: ${you.deck_count}</div>
    <div class="stat">Discard: ${you.discard_count}</div>
    <div class="stat">Players in room: ${state.players_in_room}/2</div>
  `;
  if (state.phase === "game_over" && archetype) {
    html += `<div class="stat archetype-badge">Archetype: ${archetype}</div>`;
  }
  if (you.is_spectator) {
    html += `<div class="stat">SPECTATOR MODE</div>`;
  }
  els.youStats.innerHTML = html;

  els.opponents.innerHTML = state.opponents.length
    ? state.opponents.map(op => `
      <div class="mini-panel">
        <strong>${op.username}</strong><br>
        HP: ${op.hp}<br>
        Coins: ${op.coins}<br>
        Hand: ${op.hand_count}<br>
        Achievements: ${op.achievement_count}
      </div>
    `).join("")
    : `<div class="mini-panel">Waiting for another player...</div>`;
}

function renderMission() {
  const m = state.you.mission;
  const pct = Math.min(100, Math.round((m.progress / m.goal) * 100));
  els.missionBox.innerHTML = `
    <div class="mission">
      <strong>${m.name}</strong>
      <p>${m.desc}</p>
      <div class="mission-bar"><div class="mission-fill" style="width:${pct}%"></div></div>
      <p>Progress: ${m.progress}/${m.goal} (${pct}%)</p>
      <p>Status: ${m.done ? "Completed &#9989;" : "Hidden but active &#128064;"}</p>
    </div>
  `;
}

function renderAchievements() {
  if (!state.you.achievements.length) {
    els.achievements.innerHTML = `<div class="achievement">No ridiculous achievements yet.</div>`;
    return;
  }
  els.achievements.innerHTML = state.you.achievements.map(a => `
    <div class="achievement">
      <strong>${a.name}</strong>
      <div>${a.desc}</div>
    </div>
  `).join("");
}

function renderMarket() {
  els.market.innerHTML = state.market.map((card, index) => `
    <div>
      ${cardHTML(card, true)}
      <div style="margin-top:8px;">
        <button class="buy-btn" data-buy-index="${index}" ${state.you.bought || state.phase === "game_over" || state.you.is_spectator ? "disabled" : ""}>
          ${state.you.bought ? "Bought this round" : `Buy for ${card.effective_cost}`}
        </button>
      </div>
    </div>
  `).join("");

  els.market.querySelectorAll("[data-buy-index]").forEach(btn => {
    btn.addEventListener("click", () => {
      playSound("buy");
      const r = state.market[Number(btn.dataset.buyIndex)].rarity;
      if (["rare", "epic", "legendary"].includes(r)) playSound("rare");
      socket.emit("buy_card", { room, index: Number(btn.dataset.buyIndex) });
    });
  });
}

function renderHand() {
  els.hand.innerHTML = state.you.hand.map(card => `
    <div class="game-card reveal-card ${selected.includes(card.id) ? "selected" : ""}" data-rarity="${card.rarity || "common"}" data-card-id="${card.id}" data-select-id="${card.id}">
      <div class="meta">
        <span>${card.color}</span>
        <span>${selected.includes(card.id) ? "Selected" : "Ready"}</span>
      </div>
      <div>
        <h3>${card.name}</h3>
        <div class="tribe-line">${card.tribe || "neutral"} tribe</div>
        <div class="text">${card.text}</div>
      </div>
      <div class="meta">
        <span>&#9876; ${card.power}</span>
        <span>&#128737; ${card.shield}</span>
        <span>&#10084; ${card.heal}</span>
      </div>
      <div class="meta">
        <span>&#127911; +${card.draw}</span>
        <span>&#127853; +${card.gold}</span>
        <span>&#9760; ${card.poison}</span>
      </div>
    </div>
  `).join("");

  els.hand.querySelectorAll("[data-select-id]").forEach(node => {
    node.addEventListener("click", () => {
      if (state.you.is_spectator) return;
      const id = node.dataset.selectId;
      if (selected.includes(id)) {
        selected = selected.filter(x => x !== id);
      } else if (selected.length < 3) {
        selected.push(id);
      }
      renderHand();
    });
  });
}

function renderLog() {
  els.battleLog.innerHTML = state.log.map(line => `<div class="log-line">${line}</div>`).join("");
  els.battleLog.scrollTop = els.battleLog.scrollHeight;
}

function renderCosmetics() {
  const catalog = state.cosmetics_catalog;
  const active = state.you.cosmetics;

  function pills(values, category, activeValue) {
    return values.map(value => `
      <button class="pill ${activeValue === value ? "active" : ""}" data-category="${category}" data-value="${value}" ${state.you.is_spectator ? "disabled" : ""}>
        ${value}
      </button>
    `).join("");
  }

  els.tablesOptions.innerHTML = pills(catalog.tables, "tables", active.table);
  els.backsOptions.innerHTML = pills(catalog.backs, "backs", active.back);
  els.particlesOptions.innerHTML = pills(catalog.particles, "particles", active.particles);

  document.querySelectorAll(".pill").forEach(btn => {
    if (btn.dataset.emote) return;
    btn.addEventListener("click", () => {
      socket.emit("set_cosmetic", {
        room,
        category: btn.dataset.category,
        value: btn.dataset.value
      });
    });
  });

  document.body.classList.remove(
    "theme-neon-lagoon",
    "theme-velvet-casino",
    "theme-moon-parlor",
    "theme-gold-doom"
  );
  document.body.classList.add(`theme-${active.table}`);
}

function renderPostGameStats() {
  const history = state.round_history || [];
  if (!history.length || state.phase !== "game_over") {
    const existing = document.getElementById("postgame-stats-container");
    if (existing) existing.remove();
    return;
  }

  let chartHTML = '<div class="postgame-stats"><h3>Round-by-Round Stats</h3><div class="stats-chart">';
  history.forEach(r => {
    chartHTML += `<div class="stat-bar-group">
      <div class="stat-bar-label">R${r.round}</div>
      <div class="stat-bar" style="height:${Math.min(100, r.p1_damage*10)}px;background:var(--danger)"></div>
      <div class="stat-bar" style="height:${Math.min(100, r.p1_shield*8)}px;background:var(--good)"></div>
      <div class="stat-bar" style="height:${Math.min(100, r.p1_heal*10)}px;background:var(--accent2)"></div>
    </div>`;
  });
  chartHTML += '</div></div>';

  let container = document.getElementById("postgame-stats-container");
  if (!container) {
    container = document.createElement("div");
    container.id = "postgame-stats-container";
    const centerCol = document.querySelector(".center-col");
    if (centerCol) centerCol.appendChild(container);
  }
  container.innerHTML = chartHTML;
}

function closeTutorial() {
  showTutorial = false;
  localStorage.setItem("chaos_tutorial_seen", "1");
  const modal = document.getElementById("tutorial-modal");
  if (modal) modal.classList.add("hidden");
}

function sendEmote(emote) {
  if (state.you.is_spectator) return;
  socket.emit("send_emote", { room, emote });
}

els.submitBtn.addEventListener("click", () => {
  if (!selected.length || state.you.is_spectator) return;
  playSound("lock");
  socket.emit("submit_cards", { room, cards: selected });
  selected = [];
});

document.addEventListener("click", (e) => {
  const closeBtn = e.target.closest("[data-close-tutorial]");
  if (closeBtn) closeTutorial();
});

document.querySelectorAll("[data-emote]").forEach(btn => {
  btn.addEventListener("click", () => {
    sendEmote(btn.dataset.emote);
  });
});

socket.on("connect", () => {
  socket.emit("join_game", { room, username, spectator: false });
});

socket.on("emote_received", (data) => {
  const el = document.createElement("div");
  el.className = "emote-bubble";
  el.textContent = `${data.username}: ${data.emote}`;
  document.body.appendChild(el);
  setTimeout(() => el.remove(), 3000);
});

socket.on("trade_response", (data) => {
  if (!data.success) { alert(data.message); return; }
  playSound("rare");
  const el = document.createElement("div");
  el.className = "emote-bubble";
  el.textContent = `Traded ${data.card.name} from ${data.from}!`;
  document.body.appendChild(el);
  setTimeout(() => el.remove(), 3000);
});

socket.on("state", (nextState) => {
  const newest = nextState.log?.[nextState.log.length - 1] || "";
  if (newest && newest !== lastLogLine) {
    if (newest.includes("wins the game")) { playSound("win"); synthWin(); }
    else if (newest.includes("dealt")) { playSound("hit"); synthHit(); }
    else if (newest.includes(": GG!") || newest.includes(": Wow!") || newest.includes(": Sick!")) { playSound("rare"); synthRare(); }
    lastLogLine = newest;
  }

  state = nextState;
  selected = selected.filter(id => state.you.hand.some(c => c.id === id));
  renderAll();

  if (showTutorial && state.phase === "shop" && state.round === 1 && !state.you.is_spectator) {
    setTimeout(() => {
      const modal = document.getElementById("tutorial-modal");
      if (modal) modal.classList.remove("hidden");
    }, 500);
  }
});

socket.on("error_message", (data) => {
  alert(data.message);
});

function renderAll() {
  if (!state) return;
  els.dailyName.textContent = `${state.daily.name}`;
  els.phaseBadge.textContent = `${state.phase.toUpperCase()} &bull; Round ${state.round}`;
  renderStats();
  renderMission();
  renderAchievements();
  renderMarket();
  renderHand();
  renderLog();
  renderCosmetics();
  renderPostGameStats();
}
