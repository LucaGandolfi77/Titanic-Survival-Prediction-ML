const root = document.getElementById("game-root");
if (!root) throw new Error("Missing game root.");

const socket = io();
const room = root.dataset.room;
const username = root.dataset.username;

let state = null;
let selected = [];
let lastLogLine = "";

const sfx = {
  buy: document.getElementById("sfx-buy"),
  lock: document.getElementById("sfx-lock"),
  hit: document.getElementById("sfx-hit"),
  win: document.getElementById("sfx-win"),
  rare: document.getElementById("sfx-rare")
};

function playSound(name, volume = 0.7) {
  const el = sfx[name];
  if (!el) return;
  el.currentTime = 0;
  el.volume = volume;
  el.play().catch(() => {});
}

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
  return `
    <div class="game-card" data-card-id="${card.id}" data-rarity="${card.rarity || "common"}">
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
        <span>⚔ ${card.power}</span>
        <span>🛡 ${card.shield}</span>
        <span>❤ ${card.heal}</span>
      </div>
      <div class="meta">
        <span>🎴 +${card.draw}</span>
        <span>🪙 +${card.gold}</span>
        <span>☠ ${card.poison}</span>
      </div>
      ${showBuy ? `<button class="buy-btn" data-buy-id="${card.id}">Buy</button>` : ``}
    </div>
  `;
}

function renderStats() {
  const you = state.you;
  els.youStats.innerHTML = `
    <div class="stat"><strong>${you.username}</strong></div>
    <div class="stat">HP: ${you.hp}</div>
    <div class="stat">Coins: ${you.coins}</div>
    <div class="stat">Deck: ${you.deck_count}</div>
    <div class="stat">Discard: ${you.discard_count}</div>
    <div class="stat">Players in room: ${state.players_in_room}/2</div>
  `;

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
  els.missionBox.innerHTML = `
    <div class="mission">
      <strong>${m.name}</strong>
      <p>${m.desc}</p>
      <p>Progress: ${m.progress}/${m.goal}</p>
      <p>Status: ${m.done ? "Completed ✅" : "Hidden but active 👀"}</p>
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
        <button class="buy-btn" data-buy-index="${index}" ${state.you.bought || state.phase === "game_over" ? "disabled" : ""}>
          ${state.you.bought ? "Bought this round" : `Buy for ${card.effective_cost}`}
        </button>
      </div>
    </div>
  `).join("");

  els.market.querySelectorAll("[data-buy-index]").forEach(btn => {
    btn.addEventListener("click", () => {
      playSound("buy", 0.6);
      if (["rare", "epic", "legendary"].includes(state.market[Number(btn.dataset.buyIndex)].rarity)) {
        playSound("rare", 0.75);
      }
      socket.emit("buy_card", { room, index: Number(btn.dataset.buyIndex) });
    });
  });
}

function renderHand() {
  els.hand.innerHTML = state.you.hand.map(card => `
    <div class="game-card ${selected.includes(card.id) ? "selected" : ""}" data-rarity="${card.rarity || "common"}" data-card-id="${card.id}" data-select-id="${card.id}">
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
        <span>⚔ ${card.power}</span>
        <span>🛡 ${card.shield}</span>
        <span>❤ ${card.heal}</span>
      </div>
      <div class="meta">
        <span>🎴 +${card.draw}</span>
        <span>🪙 +${card.gold}</span>
        <span>☠ ${card.poison}</span>
      </div>
    </div>
  `).join("");

  els.hand.querySelectorAll("[data-select-id]").forEach(node => {
    node.addEventListener("click", () => {
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
}

function renderCosmetics() {
  const catalog = state.cosmetics_catalog;
  const active = state.you.cosmetics;

  function pills(values, category, activeValue) {
    return values.map(value => `
      <button class="pill ${activeValue === value ? "active" : ""}" data-category="${category}" data-value="${value}">
        ${value}
      </button>
    `).join("");
  }

  els.tablesOptions.innerHTML = pills(catalog.tables, "tables", active.table);
  els.backsOptions.innerHTML = pills(catalog.backs, "backs", active.back);
  els.particlesOptions.innerHTML = pills(catalog.particles, "particles", active.particles);

  document.querySelectorAll(".pill").forEach(btn => {
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

function renderAll() {
  if (!state) return;
  els.dailyName.textContent = `${state.daily.name}`;
  els.phaseBadge.textContent = `${state.phase.toUpperCase()} • Round ${state.round}`;
  renderStats();
  renderMission();
  renderAchievements();
  renderMarket();
  renderHand();
  renderLog();
  renderCosmetics();
}

els.submitBtn.addEventListener("click", () => {
  if (!selected.length) return;
  playSound("lock", 0.6);
  socket.emit("submit_cards", { room, cards: selected });
  selected = [];
});

socket.on("connect", () => {
  socket.emit("join_game", { room, username });
});

socket.on("state", (nextState) => {
  const newest = nextState.log?.[nextState.log.length - 1] || "";
  if (newest && newest !== lastLogLine) {
    if (newest.includes("wins the game")) playSound("win", 0.8);
    else if (newest.includes("dealt")) playSound("hit", 0.55);
    lastLogLine = newest;
  }

  state = nextState;
  selected = selected.filter(id => state.you.hand.some(c => c.id === id));
  renderAll();
});

socket.on("error_message", (data) => {
  alert(data.message);
});
