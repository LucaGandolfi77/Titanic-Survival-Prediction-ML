const TIME_LIMIT = 8;
const FAST_THRESHOLD = 5.5;
const RESPONSE_DELAY = 950;

const profiles = {
  female: [
    {
      name: "Luna",
      meta: "Neon rooftop · Mischievous smile",
      avatar: "L",
      colors: ["#ff6ec7", "#7c4dff"]
    },
    {
      name: "Sofia",
      meta: "Vinyl bar · Warm energy",
      avatar: "S",
      colors: ["#ff9966", "#ff5e62"]
    },
    {
      name: "Maya",
      meta: "Arcade queen · Bold aura",
      avatar: "M",
      colors: ["#00c6ff", "#0072ff"]
    }
  ],
  male: [
    {
      name: "Noah",
      meta: "City lights · Effortless charm",
      avatar: "N",
      colors: ["#36d1dc", "#5b86e5"]
    },
    {
      name: "Leo",
      meta: "Afterparty DJ · Confident grin",
      avatar: "L",
      colors: ["#f7971e", "#ffd200"]
    },
    {
      name: "Adrian",
      meta: "Late-night café · Mystery energy",
      avatar: "A",
      colors: ["#7f00ff", "#e100ff"]
    }
  ]
};

const state = {
  playerGender: "male",
  interestGender: "female",
  character: null,
  scenes: [],
  currentSceneIndex: 0,
  score: 0,
  streak: 0,
  bestStreak: 0,
  secrets: 0,
  timer: null,
  timeLeft: TIME_LIMIT,
  awaitingChoice: false
};

const els = {
  setupScreen: document.getElementById("setup-screen"),
  gameScreen: document.getElementById("game-screen"),
  endScreen: document.getElementById("end-screen"),

  playerGender: document.getElementById("player-gender"),
  interestGender: document.getElementById("interest-gender"),
  startBtn: document.getElementById("start-btn"),
  restartBtn: document.getElementById("restart-btn"),
  playAgainBtn: document.getElementById("play-again-btn"),

  chapterLabel: document.getElementById("chapter-label"),
  characterName: document.getElementById("character-name"),
  characterMeta: document.getElementById("character-meta"),
  dialogueText: document.getElementById("dialogue-text"),
  avatar: document.getElementById("avatar"),

  timerText: document.getElementById("timer-text"),
  timerFill: document.getElementById("timer-fill"),

  scoreValue: document.getElementById("score-value"),
  streakValue: document.getElementById("streak-value"),
  secretValue: document.getElementById("secret-value"),

  choices: document.getElementById("choices"),

  endBadge: document.getElementById("end-badge"),
  endTitle: document.getElementById("end-title"),
  endText: document.getElementById("end-text"),
  finalScore: document.getElementById("final-score"),
  finalStreak: document.getElementById("final-streak"),
  finalSecret: document.getElementById("final-secret")
};

function pickRandom(arr) {
  return arr[Math.floor(Math.random() * arr.length)];
}

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function clearGameTimer() {
  if (state.timer) {
    clearInterval(state.timer);
    state.timer = null;
  }
}

function showScreen(screen) {
  els.setupScreen.classList.remove("active");
  els.gameScreen.classList.remove("active");
  els.endScreen.classList.remove("active");

  if (screen === "setup") els.setupScreen.classList.add("active");
  if (screen === "game") els.gameScreen.classList.add("active");
  if (screen === "end") els.endScreen.classList.add("active");
}

function updateStats() {
  els.scoreValue.textContent = state.score;
  els.streakValue.textContent = state.streak;
  els.secretValue.textContent = state.secrets;
}

function updateTimerUI() {
  const pct = (state.timeLeft / TIME_LIMIT) * 100;
  els.timerText.textContent = `${state.timeLeft.toFixed(1)}s`;
  els.timerFill.style.width = `${pct}%`;

  if (pct > 60) {
    els.timerFill.style.background = "linear-gradient(90deg, #6ef3c5, #5aa9ff)";
  } else if (pct > 30) {
    els.timerFill.style.background = "linear-gradient(90deg, #ffd166, #ff9f43)";
  } else {
    els.timerFill.style.background = "linear-gradient(90deg, #ff6b6b, #ff3b8d)";
  }
}

function setAvatar(character) {
  els.avatar.textContent = character.avatar;
  els.avatar.style.background = `linear-gradient(135deg, ${character.colors[0]}, ${character.colors[1]})`;
  els.avatar.style.boxShadow = `0 18px 40px ${character.colors[0]}55`;
}

function getScenes(character) {
  return [
    {
      chapter: 1,
      text: `${character.name} catches your eye near the neon drinks table and gives you a look that says, "Well?"`,
      choices: [
        {
          text: "Hit them with a playful one-liner.",
          score: 2,
          tone: "good",
          response: `${character.name} laughs immediately. Good start.`
        },
        {
          text: "Ask what brought them here tonight.",
          score: 1,
          tone: "safe",
          response: `${character.name} leans in and gives you a real answer instead of a polite one.`
        },
        {
          text: "Pretend you meant to talk to someone else.",
          score: -1,
          tone: "risky",
          response: `${character.name} raises an eyebrow. Smooth recovery needed.`
        }
      ],
      secret: `Because you answered fast, ${character.name} whispers that confidence is unfairly attractive.`
    },
    {
      chapter: 1,
      text: `${character.name} tilts their head. "Okay, you have my attention. What's your thing?"`,
      choices: [
        {
          text: "Say you collect intense moments, not small talk.",
          score: 2,
          tone: "good",
          response: `${character.name} smiles like that line landed harder than expected.`
        },
        {
          text: "Admit you're better at chemistry than introductions.",
          score: 1,
          tone: "safe",
          response: `"Honest and cute," ${character.name} says, clearly amused.`
        },
        {
          text: "Brag way too hard.",
          score: -1,
          tone: "risky",
          response: `${character.name} folds their arms, unconvinced.`
        }
      ],
      secret: `${character.name} taps your wrist and says, "Fast answers. Dangerous."`
    },
    {
      chapter: 2,
      text: `The music shifts. ${character.name} moves closer and asks whether you're always this bold.`,
      choices: [
        {
          text: "Only with someone worth the risk.",
          score: 2,
          tone: "good",
          response: `${character.name}'s expression softens into pure interest.`
        },
        {
          text: "Bold? No. Curious? Definitely.",
          score: 1,
          tone: "safe",
          response: `${character.name} nods slowly, like they want to hear more.`
        },
        {
          text: "Change the subject immediately.",
          score: -1,
          tone: "risky",
          response: `The tension slips for a second, and ${character.name} notices.`
        }
      ],
      secret: `${character.name} lets their fingers brush yours for one reckless second.`
    },
    {
      chapter: 2,
      text: `${character.name} points toward the balcony. "Two minutes away from the crowd. You in?"`,
      choices: [
        {
          text: "Say yes and make it sound effortless.",
          score: 2,
          tone: "good",
          response: `${character.name} grins. "That was the right answer."`
        },
        {
          text: "Say yes, but tease them first.",
          score: 2,
          tone: "good",
          response: `${character.name} laughs and opens the way for you to follow.`
        },
        {
          text: "Hesitate until the moment gets weird.",
          score: -1,
          tone: "risky",
          response: `${character.name} waits, but the spark flickers.`
        }
      ],
      secret: `You reach the balcony first, and ${character.name} says they were hoping you'd keep up.`
    },
    {
      chapter: 3,
      text: `City air, low music, and one last question. ${character.name} looks straight at you. "So what happens after tonight?"`,
      choices: [
        {
          text: "Tomorrow night. Same energy, less noise.",
          score: 3,
          tone: "good",
          response: `${character.name} bites back a smile, then fails completely.`
        },
        {
          text: "We trade numbers and see if this feeling survives daylight.",
          score: 2,
          tone: "safe",
          response: `${character.name} nods. "Smart answer. Still hot, though."`
        },
        {
          text: "Say you'll probably disappear dramatically.",
          score: -2,
          tone: "risky",
          response: `${character.name} laughs, but now you actually have to recover.`
        }
      ],
      secret: `${character.name} steps closer and says, "You are much more trouble than I planned for."`
    }
  ];
}

function renderScene() {
  const scene = state.scenes[state.currentSceneIndex];
  if (!scene) {
    finishGame();
    return;
  }

  els.chapterLabel.textContent = `Chapter ${scene.chapter}`;
  els.characterName.textContent = state.character.name;
  els.characterMeta.textContent = state.character.meta;
  els.dialogueText.textContent = scene.text;

  els.choices.innerHTML = "";
  scene.choices.forEach((choice, index) => {
    const btn = document.createElement("button");
    btn.className = `choice-btn ${choice.tone || ""}`;
    btn.textContent = choice.text;
    btn.addEventListener("click", () => handleChoice(index));
    els.choices.appendChild(btn);
  });

  startTimer();
}

function startTimer() {
  clearGameTimer();
  state.awaitingChoice = true;
  state.timeLeft = TIME_LIMIT;
  updateTimerUI();

  const startedAt = performance.now();

  state.timer = setInterval(() => {
    const elapsed = (performance.now() - startedAt) / 1000;
    state.timeLeft = clamp(TIME_LIMIT - elapsed, 0, TIME_LIMIT);
    updateTimerUI();

    if (state.timeLeft <= 0) {
      handleTimeout();
    }
  }, 100);
}

function disableChoices() {
  const buttons = els.choices.querySelectorAll("button");
  buttons.forEach(btn => {
    btn.disabled = true;
    btn.style.pointerEvents = "none";
    btn.style.opacity = "0.65";
  });
}

function handleTimeout() {
  if (!state.awaitingChoice) return;

  clearGameTimer();
  state.awaitingChoice = false;
  state.streak = 0;
  state.score -= 1;

  updateStats();
  disableChoices();

  els.dialogueText.textContent = `You pause too long. ${state.character.name} smirks and says, "Too slow. Try to keep up."`;

  setTimeout(() => {
    state.currentSceneIndex += 1;
    renderScene();
  }, RESPONSE_DELAY);
}

function handleChoice(choiceIndex) {
  if (!state.awaitingChoice) return;

  const scene = state.scenes[state.currentSceneIndex];
  const choice = scene.choices[choiceIndex];
  const fastChoice = state.timeLeft >= FAST_THRESHOLD;

  clearGameTimer();
  state.awaitingChoice = false;

  state.score += choice.score;

  if (fastChoice) {
    state.score += 1;
    state.streak += 1;
    state.bestStreak = Math.max(state.bestStreak, state.streak);
  } else {
    state.streak = 0;
  }

  let responseText = choice.response;

  if (fastChoice && scene.secret) {
    state.secrets += 1;
    responseText += ` ${scene.secret}`;
  }

  updateStats();
  disableChoices();
  els.dialogueText.textContent = responseText;

  setTimeout(() => {
    state.currentSceneIndex += 1;
    renderScene();
  }, RESPONSE_DELAY);
}

function getEnding() {
  const { score, secrets, bestStreak, character } = state;

  if (score >= 11) {
    return {
      badge: "Electric Ending",
      title: `${character.name} is already planning date two`,
      text: `You turned fast choices into full-on chemistry. Numbers exchanged, teasing intact, and the night clearly is not over yet. Secret scenes unlocked: ${secrets}. Best fast streak: ${bestStreak}.`
    };
  }

  if (score >= 7) {
    return {
      badge: "Sweet Ending",
      title: `${character.name} wants to see you again`,
      text: `You played it well. The spark stayed alive, the vibe stayed warm, and you leave with a real chance at something fun. Secret scenes unlocked: ${secrets}. Best fast streak: ${bestStreak}.`
    };
  }

  if (score >= 3) {
    return {
      badge: "Almost There",
      title: `You intrigued ${character.name}`,
      text: `Not every line landed, but the story still ended with a smile and a maybe. With sharper timing, this could turn dangerous in the best way. Secret scenes unlocked: ${secrets}. Best fast streak: ${bestStreak}.`
    };
  }

  return {
    badge: "Missed Signal",
    title: `The moment slipped away`,
    text: `The vibe was there, but hesitation got louder than charm. Restart, move faster, and own the next scene. Secret scenes unlocked: ${secrets}. Best fast streak: ${bestStreak}.`
  };
}

function finishGame() {
  clearGameTimer();

  const ending = getEnding();

  els.endBadge.textContent = ending.badge;
  els.endTitle.textContent = ending.title;
  els.endText.textContent = ending.text;

  els.finalScore.textContent = state.score;
  els.finalStreak.textContent = state.bestStreak;
  els.finalSecret.textContent = state.secrets;

  showScreen("end");
}

function resetState() {
  clearGameTimer();

  state.playerGender = els.playerGender.value;
  state.interestGender = els.interestGender.value;
  state.character = pickRandom(profiles[state.interestGender]);
  state.scenes = getScenes(state.character);
  state.currentSceneIndex = 0;
  state.score = 0;
  state.streak = 0;
  state.bestStreak = 0;
  state.secrets = 0;
  state.timeLeft = TIME_LIMIT;
  state.awaitingChoice = false;

  setAvatar(state.character);
  updateStats();
  updateTimerUI();
}

function startGame() {
  resetState();
  showScreen("game");
  renderScene();
}

function goToSetup() {
  clearGameTimer();
  showScreen("setup");
}

els.startBtn.addEventListener("click", startGame);
els.restartBtn.addEventListener("click", goToSetup);
els.playAgainBtn.addEventListener("click", goToSetup);

showScreen("setup");
updateStats();
updateTimerUI();
