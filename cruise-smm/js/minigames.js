/* ===== MINI-GAME ENGINE ===== */
import { $, el, randomInt, randomPick, shuffle, clamp } from './utils.js';
import { getState } from './state.js';
import { playSound } from './audio.js';

let currentMinigame = null;
let mgTimer = null;
let mgCallback = null;

export function startMinigame(type, context, callback) {
  mgCallback = callback;
  const overlay = $('#minigame-overlay');
  overlay.classList.remove('hidden');
  overlay.classList.add('active');

  switch (type) {
    case 'framePerfect': runFramePerfect(context); break;
    case 'quickEdit': runQuickEdit(context); break;
    case 'perfectQuestion': runPerfectQuestion(context); break;
    case 'hashtagRush': runHashtagRush(context); break;
    case 'danceFloor': runDanceFloor(context); break;
    case 'captionThis': runCaptionThis(context); break;
    default: endMinigame(1); break;
  }
}

function endMinigame(multiplier, bonusFame = 0) {
  if (mgTimer) { clearInterval(mgTimer); mgTimer = null; }
  const overlay = $('#minigame-overlay');

  const area = $('#minigame-canvas-area');
  const results = el('div', { class: 'mg-results' }, [
    el('h2', { text: multiplier >= 1.5 ? '🌟 AMAZING!' : multiplier >= 1 ? '👍 Good Job!' : '😬 Could be better...' }),
    el('div', { class: 'result-score', text: `${multiplier.toFixed(1)}×` }),
    bonusFame > 0 ? el('div', { class: 'result-bonus', text: `+${bonusFame} bonus fame!` }) : null,
    el('button', { class: 'mg-continue-btn', text: 'CONTINUE →', onclick: () => {
      overlay.classList.add('hidden');
      overlay.classList.remove('active');
      area.innerHTML = '';
      if (mgCallback) mgCallback(multiplier, bonusFame);
    }})
  ]);
  area.appendChild(results);
  playSound('taskComplete');
}

/* ===== MINI-GAME 1: FRAME PERFECT ===== */
function runFramePerfect(context) {
  $('#mg-title').textContent = '📸 FRAME PERFECT';
  const area = $('#minigame-canvas-area');
  const instructions = $('#minigame-instructions');
  area.innerHTML = '';
  instructions.textContent = 'Click SHOOT when the ring is at its peak! 3 moments to capture.';

  let timeLeft = 30;
  let score = 0;
  let moments = 0;
  let totalMoments = 3;
  let waiting = false;
  let peakTime = 0;
  let momentActive = false;

  $('#mg-timer').textContent = timeLeft;
  $('#mg-score').textContent = 'Stars: 0/6';

  const scene = el('div', { style: { width: '100%', height: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: '20px' } });

  const ringContainer = el('div', { style: { position: 'relative', width: '120px', height: '120px' } });
  const ring = el('div', { class: 'timing-ring', html: '📸' });
  const ringFill = el('div', { style: {
    position: 'absolute', inset: '-4px', borderRadius: '50%', border: '4px solid transparent',
    borderTopColor: '#0ea5e9', transition: 'transform 0.05s linear'
  }});
  ringContainer.appendChild(ring);
  ringContainer.appendChild(ringFill);

  const statusText = el('div', { text: 'Get ready...', style: { fontFamily: 'var(--font-mono)', fontSize: '1.2rem', color: 'var(--text-gold)' } });

  const shootBtn = el('button', { class: 'shoot-btn', text: '📸 SHOOT!', onclick: () => {
    if (!momentActive) return;
    const now = Date.now();
    const diff = Math.abs(now - peakTime);

    moments++;
    momentActive = false;

    if (diff < 300) {
      score += 2;
      ring.className = 'timing-ring perfect';
      statusText.textContent = '⭐⭐ PERFECT!';
      playSound('photoPerf');
    } else if (diff < 800) {
      score += 1;
      ring.className = 'timing-ring good';
      statusText.textContent = '⭐ Good!';
      playSound('camera');
    } else {
      ring.className = 'timing-ring miss';
      statusText.textContent = '❌ Missed!';
    }

    $('#mg-score').textContent = `Stars: ${score}/6`;

    if (moments >= totalMoments) {
      clearInterval(mgTimer);
      const multi = 0.5 + (score / 6) * 1.5;
      const bonus = score === 6 ? 500 : 0;
      setTimeout(() => endMinigame(multi, bonus), 800);
    } else {
      setTimeout(scheduleMoment, 1500);
    }
  }});

  scene.appendChild(ringContainer);
  scene.appendChild(statusText);
  scene.appendChild(shootBtn);
  area.appendChild(scene);

  let rotation = 0;
  function scheduleMoment() {
    if (moments >= totalMoments) return;
    const delay = randomInt(2000, 5000);
    statusText.textContent = 'Wait for it...';
    ring.className = 'timing-ring';

    setTimeout(() => {
      if (moments >= totalMoments) return;
      momentActive = true;
      peakTime = Date.now() + 750;
      statusText.textContent = '🎯 NOW!';
      ring.style.boxShadow = '0 0 20px rgba(251,191,36,0.6)';

      setTimeout(() => {
        if (momentActive) {
          momentActive = false;
          moments++;
          statusText.textContent = '❌ Too slow!';
          ring.className = 'timing-ring miss';
          ring.style.boxShadow = '';
          $('#mg-score').textContent = `Stars: ${score}/6`;
          if (moments >= totalMoments) {
            clearInterval(mgTimer);
            const multi = 0.5 + (score / 6) * 1.5;
            const bonus = score === 6 ? 500 : 0;
            setTimeout(() => endMinigame(multi, bonus), 800);
          } else {
            setTimeout(scheduleMoment, 1000);
          }
        }
        ring.style.boxShadow = '';
      }, 1500);
    }, delay);
  }

  mgTimer = setInterval(() => {
    timeLeft--;
    $('#mg-timer').textContent = timeLeft;
    rotation += 12;
    ringFill.style.transform = `rotate(${rotation}deg)`;
    if (timeLeft <= 0) {
      clearInterval(mgTimer);
      const multi = 0.5 + (score / 6) * 1.5;
      const bonus = score === 6 ? 500 : 0;
      endMinigame(multi, bonus);
    }
  }, 1000);

  setTimeout(scheduleMoment, 1500);
}

/* ===== MINI-GAME 2: QUICK EDIT ===== */
function runQuickEdit(context) {
  $('#mg-title').textContent = '✂️ QUICK EDIT';
  const area = $('#minigame-canvas-area');
  const instructions = $('#minigame-instructions');
  area.innerHTML = '';
  instructions.textContent = 'Click BLUE clips to include, skip RED clips! Gray is optional.';

  let timeLeft = 10;
  let correctIncl = 0;
  let wrongIncl = 0;
  let totalGood = 0;

  $('#mg-timer').textContent = timeLeft;
  $('#mg-score').textContent = 'Score: 0';

  const timeline = el('div', { class: 'edit-timeline' });
  const clips = [];
  const types = [];

  for (let i = 0; i < 20; i++) {
    const r = Math.random();
    let type;
    if (r < 0.4) { type = 'good'; totalGood++; }
    else if (r < 0.65) { type = 'bad'; }
    else { type = 'neutral'; }
    types.push(type);

    const clip = el('div', {
      class: `edit-clip ${type}`,
      text: type === 'good' ? '✓' : type === 'bad' ? '✗' : '~',
      'data-index': i,
      'data-type': type,
      onclick: (e) => {
        const c = e.currentTarget;
        if (c.classList.contains('included') || c.classList.contains('excluded')) return;

        if (type === 'good') {
          c.classList.add('included');
          correctIncl++;
          playSound('btn');
        } else if (type === 'bad') {
          c.classList.add('included');
          wrongIncl++;
          playSound('loveDn');
        } else {
          c.classList.add('included');
          playSound('btn');
        }
        updateScore();
      }
    });
    clips.push(clip);
    timeline.appendChild(clip);
  }

  const accuracy = el('div', { class: 'edit-accuracy', text: 'Accuracy: 100%' });
  area.style.position = 'relative';
  area.appendChild(timeline);
  area.appendChild(accuracy);

  function updateScore() {
    const total = correctIncl + wrongIncl;
    const acc = total > 0 ? Math.round((correctIncl / total) * 100) : 100;
    const pts = correctIncl * 10 - wrongIncl * 15;
    accuracy.textContent = `Accuracy: ${acc}%`;
    $('#mg-score').textContent = `Score: ${Math.max(0, pts)}`;
  }

  mgTimer = setInterval(() => {
    timeLeft--;
    $('#mg-timer').textContent = timeLeft;
    if (timeLeft <= 0) {
      clearInterval(mgTimer);
      const total = correctIncl + wrongIncl;
      const acc = total > 0 ? (correctIncl / total) : 0.5;
      const coverage = totalGood > 0 ? correctIncl / totalGood : 0;
      const multi = 0.5 + (acc * 0.75 + coverage * 0.75);
      const bonus = acc >= 0.8 && coverage >= 0.8 ? 300 : 0;
      endMinigame(Math.min(multi, 2), bonus);
    }
  }, 1000);
}

/* ===== MINI-GAME 3: PERFECT QUESTION ===== */
function runPerfectQuestion(context) {
  $('#mg-title').textContent = '🎤 PERFECT QUESTION';
  const area = $('#minigame-canvas-area');
  const instructions = $('#minigame-instructions');
  area.innerHTML = '';
  instructions.textContent = 'Ask questions that match the character\'s interests! Watch the mood meter.';

  const charId = context?.charId;
  const charData = context?.character;
  const preferred = charData?.preferredTopics || ['passion', 'ocean'];
  const disliked = charData?.dislikedTopics || ['secret'];

  let mood = 50;
  let questionsAsked = 0;
  let maxQuestions = 5;
  let timeLeft = 35;

  $('#mg-timer').textContent = timeLeft;
  $('#mg-score').textContent = `Mood: ${mood}`;

  const questionPool = shuffle([
    { emoji: '❤️', text: 'Tell me your passion...', topic: 'passion' },
    { emoji: '😂', text: "What's the funniest thing...", topic: 'funny' },
    { emoji: '🌊', text: 'How does the ocean make you feel...', topic: 'ocean' },
    { emoji: '🏆', text: "What's your proudest moment...", topic: 'proud' },
    { emoji: '😮', text: 'Tell me a secret about this ship...', topic: 'secret' },
    { emoji: '👨‍👩‍👧', text: 'Do you miss your family...', topic: 'family' },
    { emoji: '🌍', text: "What's your favorite port...", topic: 'port' },
    { emoji: '🎯', text: "What's your advice for young sailors...", topic: 'advice' },
  ]).slice(0, 5);

  const qArea = el('div', { class: 'question-area' });

  const moodContainer = el('div', { class: 'mood-meter-container' });
  const moodLabel = el('div', { text: `${charData?.emoji || '👤'} ${charData?.name || 'Character'} — Mood`, style: { fontSize: '0.9rem', color: 'var(--text-gold)' } });
  const moodBar = el('div', { class: 'mood-meter-bar' });
  const moodFill = el('div', { class: 'mood-meter-fill', style: { width: '50%' } });
  moodBar.appendChild(moodFill);
  moodContainer.appendChild(moodLabel);
  moodContainer.appendChild(moodBar);

  const cardsContainer = el('div', { class: 'question-cards' });

  questionPool.forEach((q, i) => {
    const card = el('div', { class: 'question-card', onclick: () => {
      if (card.classList.contains('used')) return;
      card.classList.add('used');
      questionsAsked++;

      if (preferred.includes(q.topic)) {
        mood = clamp(mood + 25, 0, 100);
        playSound('loveUp');
      } else if (disliked.includes(q.topic)) {
        mood = clamp(mood - 15, 0, 100);
        playSound('loveDn');
      } else {
        mood = clamp(mood + 10, 0, 100);
        playSound('btn');
      }

      moodFill.style.width = `${mood}%`;
      $('#mg-score').textContent = `Mood: ${mood}`;

      if (questionsAsked >= maxQuestions) {
        clearInterval(mgTimer);
        const multi = 0.5 + (mood / 100) * 1.5;
        const bonus = mood >= 100 ? 1000 : 0;
        setTimeout(() => endMinigame(multi, bonus), 500);
      }
    }}, [
      el('span', { class: 'q-emoji', text: q.emoji }),
      el('span', { text: q.text }),
    ]);
    cardsContainer.appendChild(card);
  });

  qArea.appendChild(moodContainer);
  qArea.appendChild(cardsContainer);
  area.appendChild(qArea);

  mgTimer = setInterval(() => {
    timeLeft--;
    $('#mg-timer').textContent = timeLeft;
    if (timeLeft <= 0) {
      clearInterval(mgTimer);
      const multi = 0.5 + (mood / 100) * 1.5;
      const bonus = mood >= 100 ? 1000 : 0;
      endMinigame(multi, bonus);
    }
  }, 1000);
}

/* ===== MINI-GAME 4: HASHTAG RUSH ===== */
function runHashtagRush(context) {
  $('#mg-title').textContent = '#️⃣ HASHTAG RUSH';
  const area = $('#minigame-canvas-area');
  const instructions = $('#minigame-instructions');
  area.innerHTML = '';
  instructions.textContent = 'Click GOLD trending words! Click BLUE relevant ones! AVOID RED banned words!';

  let timeLeft = 20;
  let score = 0;
  let combo = 0;
  let comboMulti = 1;

  const trending = ['luxury', 'cruise', 'sunset', 'mediterranean', 'wanderlust', 'shiplife', 'aurora', 'vista'];
  const relevant = ['food', 'crew', 'travel', 'sea', 'adventure', 'dream', 'waves', 'paradise'];
  const banned = ['boring', 'crowded', 'expensive', 'complaint', 'terrible', 'ugly'];

  $('#mg-timer').textContent = timeLeft;
  $('#mg-score').textContent = `Fame: ${score}`;

  const hashArea = el('div', { class: 'hashtag-area' });
  const comboEl = el('div', { class: 'combo-indicator', text: '' });
  hashArea.appendChild(comboEl);
  area.appendChild(hashArea);

  let wordInterval;
  function spawnWord() {
    const r = Math.random();
    let word, type;
    if (r < 0.35) { word = randomPick(trending); type = 'trending'; }
    else if (r < 0.7) { word = randomPick(relevant); type = 'relevant'; }
    else { word = randomPick(banned); type = 'banned'; }

    const y = randomInt(20, 250);
    const speed = randomInt(3, 6);
    const wordEl = el('div', {
      class: `flying-word ${type}`,
      text: `#${word}`,
      style: { top: `${y}px`, animationDuration: `${speed}s` },
      onclick: () => {
        if (wordEl.classList.contains('clicked')) return;
        wordEl.classList.add('clicked');

        // Log click with bounding rect
        try {
          requestAnimationFrame(() => {
            const r = wordEl.getBoundingClientRect();
            console.log(`[hashtagRush] click: #${word} (type=${type}) rect=${Math.round(r.left)},${Math.round(r.top)} ${Math.round(r.width)}x${Math.round(r.height)}`);
          });
        } catch (err) {}

        if (type === 'trending') {
          score += Math.round(15 * comboMulti);
          combo++;
          if (combo >= 3) { comboMulti = 1.5; comboEl.textContent = '🔥 COMBO ×1.5!'; }
          playSound('fame');
        } else if (type === 'relevant') {
          score += Math.round(8 * comboMulti);
          playSound('btn');
        } else {
          score -= 20;
          combo = 0;
          comboMulti = 1;
          comboEl.textContent = '';
          playSound('loveDn');
        }
        $('#mg-score').textContent = `Fame: ${Math.max(0, score)}`;

        setTimeout(() => {
          try { console.log(`[hashtagRush] clickedRemove: #${word} (type=${type})`); } catch (err) {}
          wordEl.remove();
        }, 300);
      }
    });

    hashArea.appendChild(wordEl);

    // Log spawned word details including DOM bounding rect (x,y,width,height)
    requestAnimationFrame(() => {
      try {
        const rect = wordEl.getBoundingClientRect();
        console.log(`[hashtagRush] spawnWord: #${word} (type=${type}, top=${y}px, speed=${speed}s) rect=${Math.round(rect.left)},${Math.round(rect.top)} ${Math.round(rect.width)}x${Math.round(rect.height)}`);
      } catch (err) {}
    });

    setTimeout(() => {
      if (wordEl.parentNode) {
        try { console.log(`[hashtagRush] autoRemove: #${word} (type=${type})`); } catch (err) {}
        wordEl.remove();
      }
    }, speed * 1000);
  }

  wordInterval = setInterval(spawnWord, 800);
  spawnWord();

  mgTimer = setInterval(() => {
    timeLeft--;
    $('#mg-timer').textContent = timeLeft;
    if (timeLeft <= 0) {
      clearInterval(mgTimer);
      clearInterval(wordInterval);
      const multi = 0.5 + Math.min(score / 100, 1.5);
      endMinigame(multi, Math.max(0, score));
    }
  }, 1000);
}

/* ===== MINI-GAME 5: DANCE FLOOR ===== */
function runDanceFloor(context) {
  $('#mg-title').textContent = '🕺 DANCE FLOOR';
  const area = $('#minigame-canvas-area');
  const instructions = $('#minigame-instructions');
  area.innerHTML = '';
  instructions.textContent = 'Press the arrow keys (← ↑ ↓ →) when arrows reach the hit zone!';

  let score = 0;
  let hits = 0;
  let total = 0;
  let timeLeft = 20;
  const arrows = ['←', '↑', '↓', '→'];
  const keys = ['ArrowLeft', 'ArrowUp', 'ArrowDown', 'ArrowRight'];

  $('#mg-timer').textContent = timeLeft;
  $('#mg-score').textContent = 'Score: 0';

  const dArea = el('div', { class: 'dance-area' });
  const lanesContainer = el('div', { class: 'dance-lanes' });
  const feedbackContainer = el('div', { style: { position: 'relative', height: '40px', width: '280px' } });

  const lanes = [];
  for (let i = 0; i < 4; i++) {
    const lane = el('div', { class: 'dance-lane' }, [
      el('div', { class: 'dance-hit-zone', text: arrows[i] })
    ]);
    lanes.push(lane);
    lanesContainer.appendChild(lane);
  }

  dArea.appendChild(lanesContainer);
  dArea.appendChild(feedbackContainer);
  area.appendChild(dArea);

  const activeArrows = [];

  function spawnArrow() {
    const laneIdx = randomInt(0, 3);
    const arrowEl = el('div', { class: 'dance-arrow', text: arrows[laneIdx],
      style: { animationDuration: '2s' }
    });
    lanes[laneIdx].appendChild(arrowEl);

    const arrowData = { el: arrowEl, lane: laneIdx, time: Date.now(), hit: false };
    activeArrows.push(arrowData);
    total++;

    setTimeout(() => {
      if (!arrowData.hit && arrowEl.parentNode) {
        arrowEl.classList.add('miss');
        showFeedback('MISS', 'miss');
        setTimeout(() => arrowEl.remove(), 300);
      }
    }, 2000);
  }

  function showFeedback(text, cls) {
    const fb = el('div', { class: `dance-feedback ${cls}`, text });
    feedbackContainer.appendChild(fb);
    setTimeout(() => fb.remove(), 500);
  }

  function handleKey(e) {
    const idx = keys.indexOf(e.key);
    if (idx === -1) return;
    e.preventDefault();

    const now = Date.now();
    let bestArrow = null;
    let bestDiff = Infinity;

    for (const a of activeArrows) {
      if (a.hit || a.lane !== idx) continue;
      const elapsed = now - a.time;
      const targetTime = 1700; // when arrow hits zone
      const diff = Math.abs(elapsed - targetTime);
      if (diff < bestDiff) { bestDiff = diff; bestArrow = a; }
    }

    if (bestArrow && bestDiff < 500) {
      bestArrow.hit = true;
      hits++;
      if (bestDiff < 100) {
        score += 15;
        bestArrow.el.classList.add('perfect');
        showFeedback('PERFECT!', 'perfect');
        playSound('photoPerf');
      } else {
        score += 8;
        bestArrow.el.classList.add('good');
        showFeedback('GOOD', 'good');
        playSound('btn');
      }
      setTimeout(() => bestArrow.el.remove(), 200);
      $('#mg-score').textContent = `Score: ${score}`;
    }
  }

  document.addEventListener('keydown', handleKey);

  let spawnInterval = setInterval(spawnArrow, 900);
  spawnArrow();

  mgTimer = setInterval(() => {
    timeLeft--;
    $('#mg-timer').textContent = timeLeft;
    if (timeLeft <= 0) {
      clearInterval(mgTimer);
      clearInterval(spawnInterval);
      document.removeEventListener('keydown', handleKey);
      const acc = total > 0 ? hits / total : 0;
      const multi = 0.5 + acc * 1.5;
      const bonus = hits === total && total > 5 ? 300 : 0;
      endMinigame(multi, bonus);
    }
  }, 1000);
}

/* ===== MINI-GAME 6: CAPTION THIS ===== */
function runCaptionThis(context) {
  $('#mg-title').textContent = '✍️ CAPTION THIS!';
  const area = $('#minigame-canvas-area');
  const instructions = $('#minigame-instructions');
  area.innerHTML = '';
  instructions.textContent = 'Build a 6-word caption! Pick from each word bank. Be creative!';

  let timeLeft = 15;
  let captionWords = [];

  const scenes = ['🌅', '🚢', '🏊', '🍽️', '🌊', '🎭', '🎵', '⚓'];
  const sceneEmoji = randomPick(scenes);
  const sceneBgs = {
    '🌅': 'linear-gradient(135deg, #f97316, #ec4899)',
    '🚢': 'linear-gradient(135deg, #0ea5e9, #1e3a5f)',
    '🏊': 'linear-gradient(135deg, #06b6d4, #0ea5e9)',
    '🍽️': 'linear-gradient(135deg, #92400e, #b45309)',
    '🌊': 'linear-gradient(135deg, #0369a1, #06b6d4)',
    '🎭': 'linear-gradient(135deg, #1a0a2e, #2d1b69)',
    '🎵': 'linear-gradient(135deg, #f59e0b, #fbbf24)',
    '⚓': 'linear-gradient(135deg, #1e3a5f, #0369a1)',
  };

  const adjectives = shuffle(['stunning', 'magical', 'luxurious', 'golden', 'infinite', 'dreamy', 'epic', 'breathtaking']);
  const nouns = shuffle(['sunset', 'voyage', 'adventure', 'moment', 'horizon', 'paradise', 'escape', 'memories']);
  const emojis = shuffle(['✨', '🌊', '💫', '🔥', '❤️', '🌴', '☀️', '🌟']);
  const hashtags = shuffle(['#cruiselife', '#shiplife', '#wanderlust', '#aurora', '#mediterranean', '#luxury', '#travelgoals', '#seaview']);

  const viralCombos = [
    ['stunning', 'sunset'], ['magical', 'voyage'], ['golden', 'moment'],
    ['breathtaking', 'horizon'], ['epic', 'adventure'], ['dreamy', 'paradise']
  ];

  $('#mg-timer').textContent = timeLeft;
  $('#mg-score').textContent = 'Words: 0/6';

  const cArea = el('div', { class: 'caption-area' });

  const sceneBox = el('div', { class: 'caption-scene', text: sceneEmoji, style: { background: sceneBgs[sceneEmoji] } });
  const captionBuilt = el('div', { class: 'caption-built', text: 'Your caption here...' });
  const banks = el('div', { class: 'word-banks' });

  function addWord(word) {
    if (captionWords.length >= 6) return;
    captionWords.push(word);
    updateCaption();
    playSound('btn');
    $('#mg-score').textContent = `Words: ${captionWords.length}/6`;

    if (captionWords.length >= 6) {
      clearInterval(mgTimer);
      setTimeout(() => scoreCaption(), 500);
    }
  }

  function updateCaption() {
    captionBuilt.innerHTML = '';
    if (captionWords.length === 0) {
      captionBuilt.textContent = 'Your caption here...';
    } else {
      captionWords.forEach((w, i) => {
        const we = el('span', { class: 'cap-word', text: w, onclick: () => {
          captionWords.splice(i, 1);
          updateCaption();
          $('#mg-score').textContent = `Words: ${captionWords.length}/6`;
        }});
        captionBuilt.appendChild(we);
      });
    }
  }

  function makeBank(title, words) {
    const bank = el('div', { class: 'word-bank' }, [
      el('div', { class: 'word-bank-title', text: title })
    ]);
    words.slice(0, 4).forEach(w => {
      const tile = el('div', { class: 'word-tile', text: w, onclick: () => {
        if (tile.classList.contains('used')) return;
        tile.classList.add('used');
        addWord(w);
      }});
      bank.appendChild(tile);
    });
    return bank;
  }

  banks.appendChild(makeBank('Adjectives', adjectives));
  banks.appendChild(makeBank('Nouns', nouns));
  banks.appendChild(makeBank('Emojis', emojis));
  banks.appendChild(makeBank('Hashtags', hashtags));

  cArea.appendChild(sceneBox);
  cArea.appendChild(captionBuilt);
  cArea.appendChild(banks);
  area.appendChild(cArea);

  function scoreCaption() {
    let viralScore = 0;
    const hasEmoji = captionWords.some(w => emojis.includes(w));
    const hasHashtag = captionWords.some(w => hashtags.includes(w));
    const usedAdj = captionWords.filter(w => adjectives.includes(w));
    const usedNouns = captionWords.filter(w => nouns.includes(w));

    if (hasEmoji) viralScore += 10;
    if (hasHashtag) viralScore += 10;

    for (const combo of viralCombos) {
      if (usedAdj.includes(combo[0]) && usedNouns.includes(combo[1])) {
        viralScore += 30;
        break;
      }
    }
    if (usedAdj.length > 0 && usedNouns.length > 0) viralScore += 15;

    viralScore = Math.min(viralScore, 65);
    const bonus = Math.round(viralScore * 6);
    const multi = 0.5 + (viralScore / 65) * 1.5;
    endMinigame(multi, bonus);
  }

  mgTimer = setInterval(() => {
    timeLeft--;
    $('#mg-timer').textContent = timeLeft;
    if (timeLeft <= 0) {
      clearInterval(mgTimer);
      if (captionWords.length > 0) {
        scoreCaption();
      } else {
        endMinigame(0.5, 0);
      }
    }
  }, 1000);
}
