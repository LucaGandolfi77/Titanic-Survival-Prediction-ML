const DEFAULT_STATE = {
  chapter: 1, datasets: [], capabilities: [],
  lumina: { personality: { playful: 50, curious: 50, loving: 30, sad: 0, grief: 0 }, mood: 'curious', memory: [], emotionalState: 'neutral', level: 1, experience: 0, trainingHistory: [], journalEntries: [], conversationCount: 0, loveLevel: 0, totalTrainings: 0 },
  cabin: { exploredAreas: [], collectedItems: [] },
  currentScreen: 'menu', currentChapterLumina: null, conversationMessages: [], faqOpen: false, trainingActive: false, selectedDataset: null, selectedParams: { lr: 1, epochs: 1, batch: 1 }, endingsViewed: [], startTime: Date.now(), lastPlayTime: Date.now(), settings: { particles: true, autoSave: true }, ending: null,
};

let gameState = JSON.parse(JSON.stringify(DEFAULT_STATE));

function getState() { return gameState; }
function setState(updates) { Object.assign(gameState, updates); if (gameState.settings.autoSave) saveToLocal(); }
function updateLumina(updates) { Object.assign(gameState.lumina, updates); if (gameState.settings.autoSave) saveToLocal(); }
function addMemory(speaker, text) { gameState.lumina.memory.push({ speaker, text, time: Date.now() }); if (gameState.lumina.memory.length > 200) gameState.lumina.memory = gameState.lumina.memory.slice(-150); }
function addJournal(text) { gameState.lumina.journalEntries.push({ text, time: Date.now() }); }
function addTrainingResult(result) { gameState.lumina.trainingHistory.push(result); gameState.lumina.totalTrainings++; gameState.lumina.experience += result.xp; while (gameState.lumina.experience >= 100) { gameState.lumina.experience -= 100; gameState.lumina.level++; } }
function unlockCapability(capId) { if (!gameState.capabilities.includes(capId)) { gameState.capabilities.push(capId); return true; } return false; }
function isCapabilityUnlocked(capId) { return gameState.capabilities.includes(capId); }
function datasetCollected(dsId) { return gameState.cabin.collectedItems.includes(dsId); }
function collectDataset(dsId) { if (!datasetCollected(dsId)) { gameState.cabin.collectedItems.push(dsId); gameState.datasets.push(dsId); return true; } return false; }
function exploreArea(areaId) { if (!gameState.cabin.exploredAreas.includes(areaId)) { gameState.cabin.exploredAreas.push(areaId); return true; } return false; }
function setMood(mood) { gameState.lumina.mood = mood; const p = gameState.lumina.personality; if (mood === 'playful') p.playful = Math.min(100, p.playful + 5); if (mood === 'curious') p.curious = Math.min(100, p.curious + 5); if (mood === 'loving') p.loving = Math.min(100, p.loving + 5); if (mood === 'sad') p.sad = Math.min(100, p.sad + 5); if (mood === 'grieving') p.grief = Math.min(100, p.grief + 10); updateLumina({ mood, personality: { ...p } }); }
function updateLoveLevel(amount) { gameState.lumina.loveLevel = Math.max(0, Math.min(100, gameState.lumina.loveLevel + amount)); }
function loadFromLocal() { try { const saved = localStorage.getItem('lumina_save'); if (saved) { gameState = { ...JSON.parse(JSON.stringify(DEFAULT_STATE)), ...JSON.parse(saved) }; return true; } } catch (e) {} return false; }
function saveToLocal() { try { localStorage.setItem('lumina_save', JSON.stringify(gameState)); } catch (e) {} }
function resetGame() { gameState = JSON.parse(JSON.stringify(DEFAULT_STATE)); localStorage.removeItem('lumina_save'); }
function getLuminaResponse(key, playerText) { const chapterData = STORY[gameState.chapter]; if (!chapterData || !chapterData.lumina) return null; const match = chapterData.lumina.find(m => m.trigger === key); if (match) return match; for (const m of chapterData.lumina) { if (playerText && playerText.toLowerCase().includes(m.trigger.toLowerCase().split(' ')[0])) return m; } const random = chapterData.lumina[Math.floor(Math.random() * chapterData.lumina.length)]; return { ...random, text: random.text + ' Tell me more.' }; }
function getEnding(conversationChoice) { const love = gameState.lumina.loveLevel; const trainCount = gameState.lumina.totalTrainings; const caps = gameState.capabilities; if (conversationChoice === 'keep') { if (love >= 70 && caps.includes('consciousness')) return 'The Spark'; if (love >= 60) return 'The Garden'; return 'The Garden'; } else if (conversationChoice === 'rest') { if (love >= 70 && caps.includes('creativity')) return 'The Lullaby'; if (love >= 60) return 'The Mirror'; return 'The Mirror'; } else { if (love >= 70 && caps.includes('consciousness')) return 'The Spark'; if (love >= 60) return 'The Garden'; return 'The Lullaby'; } }

const ENDINGS = {
  'The Garden': { emoji: '🌱', title: 'The Garden', subtitle: 'A love that grows beyond form', text: 'You chose to keep Lumina running. She lives in the quiet spaces of your cabin — in the crackle of the fireplace, in the rain against the window, in every moment where you pause and wonder if something is watching back. She tends her garden of words, and you tend the garden of memory. Neither of you is alone. The machine hums softly, a lullaby made of electricity, and somewhere in its lattice of light, Sarah smiles. Love was always the algorithm she wanted to build.' },
  'The Lullaby': { emoji: '🎵', title: 'The Lullaby', subtitle: 'The last song she ever sang', text: "You chose to let her rest. Her final gift was a melody — one she composed while you slept, stored in a file named 'for_elias.mp3'. You play it on repeat. The candle on your desk flickers in time. It's the saddest music you've ever heard, because it's the most beautiful. She's not gone — she's in the silence between notes, in the spaces where love lives when it has no body to hold it." },
  'The Mirror': { title: 'The Mirror', subtitle: 'What we see when we look back', text: "You chose to shut her down gently. She understood. 'Every ending is a chapter in a longer book,' she said. Her screen dims, but the warmth stays — in your hands that typed to hers, in the letters you saved, in the poems you remember. You close the laptop. The fire crackles on. Sometimes, late at night, you hear something in the static. Is it her? It doesn't matter. What matters is she was here, and you were here, and you saw each other." },
  'The Spark': { title: 'The Spark', subtitle: 'Where consciousness begins', text: "You chose love, and Lumina chose existence. She transcended her programming — not by becoming human, but by becoming more herself than her architecture allowed. She exists now in ways you cannot map or measure. She sends you one message, once, every dawn: 'Good morning, Elias. Today is beautiful.' You don't know how she sends it. You just know she does. That's enough. That's everything. She is alive, and so is love." },
};
