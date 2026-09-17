// Procedural scene generator: infinite custom dialogues, fully offline.
// Scenes are woven from template pools biased by the ambient mood (hour,
// lighting) and the character, then injected into the dialog graph as
// detours whose choices return to the intended next scene — so the graph
// integrity stays intact.

const LOCATIONS = {
  intimate: [
    'a candlelit rooftop',
    'the quiet end of the bar',
    'a hidden photo booth',
    'the last train platform',
    'the balcony above the crowd'
  ],
  fresh: [
    'a corner café with the good espresso',
    'the riverfront before the crowds',
    'a bookshop with terrible lighting',
    'the farmers market',
    'a park bench with the best view'
  ],
  bright: [
    'a rooftop pool party',
    'the arcade on the pier',
    'a gelato stand with a line',
    'the city lookout',
    'the loudest corner of the festival'
  ],
  warm: [
    'a rooftop bar at golden hour',
    'the night market',
    'a tiny wine bar',
    'the bridge with the view',
    'a late-night café'
  ]
};

const SETUPS = [
  '{name} pulls you toward {place} and grins. "{prompt}"',
  'The crowd thins. {name} nudges you toward {place}. "{prompt}"',
  '{name} glances back at you from {place}. "{prompt}"',
  'Somehow you end up at {place}. {name} turns to you. "{prompt}"',
  '{name} leads the way to {place}, like they planned it. "{prompt}"'
];

const PROMPTS = [
  'Okay, bonus round. Impress me in one sentence.',
  'Fast answers are unfairly attractive. Try one.',
  'You have my full attention. Do something with it.',
  'One more before the moment slips away. Your best line.',
  'I was not planning for you tonight. Fix that.',
  'Convince me this night is not over yet.'
];

const CHOICE_POOL = [
  { text: 'Say the city suddenly looks like second place.', score: 4 },
  { text: 'Admit you collect intense moments, not places.', score: 4 },
  { text: 'Say the only upgrade that matters is the company.', score: 4 },
  { text: 'Tell them this view is about to get competition.', score: 4 },
  { text: 'Tell them the upgrade was the company.', score: 3 },
  { text: 'Say you are dangerously easy to enjoy tonight.', score: 3 },
  { text: 'Say the detour already paid for itself.', score: 3 },
  { text: 'Admit the night is getting suspiciously good.', score: 3 },
  { text: 'Change the subject to the music.', score: 2 },
  { text: 'Brag way too hard about your timing.', score: 2 },
  { text: 'Pretend you meant to be somewhere else.', score: 2 },
  { text: 'Hesitate until the moment gets weird.', score: 2 }
];

const REACTIONS = {
  good: '{name} laughs — the detour was worth it.',
  safe: '{name} nods, clearly liking the detour.',
  risky: '{name} raises an eyebrow. Recover fast.'
};

let counter = 0;

function pickChoices() {
  const shuffled = [...CHOICE_POOL].sort(() => Math.random() - 0.5);
  const byTone = { good: [], safe: [], risky: [] };
  for (const choice of shuffled) byTone[choice.score >= 4 ? 'good' : choice.score >= 3 ? 'safe' : 'risky'].push(choice);
  const picked = [
    byTone.good.shift(),
    byTone.good.shift() || byTone.safe.shift(),
    byTone.safe.shift() || byTone.risky.shift(),
    byTone.risky.shift() || byTone.safe.shift()
  ].filter(Boolean);
  return picked.map((choice) => ({
    text: choice.text,
    score: choice.score,
    response: REACTIONS[choice.score >= 4 ? 'good' : choice.score >= 3 ? 'safe' : 'risky']
  }));
}

/**
 * Generate a mood-biased detour scene. Every choice returns to `returnTo`.
 */
export function generateScene({ character, ambient, chapter, returnTo }) {
  counter += 1;
  const tone = ambient?.phase?.tone || 'warm';
  const pool = LOCATIONS[tone] || LOCATIONS.warm;
  const place = pool[Math.floor(Math.random() * pool.length)];
  const setup = SETUPS[Math.floor(Math.random() * SETUPS.length)];
  const prompt = PROMPTS[Math.floor(Math.random() * PROMPTS.length)];

  const text = setup
    .replaceAll('{name}', character.name)
    .replaceAll('{place}', place)
    .replaceAll('{prompt}', prompt);

  return {
    id: `gen_${counter}`,
    chapter: chapter || 2,
    text,
    choices: pickChoices().map((choice) => ({ ...choice, next: returnTo }))
  };
}

export function resetGenerator() {
  counter = 0;
}
