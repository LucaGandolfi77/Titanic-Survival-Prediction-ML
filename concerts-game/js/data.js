/* data.js — Concert data, artist info, fictional friends, lyrics */
window.G = window.G || {};

/* ── ARTISTS ── */
G.ARTISTS = {
  swift:    { name: "Taylor Swift",  tour: "The Eras Tour 2.0",         emoji: "🎤", bonus: 50, color: "#ff2d78" },
  styles:   { name: "Harry Styles",  tour: "Together Together Tour",    emoji: "🕺", bonus: 40, color: "#00f5d4" },
  maneskin: { name: "Måneskin",      tour: "Rush! World Tour 2026",     emoji: "🎸", bonus: 60, color: "#ffd700" },
  eilish:   { name: "Billie Eilish", tour: "Hit Me Hard Tour",          emoji: "🖤", bonus: 35, color: "#80ff80" },
  beyonce:  { name: "Beyoncé",       tour: "Renaissance World Tour II", emoji: "👑", bonus: 70, color: "#ffaa00" }
};

/* ── CONCERTS — mx/my are % coords on 1000×500 SVG viewBox ── */
G.CONCERTS = [
  // Taylor Swift
  { id:0,  artistKey:"swift",   date:"2026-03-05", city:"Santiago",     country:"Chile",       venue:"Estadio Nacional",       mx:280, my:388, hint:"A pop princess heads to South America in early spring…" },
  { id:1,  artistKey:"swift",   date:"2026-04-09", city:"Mexico City",  country:"Mexico",      venue:"Estadio GNP Seguros",    mx:175, my:262, hint:"She'll shake it off south of the border in April…" },
  { id:2,  artistKey:"swift",   date:"2026-05-15", city:"London",       country:"UK",          venue:"Wembley Stadium",        mx:472, my:118, hint:"The biggest stadium in England awaits a superstar in May…" },
  { id:3,  artistKey:"swift",   date:"2026-06-07", city:"Paris",        country:"France",      venue:"Stade de France",        mx:485, my:140, hint:"Love story continues in the city of lights this June…" },
  { id:4,  artistKey:"swift",   date:"2026-07-12", city:"New York",     country:"USA",         venue:"MetLife Stadium",        mx:258, my:175, hint:"Welcome to New York — July is gonna be wild…" },
  { id:5,  artistKey:"swift",   date:"2026-08-03", city:"Tokyo",        country:"Japan",       venue:"Tokyo Dome",             mx:872, my:195, hint:"A pop icon visits the Land of the Rising Sun in August…" },
  // Harry Styles
  { id:6,  artistKey:"styles",  date:"2026-05-16", city:"Amsterdam",    country:"Netherlands", venue:"Johan Cruijff ArenA",    mx:490, my:122, hint:"Watermelon sugar in the city of canals, May…" },
  { id:7,  artistKey:"styles",  date:"2026-06-04", city:"London",       country:"UK",          venue:"Wembley Stadium",        mx:468, my:120, hint:"As it was — back in London, June…" },
  { id:8,  artistKey:"styles",  date:"2026-07-20", city:"São Paulo",    country:"Brazil",      venue:"Allianz Parque",         mx:325, my:362, hint:"Golden vibes hit Brazil in July…" },
  { id:9,  artistKey:"styles",  date:"2026-08-10", city:"New York",     country:"USA",         venue:"Madison Square Garden",  mx:260, my:177, hint:"The Garden lights up in August — night one…" },
  { id:10, artistKey:"styles",  date:"2026-08-11", city:"New York",     country:"USA",         venue:"Madison Square Garden",  mx:263, my:173, hint:"Night two in the Big Apple, August…" },
  { id:11, artistKey:"styles",  date:"2026-11-28", city:"Sydney",       country:"Australia",   venue:"Accor Stadium",          mx:892, my:398, hint:"Down under gets a special visit in November…" },
  // Måneskin
  { id:12, artistKey:"maneskin",date:"2026-04-07", city:"Milan",        country:"Italy",       venue:"Unipol Forum",           mx:505, my:152, hint:"Italian rockers come home in early April…" },
  { id:13, artistKey:"maneskin",date:"2026-04-22", city:"Berlin",       country:"Germany",     venue:"Mercedes-Benz Arena",    mx:515, my:128, hint:"Rock invades Germany in late April…" },
  { id:14, artistKey:"maneskin",date:"2026-05-03", city:"Barcelona",    country:"Spain",       venue:"Palau Sant Jordi",       mx:482, my:162, hint:"¡Vamos! A rock band hits Spain in May…" },
  { id:15, artistKey:"maneskin",date:"2026-06-14", city:"London",       country:"UK",          venue:"O2 Arena",               mx:475, my:116, hint:"Italian rock energy explodes in London, June…" },
  { id:16, artistKey:"maneskin",date:"2026-09-05", city:"New York",     country:"USA",         venue:"Madison Square Garden",  mx:255, my:179, hint:"Rush to the Garden in September…" },
  { id:17, artistKey:"maneskin",date:"2026-10-18", city:"Los Angeles",  country:"USA",         venue:"Kia Forum",              mx:138, my:208, hint:"West coast gets loud in October…" },
  // Billie Eilish
  { id:18, artistKey:"eilish",  date:"2026-03-20", city:"Los Angeles",  country:"USA",         venue:"Crypto.com Arena",       mx:135, my:205, hint:"A whispery voice echoes in LA, March…" },
  { id:19, artistKey:"eilish",  date:"2026-04-30", city:"Chicago",      country:"USA",         venue:"United Center",          mx:228, my:170, hint:"The Windy City gets hit hard in April…" },
  { id:20, artistKey:"eilish",  date:"2026-06-22", city:"Amsterdam",    country:"Netherlands", venue:"Ziggo Dome",             mx:493, my:125, hint:"Everything is strange in Amsterdam, June…" },
  { id:21, artistKey:"eilish",  date:"2026-07-08", city:"London",       country:"UK",          venue:"The O2",                 mx:478, my:122, hint:"Bad guy energy hits The O2 in July…" },
  // Beyoncé
  { id:22, artistKey:"beyonce", date:"2026-05-01", city:"Houston",      country:"USA",         venue:"NRG Stadium",            mx:198, my:232, hint:"The queen returns to her hometown in May…" },
  { id:23, artistKey:"beyonce", date:"2026-06-20", city:"Paris",        country:"France",      venue:"Stade de France",        mx:488, my:138, hint:"Formation takes over Paris in June…" },
  { id:24, artistKey:"beyonce", date:"2026-07-30", city:"London",       country:"UK",          venue:"Wembley Stadium",        mx:470, my:114, hint:"Bow down — Wembley, late July…" },
  { id:25, artistKey:"beyonce", date:"2026-08-22", city:"Dubai",        country:"UAE",         venue:"Expo City Arena",        mx:638, my:242, hint:"A desert renaissance in August…" }
];

/* ── FICTIONAL FRIENDS ── */
G.FRIENDS = [
  { name: "Luna Voss",   friendship: 85, craziness: 40, love: 70, style: "Indie pop",    emoji: "🌙" },
  { name: "Marco Daze",  friendship: 60, craziness: 90, love: 50, style: "Rock/punk",    emoji: "🔥" },
  { name: "Stella Osei", friendship: 75, craziness: 55, love: 95, style: "R&B/soul",     emoji: "⭐" },
  { name: "Rex Phantom", friendship: 40, craziness:100, love: 30, style: "Metal/chaos",   emoji: "💀" },
  { name: "Ivy Song",    friendship: 90, craziness: 65, love: 80, style: "K-pop fusion",  emoji: "🌸" }
];

G.ORGANISE_CITIES = ["London", "Milan", "Tokyo", "New York", "Paris", "Sydney"];
G.VENUE_TIERS = [
  { name: "Club",    cost: 500,  capacity: 200   },
  { name: "Theatre", cost: 2000, capacity: 1500  },
  { name: "Arena",   cost: 8000, capacity: 15000 }
];

/* ── SHORT LYRICS for cipher minigame (per artist) ── */
G.LYRICS = {
  swift:    ["SHAKE IT OFF", "LOVE STORY", "BLANK SPACE", "BAD BLOOD", "FEARLESS"],
  styles:   ["WATERMELON SUGAR", "AS IT WAS", "GOLDEN", "ADORE YOU", "SIGN OF TIMES"],
  maneskin: ["ZITTI E BUONI", "BEGGIN", "SUPERMODEL", "MAMMAMIA", "RUSH"],
  eilish:   ["BAD GUY", "LOVELY", "OCEAN EYES", "HAPPIER", "THEREFORE I AM"],
  beyonce:  ["CRAZY IN LOVE", "HALO", "FORMATION", "SINGLE LADIES", "LEMONADE"]
};

/* ── GOSSIP POST TEMPLATES (distractors) ── */
G.GOSSIP_DISTRACTORS = [
  "Just made the best avocado toast 🥑 #brunch",
  "My cat learned a new trick today 🐱",
  "Can't believe the season finale was THAT good 😱",
  "Studying for finals is brutal 📚 send help",
  "New shoes just dropped!! 👟🔥🔥",
  "Rain again. Third day in a row ☔",
  "Pizza delivery took 2 hours. Never again 🍕",
  "Started a new book, can't put it down 📖",
  "Gym grind never stops 💪 #fitlife",
  "Apparently Mercury is in retrograde again 🌀",
  "Binge-watching that new series tonight 📺",
  "My houseplant is finally thriving!! 🌿",
  "Traffic was insane today 🚗😤",
  "Making homemade pasta tonight 🍝",
  "Just adopted a puppy, meet Charlie! 🐕",
  "Monday mood: need more coffee ☕",
  "Weekend getaway plans loading… ✈️",
  "Sunset was unreal tonight 🌅",
];

/* ── DISCOVERY MINIGAME TYPES ── */
G.MG_TYPES = ["gossip", "cipher", "auction", "puzzle"];

/* ── REVIEWS for result cards ── */
G.REVIEWS = {
  "on-stage": [
    "You were spotted singing every word — the crowd went wild!",
    "The artist pulled you on stage! Unforgettable night!",
    "Security couldn't stop you. Neither could the crowd. Legend.",
    "You crowd-surfed your way to the front row. Iconic moment."
  ],
  parterre: [
    "Great energy! You were jumping the whole time!",
    "You sang along to every chorus. True fan energy!",
    "Perfect view from the parterre — money well spent!"
  ],
  seated: [
    "Not bad from the seats! You could see the stage clearly.",
    "Comfy seat, decent view. A solid concert experience.",
    "You made the most of your spot. Good vibes!"
  ],
  "no-entry": [
    "The bouncer shook their head. Maybe next time…",
    "You couldn't get in. The crowd was too wild.",
    "Technical difficulties. Your ticket didn't scan."
  ]
};
