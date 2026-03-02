# 🚢 CRUISE SHIP SOCIAL MEDIA MANAGER

A **day-management simulator** where you play as the newly hired social media manager aboard the luxury cruise ship **MS Aurora Infinita**. Your goal: grow from **0 to 1,000,000 followers** in just 7 days!

## 🎮 How to Play

Open `index.html` in any modern browser. No build tools, no server, no dependencies.

### Daily Structure
Each day has **8 time slots**: morning briefing, 5 work slots, lunch break, and evening free time.

### Work Tasks
Choose from **photo shootings**, **safety courses**, **crew interviews**, and **social media tasks** — each triggers a unique mini-game that multiplies fame earned.

### Team Management
Lead a crew of 4 specialists:
| Member | Specialty | Wage |
|--------|-----------|------|
| 🎬 Marco | Video (95) | €120/day |
| 📸 Yuki | Photo (97) | €110/day |
| ✂️ Sofia | Editing (92) | €90/day |
| 🎤 Diego | Hype (50) | €70/day |

Keep morale high to boost task performance. Treat them during lunch!

### 6 Mini-Games
- **📷 Frame Perfect** — Timing-based camera shots
- **✂️ Quick Edit** — Include/exclude clips on deadline  
- **🎤 Perfect Question** — Match questions to character preferences
- **#️⃣ Hashtag Rush** — Click trending words, dodge banned hashtags
- **💃 Dance Floor** — Rhythm arrow-key game
- **✍️ Caption This** — Assemble viral captions from word tiles

### 8 Romanceable Characters
Build relationships through dates and choices:
- 🧹 Carmen Ruiz — Housekeeping
- 🍳 Baptiste Lefebvre — Head Chef
- 🌊 Marina Costa — Diving Instructor
- 🎭 Theo Wells — Entertainment Director
- 🏥 Dr. Isabel Santos — Ship Doctor
- ⚓ Lt. James Park — Navigation Officer
- 🎵 Luna Greco — Ship Musician
- ⚓ Captain Elena Vasquez — Ship Captain (hardest to romance!)

### Equipment Shop
Upgrade your gear for permanent fame multipliers:
Pro Camera, Cinema Lens, LED Panel, Audio Interface, Social Suite, Drone.

### 5 Endings
Your ending depends on total fame and romance level:
- 🏆 **Champion** (1M+ fame)
- 🌟 **Amazing** (800K-999K)
- ⭐ **Great** (500K-800K)
- 😐 **Decent** (200K-500K)
- 💀 **Fired** (<200K)

## 📁 Project Structure

```
cruise-smm/
├── index.html
├── README.md
├── css/
│   ├── variables.css
│   ├── reset.css
│   ├── layout.css
│   ├── cards.css
│   ├── minigames.css
│   ├── ui.css
│   └── animations.css
└── js/
    ├── main.js        — Game loop & state machine
    ├── utils.js       — DOM helpers & save/load
    ├── state.js       — Global game state
    ├── characters.js  — 8 dateable characters & dialogues
    ├── tasks.js       — Task definitions & fame calculation
    ├── team.js        — Team management & wages
    ├── planner.js     — Day schedule & time slots
    ├── lunch.js       — Food, drinks & team treats
    ├── dates.js       — Dating mechanics & romance
    ├── minigames.js   — 6 interactive mini-games
    ├── economy.js     — Equipment shop & milestones
    ├── audio.js       — Web Audio API SFX & music
    └── ui.js          — All screen rendering & modals
```

## 🛠 Tech Stack
- Vanilla HTML5 / CSS3 / JavaScript (ES Modules)
- Web Audio API (oscillator-based SFX & music)
- localStorage for save/load
- CSS gradients for location backgrounds
- Google Fonts: Playfair Display, Inter, Share Tech Mono
- Zero dependencies · Zero build tools

## License
MIT
