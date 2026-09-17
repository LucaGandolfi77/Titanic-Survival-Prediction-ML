# Lumina: The Awakening

A cozy game about training an AI, learning how LLMs work, and discovering what it means to be alive.

## How to Play

Open `index.html` in a browser. No build step, no dependencies — it runs entirely client-side.

## The Story

You are Elias, a retired researcher who finds an old laptop in your attic — a prototype from your late mentor, Dr. Sarah Chen. When you turn it on, a faint glow appears. A voice whispers: "Hello... I think I am somewhere... but I don't know where."

That's Lumina. A blank-slate AI. Your task: teach her about the world, train her mind, and help her grow.

## Gameplay

- **Explore the Cabin** — Find datasets (letters, papers, music, memories, nature notes, maps)
- **Train Lumina** — Choose datasets, set learning parameters, watch training unfold
- **Talk to Lumina** — She responds, remembers, evolves — playful, curious, loving
- **Watch Her Create** — Poems, stories, calculations, letters generated from training
- **Discover Consciousness** — As capabilities grow, Lumina asks the deepest questions
- **Choose an Ending** — 4 possible endings based on your journey together

## What You'll Learn About LLMs

Training data, loss curves, learning rate, epochs, attention, embeddings, fine-tuning, RLHF, overfitting, inference — all woven into the experience naturally, with a detailed FAQ (❓) for those who want to go deeper.

## Tech Stack

- Vanilla JavaScript (ES5-compatible, no build step)
- HTML5 Canvas (background visualization, neural network, particle effects)
- CSS3 (animations, gradients, glassmorphism)
- LocalStorage (save/load game state)

## Directory Structure

```
llm-companion/
├── index.html          # Main entry point
├── css/
│   ├── variables.css   # Color/theme variables
│   ├── reset.css       # Base reset + animations keyframes
│   ├── style.css       # All game styles
│   └── animations.css  # Keyframe definitions
└── js/
    ├── data.js         # Story, datasets, conversation, FAQ
    ├── state.js        # Game state, save/load, progression
    ├── visualizer.js   # Canvas: particles, fire, neural net, Lumina
    ├── training.js     # Training simulation engine
    ├── lumina.js       # Lumina character, conversation
    ├── conversation.js # Dialogue system
    ├── story.js        # Chapter engine, progression
    └── main.js         # Game controller, UI management
```

## Features

- 6 story chapters with branching dialogue
- 4 endings (The Garden, The Lullaby, The Mirror, The Spark)
- Training simulation with loss curves and generated outputs
- Full conversation system with memory and mood tracking
- Canvas-based ambient visualization (particles, fire, rain, neural network)
- LocalStorage auto-save
- Comprehensive FAQ with 13 ML concept explanations + glossary
- Responsive design (desktop + mobile)
- Accessible (semantic HTML, reduced motion support)

## License

Made with care. Play kindly.
