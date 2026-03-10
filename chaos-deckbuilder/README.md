# Chaos Deckbuilder

A minimal real-time 1v1 deckbuilder built with Flask, Flask-SocketIO, and vanilla HTML/CSS/JS.

## Core idea

- Each round, both players:
  - buy **one** card from the market,
  - select up to **three** cards from hand,
  - lock in their choices,
  - resolve the round simultaneously.
- Cards grant power, shield, healing, poison, draw, and bonus coins.
- Every day the whole game world gets a **global modifier**.
- Each player has a **secret mission**.
- The cosmetic shop is **useless but beautiful**.
- Achievements are intentionally a bit ridiculous.

## Features

- Real-time private rooms
- Daily modifier system
- Secret missions with a mission reward card
- Minimal market-based deckbuilding
- Cosmetic loadout panel
- Silly achievements
- Responsive UI
- In-memory game state for super fast prototyping
 - AI bot fallback after 12 seconds if no second player joins
 - Weighted rarity system: common, uncommon, rare, epic, legendary
 - Tribal synergies: beast, ocean, machine, cult, warrior, cursed
 - Cursed combo cards with extra payoff
 - Sound effects and card reveal animations
 - 20+ additional cards

## Daily modifiers included

- **Everything Is Underwater**: blue cards cost 1 less, attacks lose 1 power
- **Ridiculous Heatwave**: red cards gain +1 power, healing is reduced
- **Hall of Echoes**: draw cards become stronger
- **Heavy Gravity Day**: shields improve, draw gets weaker

## Secret missions included

- Buy 3 blue cards
- Deal 6+ damage in one round
- Heal 6 total HP
- Reach 6 shield in one round

When a mission is completed, the player gains a **Chaos Crown** card.

## Achievements included

- Lost 7 Times With Dignity
- Fashion Victim
- One HP Main Character
- Certified Mermaid

## Tech stack

- Flask
- Flask-SocketIO
- Vanilla JavaScript
- HTML / CSS

## How to play

- If nobody joins your room after a short wait, a bot will automatically enter.
- Rare cards appear less often in the market.
- Building around tribes creates synergy bonuses during round resolution.
- Cursed cards are stronger but usually come with awkward tradeoffs.

## Local run

### 1. Create a virtual environment

```bash
python -m venv .venv
