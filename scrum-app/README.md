# ScrumFlow — Agile Project Management

A **complete Scrum project management** single-page application built with **vanilla HTML5, CSS3 and JavaScript (ES6+)**. No frameworks, no build tools — just open `index.html` in a browser.

![Dark Theme](https://img.shields.io/badge/theme-dark%20%2F%20light-6366f1)
![Vanilla JS](https://img.shields.io/badge/vanilla-JS%20ES6%2B-f7df1e)
![localStorage](https://img.shields.io/badge/persistence-localStorage-10b981)

---

## Features

| Area | Details |
|---|---|
| **Dashboard** | KPI cards, active sprint progress, story breakdown, points donut, activity feed |
| **Board** | 4-column Kanban (To Do → In Progress → Review → Done), drag & drop, filters, WIP limits, DoD check |
| **Backlog** | Two-panel layout (product backlog ↔ sprint backlog), inline creation, grouping, sorting, bulk actions |
| **Sprints** | Create / start / complete sprints, retro form with export, burndown data |
| **Team** | Member cards with capacity sliders, workload indicators, velocity-per-member table |
| **Analytics** | Burndown (line), Velocity (bar), Distribution (doughnut), Epic progress (stacked bar), sprint comparison table, JSON export |
| **Epics** | CRUD, expandable cards, progress bars, story listing |
| **Modals** | Card detail (60/40 split), subtasks, tags, activity log, Fibonacci points, planning poker, standup helper, global search |
| **Advanced** | Dark / light theme, sidebar collapse, keyboard shortcuts, undo stack, autosave indicator, toast notifications |

---

## Quick Start

```bash
# No install needed — just open the file
open index.html          # macOS
xdg-open index.html      # Linux
start index.html         # Windows

# Or serve with any static server
npx serve .
python3 -m http.server 8000
```

On first load click **Demo Data** in the sidebar to populate a sample project.

---

## Keyboard Shortcuts

| Key | Action |
|---|---|
| `Ctrl + K` | Open global search |
| `N` | New story |
| `B` | Go to Board |
| `D` | Go to Dashboard |
| `Ctrl + Z` | Undo last action |
| `Ctrl + Enter` | Save card detail |
| `Escape` | Close any modal |

---

## File Structure

```
scrum-app/
├── index.html              # App shell + all modal markup
├── css/
│   ├── main.css            # Variables, reset, layout, sidebar, header, dashboard
│   ├── board.css           # Kanban board columns, cards, drag states
│   ├── backlog.css         # Backlog two-panel layout, story rows
│   ├── modals.css          # All modal overlays and forms
│   ├── charts.css          # Analytics grid, chart cards, team cards
│   └── animations.css      # All @keyframes
├── js/
│   ├── store.js            # localStorage CRUD singleton (Store)
│   ├── sampleData.js       # Demo data loader (SampleData)
│   ├── components.js       # Reusable render helpers (Components)
│   ├── board.js            # Board view (BoardView)
│   ├── backlog.js          # Backlog view (BacklogView)
│   ├── sprints.js          # Sprints view (SprintsView)
│   ├── team.js             # Team view (TeamView)
│   ├── analytics.js        # Analytics view (AnalyticsView)
│   ├── epics.js            # Epics view (EpicsView)
│   ├── modals.js           # All modal handlers (Modals)
│   └── app.js              # Router, init, sidebar, theme, shortcuts (App)
└── README.md
```

---

## How to Add a New Page / View

1. **Create `js/myview.js`** — IIFE module exporting a `render()` function:
   ```js
   const MyView = (() => {
     function render() {
       const el = document.getElementById('app-content');
       el.innerHTML = `<h2>My New Page</h2>`;
     }
     return { render };
   })();
   ```

2. **Add a `<script>` tag** in `index.html` before `modals.js`:
   ```html
   <script src="js/myview.js"></script>
   ```

3. **Add a nav link** in the sidebar `<nav>`:
   ```html
   <a href="#myview" class="nav-link" data-page="myview">
     <i class="fa-solid fa-star"></i><span>My View</span>
   </a>
   ```

4. **Register the route** in `app.js`:
   ```js
   const routes = {
     // ...existing routes...
     myview: { title: 'My View', render: () => MyView.render() }
   };
   ```

5. **(Optional)** Add a CSS file `css/myview.css` and link it in the `<head>`.

---

## How to Extend the Data Model

All data lives in **localStorage** via the `Store` module. To add a new entity (e.g. *Labels*):

1. Open `js/store.js` and add a key:
   ```js
   const KEYS = { ..., LABELS: 'scrumflow_labels' };
   ```

2. Add CRUD functions:
   ```js
   function getLabels() { return _get(KEYS.LABELS) || []; }
   function createLabel(data) {
     const labels = getLabels();
     labels.push({ id: _generateId(), ...data, createdAt: _now() });
     _set(KEYS.LABELS, labels);
     return labels[labels.length - 1];
   }
   ```

3. Expose in the `return { ... }` block.

4. Use in any view: `Store.getLabels()`, `Store.createLabel({ name: 'Bug', color: '#ef4444' })`.

---

## CDN Dependencies

| Library | Version | Purpose |
|---|---|---|
| [Chart.js](https://www.chartjs.org/) | 4.4.1 | Burndown, velocity, distribution, epic charts |
| [Font Awesome](https://fontawesome.com/) | 6.5.1 | Icons throughout the UI |
| [Google Fonts — Inter](https://fonts.google.com/specimen/Inter) | — | Typography |

---

## Browser Support

Any modern browser (Chrome, Firefox, Safari, Edge). Requires ES6+ support and `localStorage`.

---

## License

MIT
