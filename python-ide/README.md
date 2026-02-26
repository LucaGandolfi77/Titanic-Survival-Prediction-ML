# 🐍 Python IDE — Browser-Based Development Environment

A fully-featured Python IDE that runs **100% in the browser** with **no backend** and **no server** required. Powered by [Pyodide](https://pyodide.org/) (CPython 3.12 compiled to WebAssembly) and custom syntax highlighting.

---

## ✨ Features

### Code Editor
- Python syntax highlighting with full tokenizer
- Multi-tab editing with dirty indicators
- Line numbers with clickable breakpoint gutter
- Auto-indent (4 spaces, PEP 8)
- Toggle comments (`Ctrl+/`)
- Duplicate lines (`Ctrl+D`)
- Go to line (`Ctrl+G`)
- Tab/Shift+Tab indentation for selections

### Python Execution
- **Full CPython 3.12** via Pyodide WebAssembly
- Runs in a **Web Worker** (off main thread — no UI freezing)
- Supports `numpy`, `pandas`, `matplotlib`, `scipy`, `scikit-learn`, `sympy`, and 100+ packages
- Auto-detects and installs missing imports
- REPL input with `>>>` prompt
- Execution timing display

### Output Panel (4 tabs)
- **Console**: Colored output (stdout, stderr, results, info)
- **Plots**: Gallery of matplotlib figures with zoom and download
- **Problems**: Errors with clickable line numbers
- **History**: Execution history with re-run and copy

### File Explorer
- IndexedDB-backed virtual filesystem (persists across sessions)
- Tree view with file type icons
- Right-click context menu (New/Rename/Delete/Download/Copy Path)
- Drag-and-drop file organization
- File upload support

### Variable Inspector
- Auto-refreshes after each execution
- Shows Name, Type, Value, Shape
- Click DataFrames for full viewer modal
- Click arrays/dicts/lists for detailed exploration

### Package Manager
- Search and install Python packages
- Browse available packages
- `requirements.txt` bulk install
- Installation progress log

### Cell Mode (Jupyter-style)
- `# %%` delimiters to split code into cells
- Per-cell run buttons
- Run all cells or cells above
- Visual cell markers

### Simple Debugger
- Breakpoints via gutter click or F9
- Call stack viewer
- Local variables per frame
- Start/Stop/Continue/Step controls

### Command Palette
- `Ctrl+Shift+P` to open
- Fuzzy search across all commands
- Keyboard navigable
- Shows shortcuts

### 7 Themes
- **Dark** (Catppuccin Mocha) — default
- **Dracula**
- **Monokai**
- **Nord**
- **Solarized Light**
- **GitHub Light**
- **One Dark**

### 12 Code Templates
Hello World, Data Analysis, Matplotlib, Machine Learning, Fibonacci, Web Scraping, File I/O, OOP, Async/Await, NumPy, Sorting Algorithms, Prime Sieve

---

## 🚀 How to Run

### Simple — Just open the HTML file:
```bash
# Using Python's built-in server (recommended for Web Worker support):
cd python-ide
python3 -m http.server 8080

# Then open: http://localhost:8080
```

### Or with any static file server:
```bash
npx serve .
# or
npx http-server .
```

> **Note:** Opening `index.html` directly via `file://` protocol will NOT work because Web Workers and ES module imports require HTTP(S).

### Cross-Origin Headers (optional, for SharedArrayBuffer)
If you need `SharedArrayBuffer` support (advanced features), serve with these headers:
```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

---

## 📁 Project Structure

```
python-ide/
├── index.html              # Main HTML shell
├── README.md               # This file
├── css/
│   ├── main.css            # Layout, reset, sidebar, splitters, toasts
│   ├── editor.css          # Tabs, gutter, syntax highlighting, cells
│   ├── output.css          # Console, plots, problems, history, REPL
│   ├── filetree.css        # File tree, outline, context menu
│   ├── modals.css          # Settings, packages, command palette, templates
│   └── splash.css          # Loading splash screen & animations
├── js/
│   ├── utils.js            # Event bus, helpers, splitter, toast, fuzzy match
│   ├── themes.js           # 7 theme definitions with CSS custom properties
│   ├── settings.js         # Settings modal with localStorage persistence
│   ├── templates.js        # 12 Python code templates with picker UI
│   ├── pythonEngine.js     # Main-thread bridge to Pyodide Web Worker
│   ├── editor.js           # Code editor: tabs, syntax highlighting, keys
│   ├── outputPanel.js      # Multi-tab output: console, plots, problems
│   ├── fileSystem.js       # IndexedDB virtual filesystem
│   ├── fileExplorer.js     # File tree UI, context menu, drag-drop
│   ├── variableInspector.js# Variable table, DataFrame viewer
│   ├── debugger.js         # Breakpoints, call stack, step controls
│   ├── packageManager.js   # Package search, install, requirements.txt
│   ├── cellMode.js         # Jupyter-style # %% cell execution
│   ├── commandPalette.js   # Ctrl+Shift+P command palette
│   ├── outline.js          # Parse & display Python symbols
│   └── app.js              # Bootstrap, toolbar wiring, run logic
└── workers/
    └── pyWorker.js         # Pyodide Web Worker (Python execution)
```

---

## ⌨️ Keyboard Shortcuts

| Shortcut | Action |
|---|---|
| `Ctrl+Enter` | Run file |
| `Shift+Enter` | Run selection |
| `Ctrl+S` | Save file |
| `Ctrl+/` | Toggle comment |
| `Ctrl+D` | Duplicate line |
| `Ctrl+G` | Go to line |
| `Ctrl+B` | Toggle sidebar |
| `Ctrl+,` | Open settings |
| `Ctrl+Shift+P` | Command palette |
| `F5` | Run / Start debug |
| `F9` | Toggle breakpoint |
| `Tab` | Indent |
| `Shift+Tab` | Dedent |

---

## 📦 Adding Packages

1. Click the **cube icon** in the toolbar (or use Command Palette → Package Manager)
2. Search for a package name
3. Click Install

Or use `requirements.txt` tab in the Package Manager to install multiple packages at once.

Packages are installed via **micropip** (Pyodide's package manager). Most pure-Python packages from PyPI work. C-extension packages must be pre-built for WebAssembly — Pyodide includes 100+ popular ones.

---

## 🎨 Adding Themes

Edit `js/themes.js` and add a new theme object to the `THEMES` object. Each theme needs ~50 CSS custom properties. See the existing themes for reference.

---

## ⚠️ Known Limitations

- **No real filesystem**: Files are stored in IndexedDB (browser storage). They persist across page reloads but not across browsers.
- **No networking**: Python's `requests`, `urllib` etc. don't have real network access in WebAssembly. Use mocked data instead.
- **No `input()` blocking**: Python's `input()` function can't block the Web Worker. Use the REPL instead.
- **Large packages**: Some scientific packages (e.g., TensorFlow, PyTorch) are not available in Pyodide.
- **Performance**: WebAssembly is ~2-5x slower than native CPython for CPU-intensive tasks.
- **Memory**: Browser tab memory is limited (~2-4 GB typical).

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Python Runtime | Pyodide 0.26.x (CPython 3.12 → WebAssembly) |
| Code Editor | Custom textarea with Python tokenizer |
| Icons | Font Awesome 6 |
| Fonts | JetBrains Mono (code), Inter (UI) |
| Storage | IndexedDB (files), localStorage (settings) |
| Execution | Web Workers (off main thread) |
| Default Theme | Catppuccin Mocha |

---

## License

MIT
