# ML Patterns Visualizer

Small local tool for editing and visualizing low-resolution binary digit images.

## Shape

- image size: 13 rows x 8 columns
- total values: 104
- each cell must be `0` or `1`

## Important note

This version does **not** use `tkinter` anymore.
It runs as a tiny local web app served by Python's standard library, so you do not need to install Tk.

## Features

- click cells to toggle pixels
- drag to paint multiple cells
- paste flattened binary values
- paste 13 text rows like `00010000`
- copy flattened pattern back out
- clear, invert, and load a sample digit
- live large preview in the browser

## Run

Use your normal Python interpreter:

```bash
cd /Users/jessicabottarelli/Desktop/Gandalf/Github/random-1/ml-patterns
python visualize.py
```

If `python` is not available, use:

```bash
python3 visualize.py
```

The script starts a local HTTP server and prints a URL like:

```text
http://127.0.0.1:54321/
```

It will also try to open your browser automatically.

Stop it with `Ctrl+C` in the terminal.

## Accepted input formats

### Flattened

```text
0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, ...
```

### Matrix

```text
00010000
01111110
11100111
11000011
11000011
11000011
11000011
11000011
11000011
11000011
11000011
01111110
00111100
```
