from __future__ import annotations

import html
import json
import re
import socket
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


ROWS = 13
COLS = 8
CELL_COUNT = ROWS * COLS

SAMPLE_ZERO = [
	[0, 0, 0, 1, 0, 0, 0, 0],
	[0, 1, 1, 1, 1, 1, 1, 0],
	[1, 1, 1, 0, 0, 1, 1, 1],
	[1, 1, 0, 0, 0, 0, 1, 1],
	[1, 1, 0, 0, 0, 0, 1, 1],
	[1, 1, 0, 0, 0, 0, 1, 1],
	[1, 1, 0, 0, 0, 0, 1, 1],
	[1, 1, 0, 0, 0, 0, 1, 1],
	[1, 1, 0, 0, 0, 0, 1, 1],
	[1, 1, 0, 0, 0, 0, 1, 1],
	[1, 1, 0, 0, 0, 0, 1, 1],
	[0, 1, 1, 1, 1, 1, 1, 0],
	[0, 0, 1, 1, 1, 1, 0, 0],
]


def parse_pattern(text: str) -> list[list[int]]:
	stripped = text.strip()
	if not stripped:
		raise ValueError(f"Input is empty. Paste {CELL_COUNT} binary values or {ROWS} lines of {COLS} digits.")

	line_candidates = [line.strip() for line in stripped.splitlines() if line.strip()]
	if len(line_candidates) == ROWS and all(re.fullmatch(rf"[01]{{{COLS}}}", line) for line in line_candidates):
		return [[int(char) for char in line] for line in line_candidates]

	tokens = re.findall(r"[01]", stripped)
	if len(tokens) != CELL_COUNT:
		raise ValueError(
			f"Pattern must contain exactly {CELL_COUNT} binary values for a {ROWS}x{COLS} image. "
			f"Received {len(tokens)} values."
		)

	values = [int(token) for token in tokens]
	return [values[index:index + COLS] for index in range(0, CELL_COUNT, COLS)]


def with_matrix_lines(grid: list[list[int]]) -> str:
	return "\n".join("".join(str(value) for value in row) for row in grid)


def free_port() -> int:
	with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
		sock.bind(("127.0.0.1", 0))
		return int(sock.getsockname()[1])


def build_html() -> str:
	sample_json = html.escape(json.dumps(SAMPLE_ZERO))
	initial_matrix = html.escape(with_matrix_lines(SAMPLE_ZERO))
	return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>13x8 Binary Digit Pattern Visualizer</title>
  <style>
    :root {{
      --bg: #f5f6fa;
      --card: #ffffff;
      --text: #13151a;
      --muted: #586173;
      --line: #d8deea;
      --accent: #2b6ef2;
      --pixel-on: #111111;
      --pixel-off: #ffffff;
      --shadow: 0 10px 30px rgba(15, 23, 42, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--text);
      background: var(--bg);
    }}
    .page {{ max-width: 1420px; margin: 0 auto; padding: 24px; }}
    h1 {{ margin: 0 0 8px; font-size: 2.3rem; }}
    p.lead {{ margin: 0 0 20px; color: var(--muted); line-height: 1.5; }}
    .layout {{ display: grid; grid-template-columns: minmax(380px, 1fr) minmax(420px, 1fr); gap: 24px; }}
    .card {{ background: var(--card); border: 1px solid var(--line); border-radius: 18px; box-shadow: var(--shadow); padding: 18px; }}
    .section-title {{ margin: 0 0 12px; font-size: 1.15rem; }}
    .meta {{ display: flex; align-items: center; gap: 10px; margin-bottom: 14px; flex-wrap: wrap; }}
    input[type="text"] {{ border: 1px solid var(--line); border-radius: 10px; padding: 10px 12px; font-size: 1rem; min-width: 90px; }}
    .grid {{ display: grid; grid-template-columns: repeat(8, 34px); gap: 3px; width: max-content; user-select: none; margin-bottom: 14px; }}
    .cell {{ width: 34px; height: 34px; border: 1px solid var(--line); background: var(--pixel-off); cursor: pointer; transition: transform 0.05s ease; }}
    .cell:active {{ transform: scale(0.96); }}
    .cell.on {{ background: var(--pixel-on); }}
    .toolbar {{ display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 14px; }}
    button {{ border: none; border-radius: 10px; padding: 10px 14px; font-size: 0.96rem; cursor: pointer; background: #eef3ff; color: #1746a2; font-weight: 600; }}
    button.primary {{ background: var(--accent); color: white; }}
    textarea {{ width: 100%; min-height: 220px; resize: vertical; border: 1px solid var(--line); border-radius: 12px; padding: 12px; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 0.98rem; line-height: 1.4; }}
    textarea.invalid {{ border-color: #d92d20; background: #fff6f5; }}
    .preview-shell {{ display: flex; justify-content: center; align-items: center; min-height: 460px; }}
    .preview-grid {{ display: grid; grid-template-columns: repeat(8, 28px); gap: 1px; background: #d6d6d6; padding: 8px; }}
    .preview-cell {{ width: 28px; height: 28px; background: var(--pixel-off); }}
    .preview-cell.on {{ background: var(--pixel-on); }}
    .summary textarea {{ min-height: 110px; }}
    .status {{ margin-top: 18px; padding: 12px 14px; background: #101827; color: #f8fafc; border-radius: 12px; font-size: 0.95rem; }}
    .note {{ color: var(--muted); font-size: 0.94rem; line-height: 1.45; }}
    @media (max-width: 1080px) {{ .layout {{ grid-template-columns: 1fr; }} .preview-shell {{ min-height: auto; }} }}
  </style>
</head>
<body>
  <div class="page">
    <h1>13x8 Binary Digit Pattern Visualizer</h1>
    <p class="lead">Edit a low-resolution digit pattern directly in the browser. Click cells to toggle pixels, paste a new pattern as 104 binary values or 13 lines of 8 digits, and copy the flattened output.</p>

    <div class="layout">
      <section class="card">
        <h2 class="section-title">Pattern editor</h2>
        <div class="meta">
          <label for="digitLabel"><strong>Label / digit:</strong></label>
          <input id="digitLabel" type="text" value="0" maxlength="12" />
          <span class="note">Image shape: 13 rows × 8 columns = 104 values.</span>
        </div>

        <div id="editorGrid" class="grid" aria-label="Editable binary grid"></div>

        <div class="toolbar">
          <button id="loadSampleBtn">Load sample 0</button>
          <button id="clearBtn">Clear</button>
          <button id="invertBtn">Invert</button>
          <button id="copyBtn">Copy flattened</button>
          <button id="applyBtn" class="primary">Apply pasted text</button>
        </div>

        <label class="section-title" for="patternInput">Paste / edit pattern text</label>
        <textarea id="patternInput">{initial_matrix}</textarea>
        <p class="note">Accepted formats:<br />- 13 lines like <code>00010000</code><br />- flattened binary values like <code>0, 0, 0, 1, ...</code></p>
      </section>

      <section class="card">
        <h2 class="section-title">Live visualization</h2>
        <p id="labelPreview"><strong>Label:</strong> 0</p>
        <div class="preview-shell">
          <div id="previewGrid" class="preview-grid" aria-label="Preview binary digit"></div>
        </div>
        <div class="summary">
          <h3 class="section-title">Flattened pattern</h3>
          <textarea id="flattenedOutput" readonly></textarea>
          <h3 class="section-title">Matrix view</h3>
          <textarea id="matrixOutput" readonly></textarea>
        </div>
      </section>
    </div>

    <div id="status" class="status">Ready.</div>
  </div>

  <script>
    const ROWS = {ROWS};
    const COLS = {COLS};
    const CELL_COUNT = ROWS * COLS;
    const SAMPLE_ZERO = JSON.parse('{sample_json}');

    const editorGrid = document.getElementById('editorGrid');
    const previewGrid = document.getElementById('previewGrid');
    const patternInput = document.getElementById('patternInput');
    const flattenedOutput = document.getElementById('flattenedOutput');
    const matrixOutput = document.getElementById('matrixOutput');
    const statusBox = document.getElementById('status');
    const digitLabel = document.getElementById('digitLabel');
    const labelPreview = document.getElementById('labelPreview');

    let grid = SAMPLE_ZERO.map(row => [...row]);
    let paintMode = null;

    function cloneGrid(source) {{
      return source.map(row => [...row]);
    }}

    function flatten(source) {{
      return source.flat();
    }}

    function matrixString(source) {{
      return source.map(row => row.join('')).join('\\n');
    }}

    function flattenedString(source) {{
      return flatten(source).join(', ');
    }}

    function setStatus(message) {{
      statusBox.textContent = message;
    }}

    function updateOutputs(syncInput = true) {{
      flattenedOutput.value = flattenedString(grid);
      matrixOutput.value = matrixString(grid);
      if (syncInput) {{
        patternInput.value = matrixString(grid);
      }}
      labelPreview.innerHTML = '<strong>Label:</strong> ' + (digitLabel.value.trim() || '?');
    }}

    function renderGrid(syncInput = true) {{
      const cells = editorGrid.querySelectorAll('.cell');
      const previewCells = previewGrid.querySelectorAll('.preview-cell');
      let index = 0;
      for (let row = 0; row < ROWS; row += 1) {{
        for (let col = 0; col < COLS; col += 1) {{
          const on = grid[row][col] === 1;
          cells[index].classList.toggle('on', on);
          previewCells[index].classList.toggle('on', on);
          index += 1;
        }}
      }}
      updateOutputs(syncInput);
    }}

    function createCells() {{
      for (let row = 0; row < ROWS; row += 1) {{
        for (let col = 0; col < COLS; col += 1) {{
          const cell = document.createElement('button');
          cell.type = 'button';
          cell.className = 'cell';
          cell.dataset.row = String(row);
          cell.dataset.col = String(col);
          cell.addEventListener('mousedown', () => {{
            const newValue = grid[row][col] === 1 ? 0 : 1;
            paintMode = newValue;
            grid[row][col] = newValue;
            renderGrid();
            setStatus(`Edited cell (${{row}}, ${{col}}) -> ${{newValue}}`);
          }});
          cell.addEventListener('mouseenter', () => {{
            if (paintMode === null) return;
            grid[row][col] = paintMode;
            renderGrid();
          }});
          editorGrid.appendChild(cell);

          const previewCell = document.createElement('div');
          previewCell.className = 'preview-cell';
          previewGrid.appendChild(previewCell);
        }}
      }}
      window.addEventListener('mouseup', () => {{
        paintMode = null;
      }});
    }}

    function parsePattern(text) {{
      const stripped = text.trim();
      if (!stripped) {{
        throw new Error(`Input is empty. Paste ${{CELL_COUNT}} binary values or ${{ROWS}} lines of ${{COLS}} digits.`);
      }}
      const lines = stripped.split(/\\r?\\n/).map(line => line.trim()).filter(Boolean);
      if (lines.length === ROWS && lines.every(line => new RegExp(`^[01]{{${{COLS}}}}$`).test(line))) {{
        return lines.map(line => Array.from(line).map(char => Number(char)));
      }}
      const tokens = stripped.match(/[01]/g) || [];
      if (tokens.length !== CELL_COUNT) {{
        throw new Error(`Pattern must contain exactly ${{CELL_COUNT}} binary values. Received ${{tokens.length}}.`);
      }}
      const values = tokens.map(Number);
      const parsed = [];
      for (let index = 0; index < values.length; index += COLS) {{
        parsed.push(values.slice(index, index + COLS));
      }}
      return parsed;
    }}

    document.getElementById('loadSampleBtn').addEventListener('click', () => {{
      grid = cloneGrid(SAMPLE_ZERO);
      digitLabel.value = '0';
      renderGrid();
      setStatus('Loaded sample digit 0.');
    }});

    document.getElementById('clearBtn').addEventListener('click', () => {{
      grid = Array.from({{ length: ROWS }}, () => Array.from({{ length: COLS }}, () => 0));
      renderGrid();
      setStatus('Cleared the grid.');
    }});

    document.getElementById('invertBtn').addEventListener('click', () => {{
      grid = grid.map(row => row.map(value => value === 1 ? 0 : 1));
      renderGrid();
      setStatus('Inverted all pixels.');
    }});

    document.getElementById('copyBtn').addEventListener('click', async () => {{
      const value = flattenedString(grid);
      try {{
        await navigator.clipboard.writeText(value);
        setStatus('Copied flattened pattern to clipboard.');
      }} catch (_error) {{
        flattenedOutput.focus();
        flattenedOutput.select();
        setStatus('Clipboard access was blocked. The flattened output is selected for manual copy.');
      }}
    }});

    document.getElementById('applyBtn').addEventListener('click', () => {{
      try {{
        grid = parsePattern(patternInput.value);
        patternInput.classList.remove('invalid');
        renderGrid(true);
        setStatus('Applied pattern from text input.');
      }} catch (error) {{
        patternInput.classList.add('invalid');
        alert(error.message);
        setStatus(error.message);
      }}
    }});

    patternInput.addEventListener('input', () => {{
      try {{
        grid = parsePattern(patternInput.value);
        patternInput.classList.remove('invalid');
        renderGrid(false);
        setStatus('Live preview updated from text input.');
      }} catch (_error) {{
        patternInput.classList.add('invalid');
      }}
    }});

    digitLabel.addEventListener('input', () => {{
      labelPreview.innerHTML = '<strong>Label:</strong> ' + (digitLabel.value.trim() || '?');
    }});

    createCells();
    renderGrid();
    setStatus(`Ready. Pattern size: ${{ROWS}}x${{COLS}} = ${{CELL_COUNT}} values. Click pixels or paste values.`);
  </script>
</body>
</html>
"""


HTML_PAGE = build_html().encode("utf-8")


class PatternHandler(BaseHTTPRequestHandler):
	def do_GET(self) -> None:
		if self.path not in {"/", "/index.html"}:
			self.send_error(404, "Not found")
			return
		self.send_response(200)
		self.send_header("Content-Type", "text/html; charset=utf-8")
		self.send_header("Content-Length", str(len(HTML_PAGE)))
		self.end_headers()
		self.wfile.write(HTML_PAGE)

	def log_message(self, _format: str, *_args: object) -> None:
		return


def main() -> None:
	port = free_port()
	server = ThreadingHTTPServer(("127.0.0.1", port), PatternHandler)
	url = f"http://127.0.0.1:{port}/"
	print("Starting binary digit pattern visualizer...")
	print(f"Open in browser: {url}")

	thread = threading.Thread(target=server.serve_forever, daemon=True)
	thread.start()

	try:
		webbrowser.open(url)
		server.serve_forever()
	except KeyboardInterrupt:
		print("\nStopping server...")
	finally:
		server.shutdown()
		server.server_close()


if __name__ == "__main__":
	main()
