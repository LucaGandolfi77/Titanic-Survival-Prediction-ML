#!/usr/bin/env python3
import re
from pathlib import Path

repo = Path(__file__).resolve().parents[1]
js_db = repo / 'js' / 'database.js'
import_txt = repo / 'assets' / 'import_cards.txt'
output_js = repo / 'scripts' / 'new_cards_snippet.js'

text = import_txt.read_text(encoding='utf-8')
raw_db = js_db.read_text(encoding='utf-8')

# find highest existing card id
ids = re.findall(r'card_(\d{3})', raw_db)
max_id = max([int(i) for i in ids]) if ids else 0
next_id = max_id + 1

# Better splitting: detect card header lines (name) followed by a mana or type line
lines = [l.rstrip() for l in text.splitlines()]
headers = []
for i, ln in enumerate(lines[:-1]):
    nxt = lines[i+1]
    # mana-like line contains braces e.g. {2}{W} or starts with { or contains X
    is_mana = bool(re.search(r"\{.*?\}", nxt))
    is_type = any(k in nxt for k in ("Creature","Enchantment","Artifact","Instant","Sorcery","Land"))
    # treat this line as a name/header if next line looks like mana or a type line
    # ensure the current line itself is not a mana or type line (avoid double-headers)
    is_ln_mana = bool(re.search(r"\{.*?\}", ln))
    is_ln_type = any(k in ln for k in ("Creature","Enchantment","Artifact","Instant","Sorcery","Land"))
    if ln.strip() and not (is_ln_mana or is_ln_type) and (is_mana or is_type):
        headers.append(i)

# If no headers found, fall back to simple blank-line splitting
if not headers:
    blocks = [b.strip() for b in re.split(r"\n\s*\n+", text) if b.strip()]
else:
    blocks = []
    for idx, start in enumerate(headers):
        end = headers[idx+1] if idx+1 < len(headers) else len(lines)
        block = '\n'.join(lines[start:end]).strip()
        if block:
            blocks.append(block)

objs = []
for block in blocks:
    lines = [l.strip() for l in block.splitlines() if l.strip()]
    if not lines:
        continue
    name = lines[0]
    mana = None
    type_line = None
    power = None
    toughness = None
    desc_lines = []
    # look for mana line in next few lines
    idx = 1
    if idx < len(lines) and '{' in lines[idx]:
        mana = lines[idx]
        idx += 1
    # type line
    if idx < len(lines) and ("Creature" in lines[idx] or "Enchantment" in lines[idx] or "Artifact" in lines[idx] or "Instant" in lines[idx] or "Sorcery" in lines[idx] or "Land" in lines[idx]):
        type_line = lines[idx]
        idx += 1
    # remaining lines — scan for final power/toughness
    for j in range(idx, len(lines)):
        m = re.match(r'^(\d+|0)\/(\d+|0)$', lines[j])
        if m:
            power = int(m.group(1))
            toughness = int(m.group(2))
        else:
            desc_lines.append(lines[j])
    desc = ' '.join(desc_lines).strip() if desc_lines else None
    obj = {
        'name': name,
        'cost': mana,
        'type': type_line,
        'power': power,
        'defense': toughness,
        'description': desc,
    }
    objs.append(obj)

# generate JS snippet (array of objects without surrounding brackets)
lines_out = []
for o in objs:
    cid = f"card_{next_id:03d}"
    next_id += 1
    name = o['name'].replace('"', '\\"')
    typev = (o['type'] or '').replace('"', '\\"')
    cost = o['cost'] or None
    desc = (o['description'] or '').replace('"', '\\"')
    power = o['power']
    defense = o['defense']
    img = f"assets/cards/{cid}.jpg"
    tags = []
    if typev:
        tags = [t.lower() for t in re.findall(r"[A-Za-z']+", typev)][:3]
    obj_lines = []
    obj_lines.append('  { id: "%s", name: "%s", type: "%s", cost: %s, power: %s, defense: %s, rarity: null, element: null, description: "%s", image: "%s", tags: [%s] },' % (
        cid, name, typev, ('"%s"' % cost) if cost else 'null', str(power) if power is not None else 'null', str(defense) if defense is not None else 'null', desc or '', img, ','.join(['"%s"' % t for t in tags])
    ))
    lines_out.extend(obj_lines)

output_js.write_text('\n'.join(lines_out), encoding='utf-8')
print(f'Wrote {len(objs)} objects to {output_js} (starting id was {max_id+1}).')
