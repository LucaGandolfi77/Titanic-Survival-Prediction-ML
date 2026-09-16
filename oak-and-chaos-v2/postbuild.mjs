import { readFile, writeFile, readdir } from 'fs/promises';
import { join } from 'path';

const BUILD = './build';
const exts = ['.js', '.d.ts'];

async function process(dir) {
  const entries = await readdir(dir, { withFileTypes: true });
  for (const entry of entries) {
    const full = join(dir, entry.name);
    if (entry.isDirectory()) { await process(full); continue; }
    if (entry.isFile() && exts.some(e => entry.name.endsWith(e))) {
    if (entry.name.endsWith('.d.ts')) continue;
      let src = await readFile(full, 'utf-8');
      const before = src;
      src = src.replace(/from\s+['"](\.[^'"]+)['"]/g, (_, path) => {
        if (path.endsWith('.js') || path.endsWith('.ts') || path.endsWith('.d.ts')) return `from '${path}'`;
        return `from '${path}.js'`;
      });
      src = src.replace(/import\(['"]([^'"]+)['"]\)/g, (_, path) => {
        if (path.endsWith('.js')) return `import('${path}')`;
        return `import('${path}.js')`;
      });
      if (src !== before) { await writeFile(full, src); console.log(`Fixed: ${full}`); }
    }
  }
}

await process(BUILD);
