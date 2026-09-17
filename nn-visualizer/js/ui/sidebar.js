export function renderSidebarList(nodesListEl, model, positions, onNodeSelect) {
  nodesListEl.innerHTML = '';
  for (let i = 0; i < model.layers.length; i++) {
    const header = document.createElement('div');
    header.textContent = `Layer ${i} (${model.layers[i].type} — ${model.layers[i].size})`;
    header.style.fontWeight = '600';
    header.style.fontSize = '13px';
    header.style.marginTop = i > 0 ? '8px' : '0';
    nodesListEl.appendChild(header);

    for (let n = 0; n < model.layers[i].size; n++) {
      const item = document.createElement('div');
      item.className = 'node-item';
      item.textContent = `Node ${i}:${n}`;
      item.dataset.layer = i;
      item.dataset.idx = n;
      if (model.layers[i].size <= 20) {
        item.onclick = () => {
          if (onNodeSelect) onNodeSelect(i, n);
        };
      }
      nodesListEl.appendChild(item);
    }
  }
}

export function updateNodeDetail(detailEl, html) {
  detailEl.innerHTML = html;
}

export function clearAllSelections(nodesListEl) {
  nodesListEl.querySelectorAll('.node-item').forEach((item) => item.classList.remove('selected'));
}
