/* ═══════════════════════════════════════════════════════════════════════════
   palette.js – Left panel: operator palette with search and categories.
   ═══════════════════════════════════════════════════════════════════════════ */
"use strict";

class PalettePanel {
  constructor(containerEl, searchInputEl, app) {
    this.container = containerEl;
    this.searchInput = searchInputEl;
    this.app = app;

    this.searchInput.addEventListener("input", () => this.render(this.searchInput.value));
    this.render();
  }

  render(filter = "") {
    this.container.innerHTML = "";
    const byCategory = operatorsByCategory();
    const filterLower = filter.toLowerCase();

    const categoryOrder = [
      OpCategory.DATA, OpCategory.TRANSFORM, OpCategory.FEATURE,
      OpCategory.MODEL, OpCategory.EVALUATION, OpCategory.VISUALIZATION,
      OpCategory.UTILITY,
    ];

    for (const cat of categoryOrder) {
      let ops = byCategory[cat] || [];
      if (filterLower) ops = ops.filter(n => n.toLowerCase().includes(filterLower));
      if (!ops.length) continue;

      const section = document.createElement("div");
      section.className = "palette-category";

      const header = document.createElement("div");
      header.className = "palette-category-header";
      header.innerHTML = `<span class="cat-dot" data-cat="${cat}"></span> ${cat} <span class="cat-count">(${ops.length})</span>`;
      header.addEventListener("click", () => {
        section.classList.toggle("collapsed");
      });
      section.appendChild(header);

      const list = document.createElement("div");
      list.className = "palette-items";
      for (const opName of ops) {
        const item = document.createElement("div");
        item.className = "palette-item";
        item.textContent = opName;
        item.draggable = true;
        item.addEventListener("dragstart", e => {
          e.dataTransfer.setData("text/plain", opName);
          e.dataTransfer.effectAllowed = "copy";
        });
        item.addEventListener("dblclick", () => {
          this.app.addOperatorToProcess(opName, 200 + Math.random() * 100, 150 + Math.random() * 100);
        });
        list.appendChild(item);
      }
      section.appendChild(list);
      this.container.appendChild(section);
    }
  }
}
