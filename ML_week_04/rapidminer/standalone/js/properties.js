/* ═══════════════════════════════════════════════════════════════════════════
   properties.js – Right panel: property inspector for selected operator.
   ═══════════════════════════════════════════════════════════════════════════ */
"use strict";

class PropertiesPanel {
  constructor(titleEl, propsEl, app) {
    this.titleEl = titleEl;
    this.propsEl = propsEl;
    this.app = app;
  }

  show(opId) {
    const op = this.app.process.operators[opId];
    if (!op) { this.clear(); return; }

    this.titleEl.textContent = op.name;
    this.propsEl.innerHTML = "";

    // Description
    if (op.description) {
      const desc = document.createElement("p");
      desc.className = "prop-desc";
      desc.textContent = op.description;
      this.propsEl.appendChild(desc);
    }

    // Category badge
    const badge = document.createElement("div");
    badge.className = "prop-badge";
    badge.innerHTML = `<span class="cat-dot" data-cat="${op.category}"></span> ${op.category}`;
    this.propsEl.appendChild(badge);

    // Ports info
    const portsInfo = document.createElement("div");
    portsInfo.className = "prop-ports";
    portsInfo.innerHTML = `<small>Inputs: ${op.inputs.map(p => p.name).join(", ") || "—"}<br>
                           Outputs: ${op.outputs.map(p => p.name).join(", ") || "—"}</small>`;
    this.propsEl.appendChild(portsInfo);

    if (op.paramSpecs.length === 0) {
      const noParams = document.createElement("p");
      noParams.className = "prop-empty";
      noParams.textContent = "No parameters.";
      this.propsEl.appendChild(noParams);
      return;
    }

    // Parameters
    for (const spec of op.paramSpecs) {
      if (spec.name.startsWith("_")) continue; // Hidden params

      const group = document.createElement("div");
      group.className = "prop-group";

      const label = document.createElement("label");
      label.textContent = spec.name;
      if (spec.description) label.title = spec.description;
      group.appendChild(label);

      let input;

      switch (spec.kind) {
        case ParamKind.BOOL:
          input = document.createElement("input");
          input.type = "checkbox";
          input.checked = !!op.params[spec.name];
          input.addEventListener("change", () => {
            op.params[spec.name] = input.checked;
            this.app.pushUndo();
          });
          break;

        case ParamKind.CHOICE:
          input = document.createElement("select");
          for (const ch of spec.choices) {
            const opt = document.createElement("option");
            opt.value = ch; opt.textContent = ch;
            if (ch === String(op.params[spec.name])) opt.selected = true;
            input.appendChild(opt);
          }
          input.addEventListener("change", () => {
            op.params[spec.name] = input.value;
            this.app.pushUndo();
          });
          break;

        case ParamKind.INT:
          input = document.createElement("input");
          input.type = "number"; input.step = "1";
          input.value = op.params[spec.name];
          input.addEventListener("change", () => {
            op.params[spec.name] = parseInt(input.value, 10);
            this.app.pushUndo();
          });
          break;

        case ParamKind.FLOAT:
          input = document.createElement("input");
          input.type = "number"; input.step = "0.01";
          input.value = op.params[spec.name];
          input.addEventListener("change", () => {
            op.params[spec.name] = parseFloat(input.value);
            this.app.pushUndo();
          });
          break;

        case ParamKind.COLUMN:
          input = document.createElement("input");
          input.type = "text";
          input.value = op.params[spec.name] || "";
          input.placeholder = "column name";
          input.addEventListener("change", () => {
            op.params[spec.name] = input.value;
            this.app.pushUndo();
          });
          break;

        default: // STRING
          input = document.createElement("input");
          input.type = "text";
          input.value = op.params[spec.name] || "";
          input.addEventListener("change", () => {
            op.params[spec.name] = input.value;
            this.app.pushUndo();
          });
          break;
      }
      group.appendChild(input);
      this.propsEl.appendChild(group);
    }

    // Delete button
    const delBtn = document.createElement("button");
    delBtn.className = "btn-delete";
    delBtn.textContent = "Delete Operator";
    delBtn.addEventListener("click", () => {
      this.app.process.removeOperator(opId);
      this.clear();
      this.app.canvasRenderer.selectedNode = null;
      this.app.canvasRenderer.draw();
      this.app.pushUndo();
    });
    this.propsEl.appendChild(delBtn);
  }

  clear() {
    this.titleEl.textContent = "Properties";
    this.propsEl.innerHTML = '<p class="prop-empty">Select an operator on the canvas to view its properties.</p>';
  }
}
