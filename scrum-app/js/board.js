/* =====================================================
   BOARD.JS — Board view + drag & drop logic
   ===================================================== */

const BoardView = (() => {
  const COLUMNS = ['todo', 'inprogress', 'review', 'done'];
  const COLUMN_LABELS = { todo: 'To Do', inprogress: 'In Progress', review: 'In Review', done: 'Done' };
  let currentFilters = { assignee: '', priority: '', epic: '', search: '' };

  function render() {
    const project = Store.getActiveProject();
    const sprint = Store.getActiveSprint(project);
    const content = document.getElementById('app-content');

    if (!project || !sprint) {
      content.innerHTML = `
        <div class="empty-state">
          <i class="fa-solid fa-columns"></i>
          <p>No active sprint. Go to <a href="#sprints" style="color:var(--accent)">Sprints</a> to start one.</p>
        </div>`;
      return;
    }

    const stories = sprint.stories || [];
    const wipLimits = Store.getWipLimits();

    content.innerHTML = `
      <div class="page-enter">
        ${sprint.goal ? `<div class="sprint-goal-banner"><i class="fa-solid fa-bullseye"></i> <strong>Sprint Goal:</strong> ${Components._esc(sprint.goal)}</div>` : ''}
        <div class="board-toolbar">
          <div class="board-toolbar-left">
            <select id="board-sprint-selector" class="filter-bar">
              ${(project.sprints || []).map(s => `<option value="${s.id}" ${s.id === sprint.id ? 'selected' : ''}>${s.name} ${s.status === 'active' ? '(Active)' : ''}</option>`).join('')}
            </select>
            <button class="btn btn-accent" id="btn-board-add-story"><i class="fa-solid fa-plus"></i> Add Story</button>
            <button class="btn btn-outline" id="btn-standup"><i class="fa-solid fa-sun"></i> Standup</button>
          </div>
          <div class="board-toolbar-right filter-bar">
            <select id="filter-assignee"><option value="">All Assignees</option>${(project.teamMembers || []).map(m => `<option value="${m.id}">${m.name}</option>`).join('')}</select>
            <select id="filter-priority"><option value="">All Priorities</option><option value="critical">Critical</option><option value="high">High</option><option value="medium">Medium</option><option value="low">Low</option></select>
            <select id="filter-epic"><option value="">All Epics</option>${(project.epics || []).map(e => `<option value="${e.id}">${e.name}</option>`).join('')}</select>
            <input type="text" id="filter-search" placeholder="Search..." style="width:140px">
          </div>
        </div>
        <div class="kanban-board" id="kanban-board">
          ${COLUMNS.map(col => {
            const colStories = _filterStories(stories.filter(s => s.status === col));
            const totalPts = colStories.reduce((s, st) => s + (st.storyPoints || 0), 0);
            const wipLimit = wipLimits[col] || 0;
            const exceeded = wipLimit > 0 && colStories.length > wipLimit;
            return `
              <div class="kanban-column ${exceeded ? 'wip-exceeded' : ''}" data-status="${col}" id="column-${col}">
                <div class="column-header">
                  <span class="column-title"><span class="column-dot ${col}"></span>${COLUMN_LABELS[col]}</span>
                  <span class="column-count">
                    <span class="count-badge">${colStories.length}</span>
                    <span>${totalPts} pts</span>
                    ${wipLimit > 0 ? `<span class="wip-warning">WIP: ${colStories.length}/${wipLimit}</span>` : ''}
                  </span>
                </div>
                <div class="column-cards" data-status="${col}">
                  ${colStories.length === 0 ? '<div class="column-empty">No stories</div>' : ''}
                  ${colStories.map(s => Components.storyCard(s, project)).join('')}
                </div>
              </div>`;
          }).join('')}
        </div>
      </div>`;

    _bindEvents(project, sprint);
  }

  function _filterStories(stories) {
    return stories.filter(s => {
      if (currentFilters.assignee && s.assigneeId !== currentFilters.assignee) return false;
      if (currentFilters.priority && s.priority !== currentFilters.priority) return false;
      if (currentFilters.epic && s.epicId !== currentFilters.epic) return false;
      if (currentFilters.search) {
        const q = currentFilters.search.toLowerCase();
        if (!s.title.toLowerCase().includes(q) && !(s.description || '').toLowerCase().includes(q)) return false;
      }
      return true;
    });
  }

  function _bindEvents(project, sprint) {
    // Sprint selector
    const sprintSel = document.getElementById('board-sprint-selector');
    if (sprintSel) {
      sprintSel.addEventListener('change', (e) => {
        // Just re-render with the selected sprint view
        render();
      });
    }

    // Add story button
    const addBtn = document.getElementById('btn-board-add-story');
    if (addBtn) {
      addBtn.addEventListener('click', () => {
        Modals.openCardDetail(null, sprint.id);
      });
    }

    // Standup button
    const standupBtn = document.getElementById('btn-standup');
    if (standupBtn) {
      standupBtn.addEventListener('click', () => Modals.openStandup());
    }

    // Filters
    ['filter-assignee', 'filter-priority', 'filter-epic'].forEach(id => {
      const el = document.getElementById(id);
      if (el) {
        el.addEventListener('change', (e) => {
          currentFilters[id.replace('filter-', '')] = e.target.value;
          render();
        });
      }
    });

    const searchInput = document.getElementById('filter-search');
    if (searchInput) {
      searchInput.addEventListener('input', (e) => {
        currentFilters.search = e.target.value;
        render();
      });
    }

    // Card clicks
    document.querySelectorAll('.story-card').forEach(card => {
      card.addEventListener('click', (e) => {
        if (e.target.closest('.backlog-story-actions')) return;
        const storyId = card.dataset.storyId;
        Modals.openCardDetail(storyId);
      });
    });

    // Drag & Drop
    _setupDragDrop(project);
  }

  function _setupDragDrop(project) {
    const cards = document.querySelectorAll('.story-card');
    const columns = document.querySelectorAll('.column-cards');

    cards.forEach(card => {
      card.addEventListener('dragstart', (e) => {
        card.classList.add('dragging');
        e.dataTransfer.setData('text/plain', card.dataset.storyId);
        e.dataTransfer.effectAllowed = 'move';
      });

      card.addEventListener('dragend', () => {
        card.classList.remove('dragging');
        document.querySelectorAll('.kanban-column').forEach(c => c.classList.remove('drag-over'));
      });
    });

    columns.forEach(column => {
      column.addEventListener('dragover', (e) => {
        e.preventDefault();
        e.dataTransfer.dropEffect = 'move';
        column.closest('.kanban-column').classList.add('drag-over');
      });

      column.addEventListener('dragleave', (e) => {
        if (!column.contains(e.relatedTarget)) {
          column.closest('.kanban-column').classList.remove('drag-over');
        }
      });

      column.addEventListener('drop', (e) => {
        e.preventDefault();
        const storyId = e.dataTransfer.getData('text/plain');
        const newStatus = column.dataset.status;

        column.closest('.kanban-column').classList.remove('drag-over');

        if (!storyId || !newStatus) return;

        const story = Store.getStoryById(storyId, project);
        if (!story) return;

        // Check DoD for "done"
        if (newStatus === 'done' && story.status !== 'done') {
          _checkDoD(storyId, newStatus, project);
          return;
        }

        // Save undo
        Store.pushUndo({ type: 'moveStatus', storyId, oldStatus: story.status, newStatus });

        Store.moveStoryToStatus(storyId, newStatus, project);
        Store.updateBurndown(Store.getActiveSprint(project)?.id, project);
        Components.showToast(`"${story.title}" moved to ${COLUMN_LABELS[newStatus]}`, 'success');
        Components.showSaved();
        render();
      });
    });
  }

  function _checkDoD(storyId, newStatus, project) {
    const dod = Store.getDoDChecklist();
    const story = Store.getStoryById(storyId, project);

    const dodContent = document.getElementById('dod-content');
    dodContent.innerHTML = `
      <p style="margin-bottom:1rem;color:var(--text-muted)">Please verify the Definition of Done for "<strong>${Components._esc(story.title)}</strong>":</p>
      ${dod.map((item, i) => `
        <div class="dod-item">
          <input type="checkbox" id="dod-check-${i}" class="dod-checkbox">
          <label for="dod-check-${i}">${Components._esc(item)}</label>
        </div>
      `).join('')}
      <button id="btn-dod-confirm" class="btn btn-success btn-full" style="margin-top:1rem" disabled>
        <i class="fa-solid fa-check"></i> Mark as Done
      </button>
    `;

    document.getElementById('modal-dod').classList.remove('hidden');

    // Enable button when all checked
    dodContent.addEventListener('change', () => {
      const allChecked = dodContent.querySelectorAll('.dod-checkbox:not(:checked)').length === 0;
      document.getElementById('btn-dod-confirm').disabled = !allChecked;
    });

    document.getElementById('btn-dod-confirm').addEventListener('click', () => {
      Store.pushUndo({ type: 'moveStatus', storyId, oldStatus: story.status, newStatus });
      Store.moveStoryToStatus(storyId, newStatus, project);
      Store.updateBurndown(Store.getActiveSprint(project)?.id, project);
      Components.showToast(`"${story.title}" marked as Done!`, 'success');
      Components.showSaved();
      document.getElementById('modal-dod').classList.add('hidden');
      render();
    });
  }

  return { render };
})();
