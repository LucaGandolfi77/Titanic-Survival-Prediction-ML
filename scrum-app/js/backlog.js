/* =====================================================
   BACKLOG.JS — Backlog view + drag to sprint
   ===================================================== */

const BacklogView = (() => {
  let selectedStories = new Set();
  let groupBy = 'none';
  let sortBy = 'priority';
  let selectedSprintId = null;

  function render() {
    const project = Store.getActiveProject();
    const content = document.getElementById('app-content');

    if (!project) {
      content.innerHTML = '<div class="empty-state"><i class="fa-solid fa-list-check"></i><p>No project selected.</p></div>';
      return;
    }

    const sprints = Store.getSprints(project).filter(s => s.status !== 'completed');
    const activeSprint = Store.getActiveSprint(project);
    if (!selectedSprintId && activeSprint) selectedSprintId = activeSprint.id;
    if (!selectedSprintId && sprints.length > 0) selectedSprintId = sprints[0].id;

    const backlogStories = _sortStories(project.backlog || []);
    const selectedSprint = sprints.find(s => s.id === selectedSprintId);
    const sprintStories = selectedSprint ? _sortStories(selectedSprint.stories || []) : [];

    const backlogPoints = backlogStories.reduce((s, st) => s + (st.storyPoints || 0), 0);
    const sprintPoints = sprintStories.reduce((s, st) => s + (st.storyPoints || 0), 0);

    content.innerHTML = `
      <div class="page-enter">
        <div class="page-header">
          <h2>Backlog</h2>
          <div class="page-header-actions">
            <button id="btn-poker" class="btn btn-outline"><i class="fa-solid fa-cards"></i> Planning Poker</button>
          </div>
        </div>
        <div class="backlog-layout">
          <!-- LEFT: Product Backlog -->
          <div class="backlog-panel">
            <div class="backlog-panel-header">
              <div class="backlog-panel-title"><i class="fa-solid fa-inbox"></i> Product Backlog</div>
              <div class="backlog-panel-stats">
                <span>${backlogStories.length} stories</span>
                <span>${backlogPoints} pts</span>
              </div>
            </div>
            <!-- Inline Create -->
            <div class="inline-create-form">
              <input type="text" id="backlog-new-title" placeholder="Story title...">
              <input type="number" id="backlog-new-points" placeholder="Pts" min="0" style="width:60px">
              <select id="backlog-new-priority">
                <option value="medium">Medium</option>
                <option value="critical">Critical</option>
                <option value="high">High</option>
                <option value="low">Low</option>
              </select>
              <button class="btn btn-accent btn-sm" id="btn-backlog-create"><i class="fa-solid fa-plus"></i></button>
            </div>
            <!-- Group / Sort bar -->
            <div class="backlog-group-bar">
              <label>Group:</label>
              <button class="group-btn ${groupBy === 'none' ? 'active' : ''}" data-group="none">None</button>
              <button class="group-btn ${groupBy === 'epic' ? 'active' : ''}" data-group="epic">Epic</button>
              <button class="group-btn ${groupBy === 'priority' ? 'active' : ''}" data-group="priority">Priority</button>
              <span style="margin-left:auto"></span>
              <label>Sort:</label>
              <button class="sort-btn ${sortBy === 'priority' ? 'active' : ''}" data-sort="priority">Priority</button>
              <button class="sort-btn ${sortBy === 'points' ? 'active' : ''}" data-sort="points">Points</button>
              <button class="sort-btn ${sortBy === 'created' ? 'active' : ''}" data-sort="created">Created</button>
            </div>
            ${selectedStories.size > 0 ? `
              <div class="bulk-actions-bar">
                <span class="selected-count">${selectedStories.size} selected</span>
                ${selectedSprintId ? `<button class="btn-sm btn-accent" id="btn-bulk-move-sprint"><i class="fa-solid fa-arrow-right"></i> Move to Sprint</button>` : ''}
                <button class="btn-sm" id="btn-bulk-delete" style="color:var(--danger)"><i class="fa-solid fa-trash"></i> Delete</button>
                <button class="btn-sm" id="btn-bulk-clear" style="margin-left:auto">Clear</button>
              </div>
            ` : ''}
            <div class="backlog-list" id="backlog-list" data-target="backlog">
              ${_renderGroupedStories(backlogStories, project)}
            </div>
          </div>

          <!-- RIGHT: Sprint Backlog -->
          <div class="backlog-panel">
            <div class="backlog-panel-header">
              <div class="backlog-panel-title"><i class="fa-solid fa-rocket"></i> Sprint Backlog</div>
              <div class="backlog-panel-actions">
                <select id="backlog-sprint-selector">
                  ${sprints.map(s => `<option value="${s.id}" ${s.id === selectedSprintId ? 'selected' : ''}>${s.name} ${s.status === 'active' ? '⚡' : ''}</option>`).join('')}
                </select>
              </div>
            </div>
            <div class="backlog-panel-header" style="border-bottom:1px solid var(--border);padding:0.5rem 1.25rem">
              <div class="backlog-panel-stats">
                <span>${sprintStories.length} stories</span>
                <span>${sprintPoints}${selectedSprint ? '/' + selectedSprint.capacity : ''} pts</span>
              </div>
              ${selectedSprint ? `<div>${Components.progressBar(selectedSprint.capacity > 0 ? (sprintPoints / selectedSprint.capacity * 100) : 0)}</div>` : ''}
            </div>
            <div class="backlog-list" id="sprint-backlog-list" data-target="sprint" data-sprint-id="${selectedSprintId || ''}">
              ${sprintStories.length === 0 ? '<div class="empty-state" style="padding:2rem"><i class="fa-solid fa-inbox" style="font-size:2rem"></i><p>Drag stories here</p></div>' : ''}
              ${sprintStories.map(s => Components.backlogStoryRow(s, project)).join('')}
            </div>
          </div>
        </div>
      </div>`;

    _bindEvents(project);
  }

  function _sortStories(stories) {
    const priorityOrder = { critical: 0, high: 1, medium: 2, low: 3 };
    const sorted = [...stories];
    switch (sortBy) {
      case 'priority':
        sorted.sort((a, b) => (priorityOrder[a.priority] || 2) - (priorityOrder[b.priority] || 2));
        break;
      case 'points':
        sorted.sort((a, b) => (b.storyPoints || 0) - (a.storyPoints || 0));
        break;
      case 'created':
        sorted.sort((a, b) => new Date(b.createdAt) - new Date(a.createdAt));
        break;
    }
    return sorted;
  }

  function _renderGroupedStories(stories, project) {
    if (groupBy === 'none' || stories.length === 0) {
      if (stories.length === 0) {
        return '<div class="empty-state" style="padding:2rem"><i class="fa-solid fa-inbox" style="font-size:2rem"></i><p>No stories in backlog</p></div>';
      }
      return stories.map(s => Components.backlogStoryRow(s, project)).join('');
    }

    const groups = {};
    stories.forEach(s => {
      let key;
      if (groupBy === 'epic') {
        const epic = s.epicId ? Store.getEpicById(s.epicId, project) : null;
        key = epic ? epic.name : 'No Epic';
      } else if (groupBy === 'priority') {
        key = s.priority || 'medium';
      }
      if (!groups[key]) groups[key] = [];
      groups[key].push(s);
    });

    let html = '';
    for (const [key, items] of Object.entries(groups)) {
      let colorHtml = '';
      if (groupBy === 'epic') {
        const epic = (project.epics || []).find(e => e.name === key);
        if (epic) colorHtml = `<span class="group-color" style="background:${epic.color}"></span>`;
      } else if (groupBy === 'priority') {
        const colors = { critical: 'var(--priority-critical)', high: 'var(--priority-high)', medium: 'var(--priority-medium)', low: 'var(--priority-low)' };
        colorHtml = `<span class="group-color" style="background:${colors[key] || 'var(--border)'}"></span>`;
      }
      html += `<div class="backlog-group-header">${colorHtml} ${Components._esc(key)} (${items.length})</div>`;
      html += items.map(s => Components.backlogStoryRow(s, project)).join('');
    }
    return html;
  }

  function _bindEvents(project) {
    // Create story
    document.getElementById('btn-backlog-create')?.addEventListener('click', _createStory);
    document.getElementById('backlog-new-title')?.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') _createStory();
    });

    // Group buttons
    document.querySelectorAll('.group-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        groupBy = btn.dataset.group;
        render();
      });
    });

    // Sort buttons
    document.querySelectorAll('.sort-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        sortBy = btn.dataset.sort;
        render();
      });
    });

    // Sprint selector
    document.getElementById('backlog-sprint-selector')?.addEventListener('change', (e) => {
      selectedSprintId = e.target.value;
      render();
    });

    // Checkboxes
    document.querySelectorAll('.backlog-story-checkbox').forEach(cb => {
      cb.addEventListener('change', (e) => {
        const id = e.target.dataset.storyId;
        if (e.target.checked) selectedStories.add(id);
        else selectedStories.delete(id);
        render();
      });
    });

    // Bulk actions
    document.getElementById('btn-bulk-move-sprint')?.addEventListener('click', () => {
      selectedStories.forEach(id => Store.moveStoryToSprint(id, selectedSprintId, project));
      Components.showToast(`Moved ${selectedStories.size} stories to sprint`, 'success');
      selectedStories.clear();
      Components.showSaved();
      render();
    });

    document.getElementById('btn-bulk-delete')?.addEventListener('click', async () => {
      const yes = await Components.confirm('Delete Stories', `Delete ${selectedStories.size} selected stories?`);
      if (yes) {
        selectedStories.forEach(id => Store.deleteStory(id, project));
        selectedStories.clear();
        Components.showSaved();
        render();
      }
    });

    document.getElementById('btn-bulk-clear')?.addEventListener('click', () => {
      selectedStories.clear();
      render();
    });

    // Card clicks
    document.querySelectorAll('.backlog-story-row').forEach(row => {
      row.addEventListener('click', (e) => {
        if (e.target.closest('.backlog-story-checkbox') || e.target.closest('.backlog-story-actions') || e.target.closest('.backlog-drag-handle')) return;
        Modals.openCardDetail(row.dataset.storyId);
      });
    });

    // Edit / Delete buttons
    document.querySelectorAll('.btn-edit-story').forEach(btn => {
      btn.addEventListener('click', (e) => {
        e.stopPropagation();
        Modals.openCardDetail(btn.dataset.storyId);
      });
    });

    document.querySelectorAll('.btn-delete-story').forEach(btn => {
      btn.addEventListener('click', async (e) => {
        e.stopPropagation();
        const yes = await Components.confirm('Delete Story', 'Are you sure you want to delete this story?');
        if (yes) {
          Store.deleteStory(btn.dataset.storyId, project);
          Components.showSaved();
          render();
        }
      });
    });

    // Planning poker
    document.getElementById('btn-poker')?.addEventListener('click', () => Modals.openPoker());

    // Drag & Drop between backlog and sprint
    _setupDragDrop(project);
  }

  function _createStory() {
    const title = document.getElementById('backlog-new-title')?.value?.trim();
    if (!title) return;
    const points = parseInt(document.getElementById('backlog-new-points')?.value) || 0;
    const priority = document.getElementById('backlog-new-priority')?.value || 'medium';

    Store.createStory({ title, storyPoints: points, priority });
    Components.showToast(`Created "${title}"`, 'success');
    Components.showSaved();
    render();
  }

  function _setupDragDrop(project) {
    const rows = document.querySelectorAll('.backlog-story-row');
    const lists = document.querySelectorAll('.backlog-list');

    rows.forEach(row => {
      row.addEventListener('dragstart', (e) => {
        row.classList.add('dragging');
        e.dataTransfer.setData('text/plain', row.dataset.storyId);
        e.dataTransfer.effectAllowed = 'move';
      });
      row.addEventListener('dragend', () => {
        row.classList.remove('dragging');
        lists.forEach(l => l.classList.remove('drag-over'));
      });
    });

    lists.forEach(list => {
      list.addEventListener('dragover', (e) => {
        e.preventDefault();
        e.dataTransfer.dropEffect = 'move';
        list.classList.add('drag-over');
      });
      list.addEventListener('dragleave', (e) => {
        if (!list.contains(e.relatedTarget)) {
          list.classList.remove('drag-over');
        }
      });
      list.addEventListener('drop', (e) => {
        e.preventDefault();
        list.classList.remove('drag-over');
        const storyId = e.dataTransfer.getData('text/plain');
        if (!storyId) return;

        const target = list.dataset.target;
        if (target === 'sprint' && selectedSprintId) {
          Store.moveStoryToSprint(storyId, selectedSprintId, project);
          const story = Store.getStoryById(storyId);
          Components.showToast(`Moved "${story?.title || 'story'}" to sprint`, 'success');
        } else if (target === 'backlog') {
          Store.moveStoryToSprint(storyId, null, project);
          const story = Store.getStoryById(storyId);
          Components.showToast(`Moved "${story?.title || 'story'}" to backlog`, 'info');
        }
        Components.showSaved();
        render();
      });
    });
  }

  return { render };
})();
