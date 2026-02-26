/* =====================================================
   EPICS.JS — Epics management
   ===================================================== */

const EpicsView = (() => {
  let expandedEpic = null;

  function render() {
    const project = Store.getActiveProject();
    const content = document.getElementById('app-content');

    if (!project) {
      content.innerHTML = '<div class="empty-state"><i class="fa-solid fa-layer-group"></i><p>No project selected.</p></div>';
      return;
    }

    const epics = Store.getEpics(project);

    content.innerHTML = `
      <div class="page-enter">
        <div class="page-header">
          <h2>Epics</h2>
          <div class="page-header-actions">
            <button class="btn btn-accent" id="btn-add-epic"><i class="fa-solid fa-plus"></i> Add Epic</button>
          </div>
        </div>
        <div class="epics-grid">
          ${epics.length === 0 ? '<div class="empty-state" style="grid-column:1/-1"><i class="fa-solid fa-layer-group"></i><p>No epics yet. Create your first epic to organize stories!</p></div>' : ''}
          ${epics.map(epic => _renderEpicCard(epic, project)).join('')}
        </div>
      </div>`;

    _bindEvents(project);
  }

  function _renderEpicCard(epic, project) {
    const stories = Store.getEpicStories(epic.id, project);
    const totalPoints = stories.reduce((s, st) => s + (st.storyPoints || 0), 0);
    const donePoints = stories.filter(s => s.status === 'done').reduce((s, st) => s + (st.storyPoints || 0), 0);
    const pct = totalPoints > 0 ? Math.round((donePoints / totalPoints) * 100) : 0;
    const isExpanded = expandedEpic === epic.id;

    return `
      <div class="epic-card" data-epic-id="${epic.id}">
        <div class="epic-card-header" style="background:${epic.color}">
          ${Components._esc(epic.name)}
        </div>
        <div class="epic-card-body">
          <div class="epic-card-desc">${Components._esc(epic.description || 'No description')}</div>
          <div class="epic-card-stats">
            <span><i class="fa-solid fa-book"></i> ${stories.length} stories</span>
            <span><i class="fa-solid fa-chart-simple"></i> ${totalPoints} pts</span>
            <span><i class="fa-solid fa-check-circle"></i> ${donePoints} done</span>
          </div>
          <div class="epic-card-progress">
            ${Components.progressBar(pct, pct >= 100 ? 'success' : '')}
            <div class="progress-text">${pct}% complete</div>
          </div>
          ${isExpanded ? `
            <div class="epic-expanded">
              ${stories.length === 0 ? '<p style="font-size:0.8rem;color:var(--text-muted)">No stories in this epic</p>' : ''}
              ${stories.map(s => `
                <div class="epic-story-item" data-story-id="${s.id}" style="cursor:pointer">
                  ${Components.priorityDot(s.priority)}
                  <span style="flex:1">${Components._esc(s.title)}</span>
                  ${Components.pointsBadgeSmall(s.storyPoints)}
                  ${Components.statusBadge(s.status)}
                </div>
              `).join('')}
            </div>
          ` : ''}
          <div class="epic-card-actions">
            <button class="btn btn-outline btn-sm toggle-expand-btn" data-epic-id="${epic.id}">
              <i class="fa-solid fa-${isExpanded ? 'chevron-up' : 'chevron-down'}"></i> ${isExpanded ? 'Collapse' : 'Expand'}
            </button>
            <button class="btn btn-outline btn-sm edit-epic-btn" data-epic-id="${epic.id}">
              <i class="fa-solid fa-pen"></i> Edit
            </button>
            <button class="btn-sm delete-epic-btn" data-epic-id="${epic.id}" style="color:var(--danger)">
              <i class="fa-solid fa-trash"></i>
            </button>
          </div>
        </div>
      </div>`;
  }

  function _bindEvents(project) {
    // Add epic
    document.getElementById('btn-add-epic')?.addEventListener('click', () => {
      document.getElementById('epic-id').value = '';
      document.getElementById('epic-name').value = '';
      document.getElementById('epic-desc').value = '';
      document.getElementById('epic-color').value = '#6366f1';
      document.getElementById('epic-modal-title').textContent = 'Add Epic';
      document.getElementById('modal-epic').classList.remove('hidden');
    });

    // Toggle expand
    document.querySelectorAll('.toggle-expand-btn').forEach(btn => {
      btn.addEventListener('click', (e) => {
        e.stopPropagation();
        const epicId = btn.dataset.epicId;
        expandedEpic = expandedEpic === epicId ? null : epicId;
        render();
      });
    });

    // Edit epic
    document.querySelectorAll('.edit-epic-btn').forEach(btn => {
      btn.addEventListener('click', (e) => {
        e.stopPropagation();
        const epic = Store.getEpicById(btn.dataset.epicId, project);
        if (!epic) return;
        document.getElementById('epic-id').value = epic.id;
        document.getElementById('epic-name').value = epic.name;
        document.getElementById('epic-desc').value = epic.description || '';
        document.getElementById('epic-color').value = epic.color;
        document.getElementById('epic-modal-title').textContent = 'Edit Epic';
        document.getElementById('modal-epic').classList.remove('hidden');
      });
    });

    // Delete epic
    document.querySelectorAll('.delete-epic-btn').forEach(btn => {
      btn.addEventListener('click', async (e) => {
        e.stopPropagation();
        const yes = await Components.confirm('Delete Epic', 'Delete this epic? Stories will be unlinked.');
        if (yes) {
          Store.deleteEpic(btn.dataset.epicId, project);
          Components.showToast('Epic deleted', 'info');
          Components.showSaved();
          render();
        }
      });
    });

    // Story clicks in expanded epic
    document.querySelectorAll('.epic-story-item').forEach(item => {
      item.addEventListener('click', () => {
        Modals.openCardDetail(item.dataset.storyId);
      });
    });
  }

  return { render };
})();
