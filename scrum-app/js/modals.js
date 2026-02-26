/* =====================================================
   MODALS.JS — Card detail modal + all other modals
   ===================================================== */

const Modals = (() => {
  let currentStoryId = null;
  let retroActions = [];

  function init() {
    // Close modal buttons
    document.querySelectorAll('.modal-close').forEach(btn => {
      btn.addEventListener('click', () => {
        const modalId = btn.dataset.modal;
        if (modalId) document.getElementById(modalId)?.classList.add('hidden');
      });
    });

    // Close modal on overlay click
    document.querySelectorAll('.modal-overlay').forEach(overlay => {
      overlay.addEventListener('click', (e) => {
        if (e.target === overlay) overlay.classList.add('hidden');
      });
    });

    // ---- New Project Modal ----
    document.getElementById('btn-new-project')?.addEventListener('click', () => {
      document.getElementById('new-project-name').value = '';
      document.getElementById('new-project-desc').value = '';
      document.getElementById('modal-new-project').classList.remove('hidden');
    });

    document.getElementById('btn-create-project')?.addEventListener('click', () => {
      const name = document.getElementById('new-project-name').value.trim();
      if (!name) return;
      const desc = document.getElementById('new-project-desc').value.trim();
      Store.createProject(name, desc);
      document.getElementById('modal-new-project').classList.add('hidden');
      Components.showToast(`Project "${name}" created!`, 'success');
      Components.showSaved();
      App.refreshAll();
    });

    // ---- New Sprint Modal ----
    document.getElementById('btn-create-sprint')?.addEventListener('click', () => {
      const name = document.getElementById('new-sprint-name').value.trim();
      if (!name) return;
      Store.createSprint({
        name,
        goal: document.getElementById('new-sprint-goal').value.trim(),
        startDate: document.getElementById('new-sprint-start').value,
        endDate: document.getElementById('new-sprint-end').value,
        capacity: parseInt(document.getElementById('new-sprint-capacity').value) || 40
      });
      document.getElementById('modal-new-sprint').classList.add('hidden');
      Components.showToast(`Sprint "${name}" created!`, 'success');
      Components.showSaved();
      App.renderCurrentPage();
    });

    // ---- Team Member Modal ----
    document.getElementById('btn-save-member')?.addEventListener('click', () => {
      const id = document.getElementById('team-member-id').value;
      const data = {
        name: document.getElementById('team-member-name').value.trim(),
        role: document.getElementById('team-member-role').value,
        capacity: parseInt(document.getElementById('team-member-capacity').value) || 10,
        color: document.getElementById('team-member-color').value
      };
      if (!data.name) return;

      if (id) {
        Store.updateTeamMember(id, data);
        Components.showToast('Member updated', 'success');
      } else {
        Store.createTeamMember(data);
        Components.showToast(`Added "${data.name}"`, 'success');
      }
      document.getElementById('modal-team-member').classList.add('hidden');
      Components.showSaved();
      App.renderCurrentPage();
    });

    // ---- Epic Modal ----
    document.getElementById('btn-save-epic')?.addEventListener('click', () => {
      const id = document.getElementById('epic-id').value;
      const data = {
        name: document.getElementById('epic-name').value.trim(),
        description: document.getElementById('epic-desc').value.trim(),
        color: document.getElementById('epic-color').value
      };
      if (!data.name) return;

      if (id) {
        Store.updateEpic(id, data);
        Components.showToast('Epic updated', 'success');
      } else {
        Store.createEpic(data);
        Components.showToast(`Created epic "${data.name}"`, 'success');
      }
      document.getElementById('modal-epic').classList.add('hidden');
      Components.showSaved();
      App.renderCurrentPage();
    });

    // ---- Card Detail Modal ----
    _initCardDetailModal();

    // ---- Complete Sprint Modal ----
    _initCompleteSprintModal();

    // ---- Search Modal ----
    _initSearchModal();
  }

  // =============== Card Detail ===============
  function _initCardDetailModal() {
    // Fibonacci selector
    document.querySelectorAll('.fib-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.fib-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
      });
    });

    // Tags
    document.getElementById('tag-input')?.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        e.preventDefault();
        const val = e.target.value.trim();
        if (!val) return;
        _addTag(val);
        e.target.value = '';
      }
    });

    // Subtasks
    document.getElementById('btn-add-subtask')?.addEventListener('click', _addSubtask);
    document.getElementById('subtask-input')?.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') { e.preventDefault(); _addSubtask(); }
    });

    // Save
    document.getElementById('btn-save-card')?.addEventListener('click', _saveCard);

    // Delete
    document.getElementById('btn-delete-card')?.addEventListener('click', async () => {
      if (!currentStoryId) return;
      const yes = await Components.confirm('Delete Story', 'Are you sure you want to delete this story?');
      if (yes) {
        Store.deleteStory(currentStoryId);
        document.getElementById('modal-card-detail').classList.add('hidden');
        Components.showToast('Story deleted', 'info');
        Components.showSaved();
        App.renderCurrentPage();
      }
    });

    // Keyboard shortcuts
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') {
        document.querySelectorAll('.modal-overlay:not(.hidden)').forEach(m => m.classList.add('hidden'));
      }
      if (e.ctrlKey && e.key === 'Enter') {
        const modal = document.getElementById('modal-card-detail');
        if (!modal.classList.contains('hidden')) {
          _saveCard();
        }
      }
    });
  }

  function openCardDetail(storyId, defaultSprintId) {
    const project = Store.getActiveProject();
    if (!project) return;

    currentStoryId = storyId;
    const story = storyId ? Store.getStoryById(storyId, project) : null;

    // Populate dropdowns
    _populateCardDropdowns(project);

    if (story) {
      document.getElementById('card-detail-title').value = story.title;
      document.getElementById('card-detail-desc').value = story.description || '';
      document.getElementById('card-detail-criteria').value = story.acceptanceCriteria || '';
      document.getElementById('card-detail-status').value = story.status;
      document.getElementById('card-detail-assignee').value = story.assigneeId || '';
      document.getElementById('card-detail-priority').value = story.priority;
      document.getElementById('card-detail-epic').value = story.epicId || '';
      document.getElementById('card-detail-sprint').value = story.sprintId || '';
      document.getElementById('card-detail-created').textContent = Components.formatDate(story.createdAt);
      document.getElementById('card-detail-updated').textContent = Components.formatDate(story.updatedAt);

      // Points
      document.querySelectorAll('.fib-btn').forEach(b => {
        b.classList.toggle('active', parseInt(b.dataset.points) === story.storyPoints);
      });

      // Tags
      _renderTags(story.tags || []);

      // Subtasks
      _renderSubtasks(story.subtasks || []);

      // Activity log
      _renderActivityLog(story.activityLog || []);
    } else {
      // New story
      document.getElementById('card-detail-title').value = '';
      document.getElementById('card-detail-desc').value = '';
      document.getElementById('card-detail-criteria').value = '';
      document.getElementById('card-detail-status').value = 'todo';
      document.getElementById('card-detail-assignee').value = '';
      document.getElementById('card-detail-priority').value = 'medium';
      document.getElementById('card-detail-epic').value = '';
      document.getElementById('card-detail-sprint').value = defaultSprintId || '';
      document.getElementById('card-detail-created').textContent = '—';
      document.getElementById('card-detail-updated').textContent = '—';
      document.querySelectorAll('.fib-btn').forEach(b => b.classList.remove('active'));
      _renderTags([]);
      _renderSubtasks([]);
      _renderActivityLog([]);
    }

    document.getElementById('modal-card-detail').classList.remove('hidden');
    document.getElementById('card-detail-title').focus();
  }

  function _populateCardDropdowns(project) {
    // Assignee
    const assigneeSel = document.getElementById('card-detail-assignee');
    assigneeSel.innerHTML = '<option value="">Unassigned</option>';
    (project.teamMembers || []).forEach(m => {
      assigneeSel.innerHTML += `<option value="${m.id}">${m.name}</option>`;
    });

    // Epic
    const epicSel = document.getElementById('card-detail-epic');
    epicSel.innerHTML = '<option value="">No Epic</option>';
    (project.epics || []).forEach(e => {
      epicSel.innerHTML += `<option value="${e.id}">${e.name}</option>`;
    });

    // Sprint
    const sprintSel = document.getElementById('card-detail-sprint');
    sprintSel.innerHTML = '<option value="">Backlog</option>';
    (project.sprints || []).filter(s => s.status !== 'completed').forEach(s => {
      sprintSel.innerHTML += `<option value="${s.id}">${s.name}</option>`;
    });
  }

  function _renderTags(tags) {
    const container = document.getElementById('card-detail-tags');
    container.innerHTML = tags.map(t =>
      `<span class="tag-chip">${Components._esc(t)} <span class="tag-remove" data-tag="${t}">&times;</span></span>`
    ).join('');

    container.querySelectorAll('.tag-remove').forEach(btn => {
      btn.addEventListener('click', () => {
        btn.closest('.tag-chip').remove();
      });
    });
  }

  function _addTag(tag) {
    const container = document.getElementById('card-detail-tags');
    const chip = document.createElement('span');
    chip.className = 'tag-chip';
    chip.innerHTML = `${Components._esc(tag)} <span class="tag-remove" data-tag="${tag}">&times;</span>`;
    chip.querySelector('.tag-remove').addEventListener('click', () => chip.remove());
    container.appendChild(chip);
  }

  function _renderSubtasks(subtasks) {
    const list = document.getElementById('card-detail-subtasks');
    list.innerHTML = subtasks.map((st, i) => `
      <div class="subtask-item" data-index="${i}">
        <input type="checkbox" ${st.completed ? 'checked' : ''} class="subtask-checkbox">
        <span class="subtask-text ${st.completed ? 'completed' : ''}">${Components._esc(st.text)}</span>
        <button class="subtask-delete"><i class="fa-solid fa-xmark"></i></button>
      </div>
    `).join('');

    // Checkbox toggles
    list.querySelectorAll('.subtask-checkbox').forEach(cb => {
      cb.addEventListener('change', () => {
        const text = cb.nextElementSibling;
        text.classList.toggle('completed', cb.checked);
        _updateSubtaskProgress();
      });
    });

    // Delete buttons
    list.querySelectorAll('.subtask-delete').forEach(btn => {
      btn.addEventListener('click', () => {
        btn.closest('.subtask-item').remove();
        _updateSubtaskProgress();
      });
    });

    _updateSubtaskProgress();
  }

  function _addSubtask() {
    const input = document.getElementById('subtask-input');
    const text = input.value.trim();
    if (!text) return;

    const list = document.getElementById('card-detail-subtasks');
    const item = document.createElement('div');
    item.className = 'subtask-item';
    item.innerHTML = `
      <input type="checkbox" class="subtask-checkbox">
      <span class="subtask-text">${Components._esc(text)}</span>
      <button class="subtask-delete"><i class="fa-solid fa-xmark"></i></button>
    `;

    item.querySelector('.subtask-checkbox').addEventListener('change', (e) => {
      e.target.nextElementSibling.classList.toggle('completed', e.target.checked);
      _updateSubtaskProgress();
    });

    item.querySelector('.subtask-delete').addEventListener('click', () => {
      item.remove();
      _updateSubtaskProgress();
    });

    list.appendChild(item);
    input.value = '';
    _updateSubtaskProgress();
  }

  function _updateSubtaskProgress() {
    const items = document.querySelectorAll('#card-detail-subtasks .subtask-item');
    const total = items.length;
    const done = document.querySelectorAll('#card-detail-subtasks .subtask-checkbox:checked').length;
    const pct = total > 0 ? Math.round((done / total) * 100) : 0;

    document.getElementById('subtask-progress-bar').style.width = pct + '%';
    document.getElementById('subtask-progress-text').textContent = `${done}/${total}`;

    const wrap = document.getElementById('subtask-progress-wrap');
    wrap.style.display = total > 0 ? 'flex' : 'none';
  }

  function _renderActivityLog(log) {
    const container = document.getElementById('card-detail-activity');
    if (log.length === 0) {
      container.innerHTML = '<p style="color:var(--text-muted);font-size:0.8rem">No activity yet</p>';
      return;
    }
    container.innerHTML = log.map(entry => `
      <div class="activity-log-item">
        <div>${Components._esc(entry.text)}</div>
        <div class="log-time">${Components.timeAgo(entry.timestamp)}</div>
      </div>
    `).join('');
  }

  function _saveCard() {
    const project = Store.getActiveProject();
    if (!project) return;

    const title = document.getElementById('card-detail-title').value.trim();
    if (!title) {
      Components.showToast('Title is required', 'warning');
      return;
    }

    // Collect tags
    const tags = Array.from(document.querySelectorAll('#card-detail-tags .tag-chip'))
      .map(chip => chip.textContent.replace('×', '').trim())
      .filter(t => t);

    // Collect subtasks
    const subtasks = Array.from(document.querySelectorAll('#card-detail-subtasks .subtask-item'))
      .map(item => ({
        id: Store.generateId(),
        text: item.querySelector('.subtask-text').textContent,
        completed: item.querySelector('.subtask-checkbox').checked
      }));

    // Points
    const activePoint = document.querySelector('.fib-btn.active');
    const points = activePoint ? parseInt(activePoint.dataset.points) : 0;

    const data = {
      title,
      description: document.getElementById('card-detail-desc').value,
      acceptanceCriteria: document.getElementById('card-detail-criteria').value,
      status: document.getElementById('card-detail-status').value,
      assigneeId: document.getElementById('card-detail-assignee').value || null,
      storyPoints: points,
      priority: document.getElementById('card-detail-priority').value,
      epicId: document.getElementById('card-detail-epic').value || null,
      sprintId: document.getElementById('card-detail-sprint').value || null,
      tags,
      subtasks
    };

    if (currentStoryId) {
      Store.updateStory(currentStoryId, data, project);
      Components.showToast('Story updated', 'success');

      // Update burndown if status changed
      const activeSprint = Store.getActiveSprint(project);
      if (activeSprint) Store.updateBurndown(activeSprint.id, project);
    } else {
      Store.createStory(data, project);
      Components.showToast(`Created "${title}"`, 'success');
    }

    document.getElementById('modal-card-detail').classList.add('hidden');
    Components.showSaved();
    App.renderCurrentPage();
  }

  // =============== Complete Sprint ===============
  function _initCompleteSprintModal() {
    retroActions = [];

    document.getElementById('btn-add-retro-action')?.addEventListener('click', () => {
      const input = document.getElementById('retro-action-input');
      const text = input.value.trim();
      if (!text) return;
      retroActions.push(text);
      input.value = '';
      _renderRetroActions();
    });

    document.getElementById('retro-action-input')?.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        e.preventDefault();
        document.getElementById('btn-add-retro-action').click();
      }
    });

    document.getElementById('btn-export-retro')?.addEventListener('click', () => {
      const project = Store.getActiveProject();
      const sprintId = document.getElementById('modal-complete-sprint').dataset.sprintId;
      const sprint = Store.getSprintById(sprintId, project);
      if (!sprint) return;

      const report = {
        sprint: sprint.name,
        goal: sprint.goal,
        dates: `${sprint.startDate} → ${sprint.endDate}`,
        retrospective: {
          well: document.getElementById('retro-well').value,
          improve: document.getElementById('retro-improve').value,
          actions: retroActions
        },
        stories: (sprint.stories || []).map(s => ({
          title: s.title, status: s.status, points: s.storyPoints, priority: s.priority
        }))
      };

      const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `retro-${sprint.name.replace(/\s+/g, '-')}.json`;
      a.click();
      URL.revokeObjectURL(url);
      Components.showToast('Retrospective exported!', 'success');
    });
  }

  function _renderRetroActions() {
    const list = document.getElementById('retro-actions-list');
    list.innerHTML = retroActions.map((a, i) => `
      <div class="retro-action-item">
        <span>${Components._esc(a)}</span>
        <button data-index="${i}"><i class="fa-solid fa-xmark"></i></button>
      </div>
    `).join('');

    list.querySelectorAll('button').forEach(btn => {
      btn.addEventListener('click', () => {
        retroActions.splice(parseInt(btn.dataset.index), 1);
        _renderRetroActions();
      });
    });
  }

  function openCompleteSprint(sprintId) {
    const project = Store.getActiveProject();
    const sprint = Store.getSprintById(sprintId, project);
    if (!sprint) return;

    retroActions = [];
    const stories = sprint.stories || [];
    const doneStories = stories.filter(s => s.status === 'done');
    const undoneStories = stories.filter(s => s.status !== 'done');
    const totalPts = stories.reduce((s, st) => s + (st.storyPoints || 0), 0);
    const donePts = doneStories.reduce((s, st) => s + (st.storyPoints || 0), 0);

    const summary = document.getElementById('complete-sprint-summary');
    summary.innerHTML = `
      <div class="summary-item"><span class="summary-value text-success">${doneStories.length}</span><span class="summary-label">Done</span></div>
      <div class="summary-item"><span class="summary-value text-warning">${undoneStories.length}</span><span class="summary-label">Undone</span></div>
      <div class="summary-item"><span class="summary-value text-accent">${donePts}</span><span class="summary-label">Points Done</span></div>
      <div class="summary-item"><span class="summary-value">${totalPts - donePts}</span><span class="summary-label">Points Left</span></div>
    `;

    if (undoneStories.length > 0) {
      summary.innerHTML += `<p style="width:100%;font-size:0.8rem;color:var(--warning);margin-top:0.5rem"><i class="fa-solid fa-triangle-exclamation"></i> ${undoneStories.length} undone stories will be moved back to backlog.</p>`;
    }

    document.getElementById('retro-well').value = '';
    document.getElementById('retro-improve').value = '';
    document.getElementById('retro-actions-list').innerHTML = '';

    const modal = document.getElementById('modal-complete-sprint');
    modal.dataset.sprintId = sprintId;
    modal.classList.remove('hidden');

    // Confirm button
    const confirmBtn = document.getElementById('btn-confirm-complete-sprint');
    const newBtn = confirmBtn.cloneNode(true);
    confirmBtn.parentNode.replaceChild(newBtn, confirmBtn);

    newBtn.addEventListener('click', () => {
      const retro = {
        well: document.getElementById('retro-well').value.trim(),
        improve: document.getElementById('retro-improve').value.trim(),
        actions: [...retroActions]
      };

      Store.completeSprint(sprintId, retro, project);
      modal.classList.add('hidden');
      Components.showToast(`Sprint completed! Velocity: ${donePts}`, 'success');
      Components.showSaved();
      App.updateSidebarSprintInfo();
      App.renderCurrentPage();
    });
  }

  // =============== Search Modal ===============
  function _initSearchModal() {
    const input = document.getElementById('search-modal-input');
    const globalInput = document.getElementById('global-search-input');

    globalInput?.addEventListener('focus', () => openSearch());
    globalInput?.addEventListener('click', () => openSearch());

    input?.addEventListener('input', (e) => {
      _performSearch(e.target.value.trim());
    });
  }

  function openSearch() {
    const modal = document.getElementById('modal-search');
    modal.classList.remove('hidden');
    const input = document.getElementById('search-modal-input');
    input.value = '';
    input.focus();
    document.getElementById('search-results').innerHTML = '';
  }

  function _performSearch(query) {
    const results = document.getElementById('search-results');
    if (!query) {
      results.innerHTML = '<p style="text-align:center;color:var(--text-muted);padding:2rem">Type to search stories...</p>';
      return;
    }

    const project = Store.getActiveProject();
    if (!project) return;

    const allStories = Store.getAllStories(project);
    const q = query.toLowerCase();
    const matched = allStories.filter(s =>
      s.title.toLowerCase().includes(q) || (s.description || '').toLowerCase().includes(q)
    ).slice(0, 10);

    if (matched.length === 0) {
      results.innerHTML = '<p style="text-align:center;color:var(--text-muted);padding:2rem">No results found</p>';
      return;
    }

    const priorityColors = { critical: 'var(--priority-critical)', high: 'var(--priority-high)', medium: 'var(--priority-medium)', low: 'var(--priority-low)' };

    results.innerHTML = matched.map(s => `
      <div class="search-result-item" data-story-id="${s.id}">
        <div class="result-priority" style="background:${priorityColors[s.priority] || 'var(--border)'}"></div>
        <div class="result-info">
          <div class="result-title">${Components._esc(s.title)}</div>
          <div class="result-desc">${Components._esc((s.description || '').substring(0, 80))}</div>
        </div>
        ${Components.pointsBadgeSmall(s.storyPoints)}
        ${Components.statusBadge(s.status)}
      </div>
    `).join('');

    results.querySelectorAll('.search-result-item').forEach(item => {
      item.addEventListener('click', () => {
        document.getElementById('modal-search').classList.add('hidden');
        openCardDetail(item.dataset.storyId);
      });
    });
  }

  // =============== Standup ===============
  function openStandup() {
    const project = Store.getActiveProject();
    const sprint = Store.getActiveSprint(project);
    const content = document.getElementById('standup-content');

    if (!sprint) {
      content.innerHTML = '<p style="color:var(--text-muted)">No active sprint</p>';
      document.getElementById('modal-standup').classList.remove('hidden');
      return;
    }

    const members = Store.getTeamMembers(project);
    const stories = sprint.stories || [];

    content.innerHTML = members.map(m => {
      const memberStories = stories.filter(s => s.assigneeId === m.id && s.status === 'inprogress');
      const reviewStories = stories.filter(s => s.assigneeId === m.id && s.status === 'review');
      const todoStories = stories.filter(s => s.assigneeId === m.id && s.status === 'todo');

      return `
        <div class="standup-member">
          <div class="standup-member-header">
            ${Components.avatar(m, 'md')}
            <div>
              <strong>${Components._esc(m.name)}</strong>
              ${Components.roleBadge(m.role)}
            </div>
          </div>
          <div class="standup-stories">
            ${memberStories.length > 0 ? '<p style="font-size:0.75rem;font-weight:600;color:var(--accent);margin-top:0.5rem">🔵 In Progress</p>' : ''}
            ${memberStories.map(s => `<div class="standup-story">• ${Components._esc(s.title)} (${s.storyPoints || 0} pts)</div>`).join('')}
            ${reviewStories.length > 0 ? '<p style="font-size:0.75rem;font-weight:600;color:var(--warning);margin-top:0.5rem">🟡 In Review</p>' : ''}
            ${reviewStories.map(s => `<div class="standup-story">• ${Components._esc(s.title)} (${s.storyPoints || 0} pts)</div>`).join('')}
            ${todoStories.length > 0 ? '<p style="font-size:0.75rem;font-weight:600;color:var(--text-muted);margin-top:0.5rem">⚪ To Do</p>' : ''}
            ${todoStories.map(s => `<div class="standup-story">• ${Components._esc(s.title)} (${s.storyPoints || 0} pts)</div>`).join('')}
            ${memberStories.length === 0 && reviewStories.length === 0 && todoStories.length === 0 ? '<p style="font-size:0.8rem;color:var(--text-muted)">No assigned stories</p>' : ''}
          </div>
        </div>
      `;
    }).join('');

    document.getElementById('modal-standup').classList.remove('hidden');
  }

  // =============== Planning Poker ===============
  function openPoker() {
    const project = Store.getActiveProject();
    const content = document.getElementById('poker-content');
    const backlogStories = (project?.backlog || []).filter(s => !s.storyPoints || s.storyPoints === 0);
    const members = Store.getTeamMembers(project);

    if (backlogStories.length === 0) {
      content.innerHTML = '<p style="text-align:center;color:var(--text-muted)">All stories have estimates!</p>';
      document.getElementById('modal-poker').classList.remove('hidden');
      return;
    }

    const story = backlogStories[0];
    let selectedValue = null;
    let votes = {};
    let revealed = false;

    const renderPoker = () => {
      content.innerHTML = `
        <div class="poker-story">${Components._esc(story.title)}</div>
        <p style="text-align:center;color:var(--text-muted);font-size:0.8rem;margin-bottom:1rem">${Components._esc(story.description || 'No description')}</p>
        <div class="poker-cards">
          ${[1,2,3,5,8,13,21].map(n => `
            <div class="poker-card ${selectedValue === n ? 'selected' : ''}" data-value="${n}">${n}</div>
          `).join('')}
        </div>
        <div style="text-align:center;margin-bottom:1rem">
          <button class="btn btn-accent" id="btn-poker-reveal" ${!selectedValue ? 'disabled' : ''}><i class="fa-solid fa-eye"></i> Reveal Votes</button>
        </div>
        <div class="poker-votes">
          ${members.map(m => `
            <div class="poker-vote">
              <div class="vote-card ${revealed ? 'revealed' : ''}">${revealed ? (votes[m.id] || '?') : '?'}</div>
              <div class="vote-name">${Components._esc(m.name)}</div>
            </div>
          `).join('')}
        </div>
        ${revealed ? `
          <div style="text-align:center;margin-top:1rem">
            <p style="color:var(--text-muted);font-size:0.8rem">Consensus: <strong style="color:var(--accent);font-size:1.2rem">${_getConsensus(votes)}</strong></p>
            <button class="btn btn-success" id="btn-poker-accept" style="margin-top:0.5rem"><i class="fa-solid fa-check"></i> Accept Estimate</button>
          </div>
        ` : ''}
      `;

      // Bind poker card clicks
      content.querySelectorAll('.poker-card').forEach(card => {
        card.addEventListener('click', () => {
          selectedValue = parseInt(card.dataset.value);
          // Simulate votes
          members.forEach(m => {
            const offset = Math.random() > 0.5 ? 1 : -1;
            const fib = [1,2,3,5,8,13,21];
            const idx = fib.indexOf(selectedValue);
            const simIdx = Math.max(0, Math.min(fib.length - 1, idx + (Math.random() > 0.7 ? offset : 0)));
            votes[m.id] = fib[simIdx];
          });
          renderPoker();
        });
      });

      content.querySelector('#btn-poker-reveal')?.addEventListener('click', () => {
        revealed = true;
        renderPoker();
      });

      content.querySelector('#btn-poker-accept')?.addEventListener('click', () => {
        const consensus = _getConsensus(votes);
        Store.updateStory(story.id, { storyPoints: consensus }, project);
        Components.showToast(`Set ${story.title} to ${consensus} points`, 'success');
        Components.showSaved();
        document.getElementById('modal-poker').classList.add('hidden');
        App.renderCurrentPage();
      });
    };

    renderPoker();
    document.getElementById('modal-poker').classList.remove('hidden');
  }

  function _getConsensus(votes) {
    const vals = Object.values(votes).filter(v => typeof v === 'number');
    if (vals.length === 0) return 0;
    const avg = vals.reduce((a, b) => a + b, 0) / vals.length;
    const fib = [1,2,3,5,8,13,21];
    return fib.reduce((prev, curr) => Math.abs(curr - avg) < Math.abs(prev - avg) ? curr : prev);
  }

  return {
    init, openCardDetail, openCompleteSprint, openSearch, openStandup, openPoker
  };
})();
