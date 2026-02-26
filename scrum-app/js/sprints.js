/* =====================================================
   SPRINTS.JS — Sprint management
   ===================================================== */

const SprintsView = (() => {

  function render() {
    const project = Store.getActiveProject();
    const content = document.getElementById('app-content');

    if (!project) {
      content.innerHTML = '<div class="empty-state"><i class="fa-solid fa-rocket"></i><p>No project selected.</p></div>';
      return;
    }

    const sprints = Store.getSprints(project);
    const activeSprint = Store.getActiveSprint(project);

    content.innerHTML = `
      <div class="page-enter">
        <div class="page-header">
          <h2>Sprints</h2>
          <div class="page-header-actions">
            <button class="btn btn-accent" id="btn-new-sprint-page"><i class="fa-solid fa-plus"></i> New Sprint</button>
          </div>
        </div>
        <div class="sprint-list" id="sprint-list">
          ${sprints.length === 0 ? '<div class="empty-state"><i class="fa-solid fa-rocket"></i><p>No sprints yet. Create your first sprint!</p></div>' : ''}
          ${sprints.map(sprint => _renderSprintRow(sprint, project, activeSprint)).join('')}
        </div>
      </div>`;

    _bindEvents(project);
  }

  function _renderSprintRow(sprint, project, activeSprint) {
    const stories = sprint.stories || [];
    const totalPoints = stories.reduce((s, st) => s + (st.storyPoints || 0), 0);
    const donePoints = stories.filter(s => s.status === 'done').reduce((s, st) => s + (st.storyPoints || 0), 0);
    const pct = totalPoints > 0 ? Math.round((donePoints / totalPoints) * 100) : 0;
    const daysLeft = Components.daysRemaining(sprint.endDate);

    const isActive = sprint.status === 'active';
    const isPlanned = sprint.status === 'planned';
    const isCompleted = sprint.status === 'completed';

    return `
      <div class="sprint-row ${isActive ? 'active-sprint' : ''}" data-sprint-id="${sprint.id}">
        <div class="sprint-row-header">
          <div class="sprint-row-title">
            <h3>${Components._esc(sprint.name)}</h3>
            ${Components.sprintStatusBadge(sprint.status)}
          </div>
          <div class="sprint-row-actions">
            ${isPlanned && !activeSprint ? `<button class="btn btn-success btn-sm start-sprint-btn" data-sprint-id="${sprint.id}"><i class="fa-solid fa-play"></i> Start Sprint</button>` : ''}
            ${isActive ? `<button class="btn btn-outline btn-sm complete-sprint-btn" data-sprint-id="${sprint.id}"><i class="fa-solid fa-flag-checkered"></i> Complete Sprint</button>` : ''}
            ${!isActive ? `<button class="btn-icon delete-sprint-btn" data-sprint-id="${sprint.id}" title="Delete"><i class="fa-solid fa-trash"></i></button>` : ''}
          </div>
        </div>
        ${sprint.goal ? `<p style="font-size:0.85rem;color:var(--text-muted);margin-bottom:0.5rem"><i class="fa-solid fa-bullseye"></i> ${Components._esc(sprint.goal)}</p>` : ''}
        <div class="sprint-row-meta">
          <span><i class="fa-solid fa-calendar"></i> ${Components.formatDateShort(sprint.startDate)} → ${Components.formatDateShort(sprint.endDate)}</span>
          <span><i class="fa-solid fa-book"></i> ${stories.length} stories</span>
          <span><i class="fa-solid fa-chart-simple"></i> ${donePoints}/${totalPoints} pts</span>
          ${isActive ? `<span><i class="fa-solid fa-hourglass-half"></i> ${daysLeft} days left</span>` : ''}
          ${isCompleted ? `<span><i class="fa-solid fa-tachometer-alt"></i> Velocity: ${sprint.velocity}</span>` : ''}
        </div>
        <div class="sprint-row-progress">
          ${Components.progressBar(pct, pct >= 100 ? 'success' : '')}
          <small style="color:var(--text-muted)">${pct}% complete</small>
        </div>
        <details class="sprint-detail-expand">
          <summary style="cursor:pointer;font-size:0.8rem;color:var(--text-muted);font-weight:600;">Show Stories (${stories.length})</summary>
          <div style="margin-top:0.75rem">
            ${stories.length === 0 ? '<p style="color:var(--text-muted);font-size:0.8rem">No stories in this sprint</p>' : ''}
            ${stories.map(s => `
              <div class="backlog-story-row" style="cursor:pointer" data-story-id="${s.id}">
                ${Components.priorityDot(s.priority)}
                <span class="backlog-story-title">${Components._esc(s.title)}</span>
                ${Components.pointsBadgeSmall(s.storyPoints)}
                ${Components.statusBadge(s.status)}
              </div>
            `).join('')}
          </div>
          ${isCompleted && sprint.retrospective ? `
            <div style="margin-top:1rem;padding:1rem;background:var(--bg-primary);border-radius:var(--radius)">
              <h4 style="font-size:0.85rem;margin-bottom:0.5rem"><i class="fa-solid fa-rotate-left"></i> Retrospective</h4>
              <p style="font-size:0.8rem;margin-bottom:0.5rem"><strong>What went well:</strong> ${Components._esc(sprint.retrospective.well || '')}</p>
              <p style="font-size:0.8rem;margin-bottom:0.5rem"><strong>What to improve:</strong> ${Components._esc(sprint.retrospective.improve || '')}</p>
              ${(sprint.retrospective.actions || []).length > 0 ? `
                <p style="font-size:0.8rem;font-weight:600">Action Items:</p>
                <ul style="font-size:0.8rem;padding-left:1.5rem">
                  ${sprint.retrospective.actions.map(a => `<li>${Components._esc(a)}</li>`).join('')}
                </ul>
              ` : ''}
            </div>
          ` : ''}
        </details>
      </div>`;
  }

  function _bindEvents(project) {
    // New sprint
    document.getElementById('btn-new-sprint-page')?.addEventListener('click', () => {
      document.getElementById('modal-new-sprint').classList.remove('hidden');
    });

    // Start sprint
    document.querySelectorAll('.start-sprint-btn').forEach(btn => {
      btn.addEventListener('click', async (e) => {
        e.stopPropagation();
        const sprintId = btn.dataset.sprintId;
        const sprint = Store.getSprintById(sprintId, project);
        if (!sprint) return;

        if (!sprint.startDate || !sprint.endDate) {
          Components.showToast('Please set start and end dates first', 'warning');
          return;
        }

        const result = Store.startSprint(sprintId, project);
        if (result) {
          Components.showToast(`Sprint "${sprint.name}" started!`, 'success');
          Components.showSaved();
          App.updateSidebarSprintInfo();
          render();
        } else {
          Components.showToast('Cannot start sprint. Another sprint may be active.', 'danger');
        }
      });
    });

    // Complete sprint
    document.querySelectorAll('.complete-sprint-btn').forEach(btn => {
      btn.addEventListener('click', (e) => {
        e.stopPropagation();
        Modals.openCompleteSprint(btn.dataset.sprintId);
      });
    });

    // Delete sprint
    document.querySelectorAll('.delete-sprint-btn').forEach(btn => {
      btn.addEventListener('click', async (e) => {
        e.stopPropagation();
        const yes = await Components.confirm('Delete Sprint', 'Are you sure you want to delete this sprint? Stories will be moved to backlog.');
        if (yes) {
          const sprintId = btn.dataset.sprintId;
          const sprint = Store.getSprintById(sprintId, project);
          if (sprint) {
            // Move stories to backlog
            (sprint.stories || []).forEach(s => {
              s.sprintId = null;
              project.backlog.push(s);
            });
            project.sprints = project.sprints.filter(s => s.id !== sprintId);
            Store.saveProject(project);
            Components.showToast('Sprint deleted', 'info');
            Components.showSaved();
            render();
          }
        }
      });
    });

    // Story clicks in sprint details
    document.querySelectorAll('.sprint-detail-expand .backlog-story-row').forEach(row => {
      row.addEventListener('click', () => {
        Modals.openCardDetail(row.dataset.storyId);
      });
    });
  }

  return { render };
})();
