/* =====================================================
   TEAM.JS — Team management
   ===================================================== */

const TeamView = (() => {

  function render() {
    const project = Store.getActiveProject();
    const content = document.getElementById('app-content');

    if (!project) {
      content.innerHTML = '<div class="empty-state"><i class="fa-solid fa-users"></i><p>No project selected.</p></div>';
      return;
    }

    const members = Store.getTeamMembers(project);
    const activeSprint = Store.getActiveSprint(project);
    const allStories = Store.getAllStories(project);
    const sprintStories = activeSprint ? (activeSprint.stories || []) : [];
    const completedSprints = (project.sprints || []).filter(s => s.status === 'completed').slice(-5);

    content.innerHTML = `
      <div class="page-enter">
        <div class="page-header">
          <h2>Team</h2>
          <div class="page-header-actions">
            <button class="btn btn-accent" id="btn-add-member"><i class="fa-solid fa-plus"></i> Add Member</button>
          </div>
        </div>
        <div class="team-grid">
          ${members.length === 0 ? '<div class="empty-state" style="grid-column:1/-1"><i class="fa-solid fa-users"></i><p>No team members yet.</p></div>' : ''}
          ${members.map(m => _renderMemberCard(m, project, sprintStories)).join('')}
        </div>
        ${completedSprints.length > 0 && members.length > 0 ? `
          <div class="dashboard-section" style="margin-top:1.5rem">
            <h3><i class="fa-solid fa-chart-simple"></i> Velocity by Member (Last ${completedSprints.length} Sprints)</h3>
            <table class="velocity-table">
              <thead>
                <tr>
                  <th>Member</th>
                  ${completedSprints.map(s => `<th>${Components._esc(s.name)}</th>`).join('')}
                  <th>Total</th>
                </tr>
              </thead>
              <tbody>
                ${members.map(m => {
                  let total = 0;
                  const cells = completedSprints.map(sprint => {
                    const pts = (sprint.stories || [])
                      .filter(s => s.assigneeId === m.id && s.status === 'done')
                      .reduce((sum, s) => sum + (s.storyPoints || 0), 0);
                    total += pts;
                    return `<td>${pts}</td>`;
                  }).join('');
                  return `<tr><td>${Components.avatar(m, 'sm')} ${Components._esc(m.name)}</td>${cells}<td><strong>${total}</strong></td></tr>`;
                }).join('')}
              </tbody>
            </table>
          </div>
        ` : ''}
      </div>`;

    _bindEvents(project);
  }

  function _renderMemberCard(member, project, sprintStories) {
    const assigned = sprintStories.filter(s => s.assigneeId === member.id);
    const assignedPoints = assigned.reduce((s, st) => s + (st.storyPoints || 0), 0);
    const capacity = member.capacity || 10;
    const loadPct = capacity > 0 ? Math.round((assignedPoints / capacity) * 100) : 0;

    let loadColor = 'var(--success)';
    let loadText = 'Available';
    if (loadPct > 100) { loadColor = 'var(--danger)'; loadText = 'Overloaded'; }
    else if (loadPct > 75) { loadColor = 'var(--warning)'; loadText = 'Near Capacity'; }
    else if (loadPct > 0) { loadText = 'Active'; }

    return `
      <div class="team-card" data-member-id="${member.id}">
        ${Components.avatar(member, 'xl')}
        <div class="team-name">${Components._esc(member.name)}</div>
        ${Components.roleBadge(member.role)}
        <div class="team-stats">
          <div class="team-stat">
            <span class="stat-value">${assigned.length}</span>
            <span class="stat-label">Stories</span>
          </div>
          <div class="team-stat">
            <span class="stat-value">${assignedPoints}</span>
            <span class="stat-label">Points</span>
          </div>
        </div>
        <div class="capacity-slider">
          <div class="capacity-label">
            <span>Capacity</span>
            <span>${capacity} pts/sprint</span>
          </div>
          <input type="range" min="0" max="50" value="${capacity}" data-member-id="${member.id}" class="capacity-range">
        </div>
        <div class="workload-indicator">
          <div class="workload-bar">
            <div class="workload-fill" style="width:${Math.min(loadPct, 100)}%;background:${loadColor}"></div>
          </div>
          <div class="workload-text" style="color:${loadColor}">${loadText} (${loadPct}%)</div>
        </div>
        <div class="team-card-actions">
          <button class="btn btn-outline btn-sm edit-member-btn" data-member-id="${member.id}"><i class="fa-solid fa-pen"></i> Edit</button>
          <button class="btn btn-sm" style="color:var(--danger)" data-member-id="${member.id}"><i class="fa-solid fa-trash"></i></button>
        </div>
      </div>`;
  }

  function _bindEvents(project) {
    // Add member
    document.getElementById('btn-add-member')?.addEventListener('click', () => {
      document.getElementById('team-member-id').value = '';
      document.getElementById('team-member-name').value = '';
      document.getElementById('team-member-role').value = 'Developer';
      document.getElementById('team-member-capacity').value = '10';
      document.getElementById('team-member-color').value = '#6366f1';
      document.getElementById('team-modal-title').textContent = 'Add Team Member';
      document.getElementById('modal-team-member').classList.remove('hidden');
    });

    // Edit member
    document.querySelectorAll('.edit-member-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const member = Store.getTeamMember(btn.dataset.memberId, project);
        if (!member) return;
        document.getElementById('team-member-id').value = member.id;
        document.getElementById('team-member-name').value = member.name;
        document.getElementById('team-member-role').value = member.role;
        document.getElementById('team-member-capacity').value = member.capacity;
        document.getElementById('team-member-color').value = member.color;
        document.getElementById('team-modal-title').textContent = 'Edit Team Member';
        document.getElementById('modal-team-member').classList.remove('hidden');
      });
    });

    // Delete member
    document.querySelectorAll('.team-card-actions button:last-child').forEach(btn => {
      btn.addEventListener('click', async () => {
        const memberId = btn.dataset.memberId;
        if (!memberId) return;
        const yes = await Components.confirm('Remove Member', 'Remove this team member?');
        if (yes) {
          Store.deleteTeamMember(memberId, project);
          Components.showSaved();
          render();
        }
      });
    });

    // Capacity sliders
    document.querySelectorAll('.capacity-range').forEach(slider => {
      slider.addEventListener('input', (e) => {
        const memberId = e.target.dataset.memberId;
        const value = parseInt(e.target.value);
        Store.updateTeamMember(memberId, { capacity: value }, project);
        // Update label in real-time
        const label = e.target.closest('.capacity-slider').querySelector('.capacity-label span:last-child');
        if (label) label.textContent = `${value} pts/sprint`;
      });
      slider.addEventListener('change', () => {
        Components.showSaved();
      });
    });
  }

  return { render };
})();
