/* =====================================================
   APP.JS — Router, init, sidebar, theme, keyboard shortcuts
   ===================================================== */

const App = (() => {
  const routes = {
    dashboard: { title: 'Dashboard', render: _renderDashboard },
    board: { title: 'Board', render: () => BoardView.render() },
    backlog: { title: 'Backlog', render: () => BacklogView.render() },
    sprints: { title: 'Sprints', render: () => SprintsView.render() },
    team: { title: 'Team', render: () => TeamView.render() },
    analytics: { title: 'Analytics', render: () => AnalyticsView.render() },
    epics: { title: 'Epics', render: () => EpicsView.render() }
  };

  let currentPage = 'dashboard';

  function init() {
    // Theme
    const theme = Store.getTheme();
    document.body.classList.toggle('dark-mode', theme === 'dark');
    document.body.classList.toggle('light-mode', theme === 'light');
    _updateThemeIcon();

    // Sidebar state
    const sbState = Store.getSidebarState();
    if (sbState === 'collapsed') {
      document.getElementById('sidebar').classList.add('collapsed');
      document.getElementById('main-wrapper').classList.add('sidebar-collapsed');
    }

    // Populate project dropdown
    _populateProjectDropdown();

    // Event listeners
    _bindSidebar();
    _bindTheme();
    _bindNotifications();
    _bindKeyboard();

    // Init modals
    Modals.init();

    // Router
    window.addEventListener('hashchange', _onHashChange);
    _onHashChange();

    // Sidebar sprint info
    updateSidebarSprintInfo();
  }

  // =============== Router ===============
  function _onHashChange() {
    const hash = (location.hash || '#dashboard').replace('#', '');
    const route = routes[hash] || routes.dashboard;
    currentPage = hash in routes ? hash : 'dashboard';

    // Update active nav
    document.querySelectorAll('.nav-link').forEach(link => {
      link.classList.toggle('active', link.dataset.page === currentPage);
    });

    // Page title
    document.getElementById('page-title').textContent = route.title;

    // Render
    route.render();
  }

  function renderCurrentPage() {
    const route = routes[currentPage] || routes.dashboard;
    route.render();
  }

  function refreshAll() {
    _populateProjectDropdown();
    updateSidebarSprintInfo();
    renderCurrentPage();
  }

  // =============== Dashboard ===============
  function _renderDashboard() {
    const el = document.getElementById('app-content');
    const project = Store.getActiveProject();

    if (!project) {
      el.innerHTML = `
        <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;height:60vh;text-align:center">
          <i class="fa-solid fa-arrows-spin" style="font-size:4rem;color:var(--accent);margin-bottom:1rem"></i>
          <h2 style="color:var(--text-primary)">Welcome to ScrumFlow</h2>
          <p style="color:var(--text-muted);margin:0.5rem 0 1.5rem">Create a project or load demo data to get started.</p>
          <div style="display:flex;gap:1rem">
            <button onclick="document.getElementById('btn-new-project').click()" class="btn btn-accent"><i class="fa-solid fa-plus"></i> New Project</button>
            <button onclick="document.getElementById('btn-import-sample').click()" class="btn btn-outline"><i class="fa-solid fa-download"></i> Load Demo Data</button>
          </div>
        </div>
      `;
      return;
    }

    const allStories = Store.getAllStories(project);
    const activeSprint = Store.getActiveSprint(project);
    const sprints = project.sprints || [];
    const completedSprints = sprints.filter(s => s.status === 'completed');
    const members = Store.getTeamMembers(project);

    // Stats
    const totalStories = allStories.length;
    const doneStories = allStories.filter(s => s.status === 'done').length;
    const totalPts = allStories.reduce((sum, s) => sum + (s.storyPoints || 0), 0);
    const donePts = allStories.filter(s => s.status === 'done').reduce((sum, s) => sum + (s.storyPoints || 0), 0);

    // Velocity (avg of completed sprints)
    const velocities = completedSprints.map(s => s.velocity || 0);
    const avgVelocity = velocities.length > 0 ? Math.round(velocities.reduce((a, b) => a + b, 0) / velocities.length) : 0;

    // Sprint progress
    let sprintProgress = 0;
    let sprintDaysLeft = '';
    if (activeSprint) {
      const sprintStories = activeSprint.stories || [];
      const sprintDone = sprintStories.filter(s => s.status === 'done').length;
      sprintProgress = sprintStories.length > 0 ? Math.round((sprintDone / sprintStories.length) * 100) : 0;
      const days = Components.daysRemaining(activeSprint.endDate);
      sprintDaysLeft = days >= 0 ? `${days} days left` : 'Overdue';
    }

    const recentActivity = Store.getActivityLog().slice(0, 8);

    el.innerHTML = `
      <div class="dashboard-page">
        <!-- KPI Cards -->
        <div class="kpi-grid">
          <div class="kpi-card">
            <div class="kpi-icon" style="background:var(--accent-alpha)"><i class="fa-solid fa-book" style="color:var(--accent)"></i></div>
            <div class="kpi-data">
              <div class="kpi-value">${totalStories}</div>
              <div class="kpi-label">Total Stories</div>
            </div>
          </div>
          <div class="kpi-card">
            <div class="kpi-icon" style="background:rgba(16,185,129,0.15)"><i class="fa-solid fa-circle-check" style="color:#10b981"></i></div>
            <div class="kpi-data">
              <div class="kpi-value">${doneStories}</div>
              <div class="kpi-label">Completed</div>
            </div>
          </div>
          <div class="kpi-card">
            <div class="kpi-icon" style="background:rgba(245,158,11,0.15)"><i class="fa-solid fa-fire" style="color:#f59e0b"></i></div>
            <div class="kpi-data">
              <div class="kpi-value">${avgVelocity}</div>
              <div class="kpi-label">Avg Velocity</div>
            </div>
          </div>
          <div class="kpi-card">
            <div class="kpi-icon" style="background:rgba(99,102,241,0.15)"><i class="fa-solid fa-users" style="color:var(--accent)"></i></div>
            <div class="kpi-data">
              <div class="kpi-value">${members.length}</div>
              <div class="kpi-label">Team Members</div>
            </div>
          </div>
        </div>

        <!-- Sprint + Progress Row -->
        <div class="dashboard-row">
          <div class="dash-card">
            <h3><i class="fa-solid fa-rocket"></i> Active Sprint</h3>
            ${activeSprint ? `
              <div class="dash-sprint-info">
                <h4>${Components._esc(activeSprint.name)}</h4>
                <p style="color:var(--text-muted);font-size:0.8rem">${Components._esc(activeSprint.goal || '')}</p>
                <div style="display:flex;align-items:center;gap:0.75rem;margin-top:0.75rem">
                  ${Components.progressBar(sprintProgress)}
                  <span style="font-weight:600;color:var(--accent)">${sprintProgress}%</span>
                </div>
                <p style="color:var(--text-muted);font-size:0.75rem;margin-top:0.5rem">${sprintDaysLeft}</p>
              </div>
            ` : '<p style="color:var(--text-muted)">No active sprint. <a href="#sprints" style="color:var(--accent)">Go to Sprints →</a></p>'}
          </div>
          <div class="dash-card">
            <h3><i class="fa-solid fa-chart-pie"></i> Story Breakdown</h3>
            <div class="status-breakdown">
              ${_statusCount(allStories, 'todo', 'To Do', 'var(--text-muted)')}
              ${_statusCount(allStories, 'inprogress', 'In Progress', 'var(--accent)')}
              ${_statusCount(allStories, 'review', 'In Review', 'var(--warning)')}
              ${_statusCount(allStories, 'done', 'Done', 'var(--success)')}
            </div>
          </div>
          <div class="dash-card">
            <h3><i class="fa-solid fa-bullseye"></i> Points Progress</h3>
            <div style="text-align:center;padding:1rem 0">
              <div style="position:relative;width:120px;height:120px;margin:0 auto">
                <svg viewBox="0 0 120 120" style="transform:rotate(-90deg)">
                  <circle cx="60" cy="60" r="50" fill="none" stroke="var(--border)" stroke-width="10"/>
                  <circle cx="60" cy="60" r="50" fill="none" stroke="var(--accent)" stroke-width="10"
                    stroke-dasharray="${totalPts > 0 ? (donePts / totalPts) * 314 : 0} 314"
                    stroke-linecap="round"/>
                </svg>
                <div style="position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center">
                  <span style="font-size:1.5rem;font-weight:800;color:var(--text-primary)">${donePts}</span>
                  <span style="font-size:0.65rem;color:var(--text-muted)">of ${totalPts} pts</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        <!-- Recent Activity -->
        <div class="dash-card dash-full-width">
          <h3><i class="fa-solid fa-clock-rotate-left"></i> Recent Activity</h3>
          ${recentActivity.length > 0 ? `
            <div class="activity-feed">
              ${recentActivity.map(a => `
                <div class="activity-feed-item">
                  <div class="activity-dot"></div>
                  <div class="activity-text">${Components._esc(a.text)}</div>
                  <div class="activity-time">${Components.timeAgo(a.timestamp)}</div>
                </div>
              `).join('')}
            </div>
          ` : '<p style="color:var(--text-muted)">No activity yet</p>'}
        </div>
      </div>
    `;
  }

  function _statusCount(stories, status, label, color) {
    const count = stories.filter(s => s.status === status).length;
    return `
      <div class="status-count-item">
        <span class="status-dot" style="background:${color}"></span>
        <span class="status-label">${label}</span>
        <span class="status-value">${count}</span>
      </div>
    `;
  }

  // =============== Sidebar ===============
  function _bindSidebar() {
    document.getElementById('sidebar-toggle')?.addEventListener('click', () => {
      const sidebar = document.getElementById('sidebar');
      const main = document.getElementById('main-wrapper');
      sidebar.classList.toggle('collapsed');
      main.classList.toggle('sidebar-collapsed');
      Store.setSidebarState(sidebar.classList.contains('collapsed') ? 'collapsed' : 'expanded');
    });

    document.getElementById('btn-import-sample')?.addEventListener('click', async () => {
      const yes = await Components.confirm('Load Demo Data', 'This will add a sample project with stories, sprints, and team members. Continue?');
      if (yes) {
        SampleData.load();
        Components.showToast('Demo data loaded!', 'success');
        Components.showSaved();
        refreshAll();
      }
    });
  }

  // =============== Theme ===============
  function _bindTheme() {
    document.getElementById('btn-theme-toggle')?.addEventListener('click', () => {
      const isDark = document.body.classList.contains('dark-mode');
      document.body.classList.toggle('dark-mode', !isDark);
      document.body.classList.toggle('light-mode', isDark);
      Store.setTheme(isDark ? 'light' : 'dark');
      _updateThemeIcon();
    });
  }

  function _updateThemeIcon() {
    const btn = document.getElementById('btn-theme-toggle');
    if (!btn) return;
    const isDark = document.body.classList.contains('dark-mode');
    btn.innerHTML = isDark
      ? '<i class="fa-solid fa-moon"></i>'
      : '<i class="fa-solid fa-sun"></i>';
  }

  // =============== Project Dropdown ===============
  function _populateProjectDropdown() {
    const dropdown = document.getElementById('project-dropdown');
    const projects = Store.getProjects();
    const activeId = Store.getActiveProjectId();

    dropdown.innerHTML = '';

    if (projects.length === 0) {
      dropdown.innerHTML = '<option value="">No projects</option>';
      return;
    }

    projects.forEach(p => {
      const opt = document.createElement('option');
      opt.value = p.id;
      opt.textContent = p.name;
      opt.selected = p.id === activeId;
      dropdown.appendChild(opt);
    });

    dropdown.addEventListener('change', (e) => {
      Store.setActiveProjectId(e.target.value);
      refreshAll();
    });
  }

  // =============== Notifications ===============
  function _bindNotifications() {
    const btn = document.getElementById('btn-notification');
    const panel = document.getElementById('notification-panel');

    btn?.addEventListener('click', () => {
      panel.classList.toggle('hidden');
      if (!panel.classList.contains('hidden')) _renderNotifications();
    });

    document.getElementById('btn-clear-notifications')?.addEventListener('click', () => {
      Store.clearActivityLog();
      _renderNotifications();
      document.getElementById('notification-badge').textContent = '0';
      Components.showToast('Notifications cleared', 'info');
    });

    // Close panel on outside click
    document.addEventListener('click', (e) => {
      if (!panel.contains(e.target) && !btn.contains(e.target)) {
        panel.classList.add('hidden');
      }
    });

    // Update badge
    _updateNotificationBadge();
  }

  function _renderNotifications() {
    const list = document.getElementById('notification-list');
    const logs = Store.getActivityLog().slice(0, 20);
    if (logs.length === 0) {
      list.innerHTML = '<p style="text-align:center;color:var(--text-muted);padding:2rem">No notifications</p>';
      return;
    }
    list.innerHTML = logs.map(l => `
      <div class="notification-item">
        <div class="notification-text">${Components._esc(l.text)}</div>
        <div class="notification-time">${Components.timeAgo(l.timestamp)}</div>
      </div>
    `).join('');
  }

  function _updateNotificationBadge() {
    const logs = Store.getActivityLog();
    const badge = document.getElementById('notification-badge');
    badge.textContent = Math.min(logs.length, 99);
    badge.style.display = logs.length > 0 ? 'flex' : 'none';
  }

  // =============== Sidebar Sprint Info ===============
  function updateSidebarSprintInfo() {
    const project = Store.getActiveProject();
    const sprint = project ? Store.getActiveSprint(project) : null;
    const nameEl = document.getElementById('sidebar-sprint-name');
    const daysEl = document.getElementById('sidebar-sprint-days');
    const badgeEl = document.getElementById('header-sprint-badge');

    if (sprint) {
      nameEl.textContent = sprint.name;
      const days = Components.daysRemaining(sprint.endDate);
      daysEl.textContent = days >= 0 ? `${days} days remaining` : 'Sprint overdue!';
      daysEl.style.color = days < 3 ? 'var(--error)' : 'var(--text-muted)';
      badgeEl.innerHTML = `<i class="fa-solid fa-rocket"></i> ${sprint.name}`;
      badgeEl.style.display = 'inline-flex';
    } else {
      nameEl.textContent = 'No active sprint';
      daysEl.textContent = '';
      badgeEl.style.display = 'none';
    }
  }

  // =============== Keyboard Shortcuts ===============
  function _bindKeyboard() {
    document.addEventListener('keydown', (e) => {
      // Don't fire if inside an input/textarea
      const tag = (e.target.tagName || '').toLowerCase();
      const isInput = tag === 'input' || tag === 'textarea' || tag === 'select';

      // Ctrl+K — search
      if (e.ctrlKey && e.key === 'k') {
        e.preventDefault();
        Modals.openSearch();
        return;
      }

      // Ctrl+Z — undo
      if (e.ctrlKey && e.key === 'z' && !isInput) {
        e.preventDefault();
        const action = Store.popUndo();
        if (action) {
          Components.showToast(`Undo: ${action.description || 'last action'}`, 'info');
          // Restore previous state if available
          if (action.restore) {
            try {
              action.restore();
            } catch (_) {}
          }
          renderCurrentPage();
        } else {
          Components.showToast('Nothing to undo', 'warning');
        }
        return;
      }

      if (isInput) return;

      // B → Board
      if (e.key === 'b' || e.key === 'B') {
        location.hash = '#board';
        return;
      }

      // N → New story
      if (e.key === 'n' || e.key === 'N') {
        Modals.openCardDetail(null);
        return;
      }

      // D → Dashboard
      if (e.key === 'd' || e.key === 'D') {
        location.hash = '#dashboard';
        return;
      }
    });
  }

  return {
    init, renderCurrentPage, refreshAll, updateSidebarSprintInfo
  };
})();

// =============== Bootstrap ===============
document.addEventListener('DOMContentLoaded', () => {
  App.init();
});
