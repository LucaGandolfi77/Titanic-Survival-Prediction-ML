/* =====================================================
   ANALYTICS.JS — Chart.js setup + data prep
   ===================================================== */

const AnalyticsView = (() => {
  let charts = {};

  function render() {
    const project = Store.getActiveProject();
    const content = document.getElementById('app-content');

    if (!project) {
      content.innerHTML = '<div class="empty-state"><i class="fa-solid fa-chart-line"></i><p>No project selected.</p></div>';
      return;
    }

    const sprints = Store.getSprints(project);
    const activeSprint = Store.getActiveSprint(project);
    const completedSprints = sprints.filter(s => s.status === 'completed');
    const allStories = Store.getAllStories(project);

    content.innerHTML = `
      <div class="page-enter">
        <div class="page-header">
          <h2>Analytics</h2>
          <div class="page-header-actions">
            <button class="btn btn-outline" id="btn-export-report"><i class="fa-solid fa-file-export"></i> Export Sprint Report</button>
          </div>
        </div>
        <div class="analytics-grid">
          <!-- Burndown Chart -->
          <div class="chart-card">
            <h3><i class="fa-solid fa-fire"></i> Sprint Burndown</h3>
            <div class="chart-container">
              <canvas id="burndown-chart"></canvas>
            </div>
            ${!activeSprint ? '<p style="text-align:center;color:var(--text-muted);font-size:0.8rem;margin-top:0.5rem">No active sprint</p>' : ''}
          </div>

          <!-- Velocity Chart -->
          <div class="chart-card">
            <h3><i class="fa-solid fa-gauge-high"></i> Velocity</h3>
            <div class="chart-container">
              <canvas id="velocity-chart"></canvas>
            </div>
            ${completedSprints.length === 0 ? '<p style="text-align:center;color:var(--text-muted);font-size:0.8rem;margin-top:0.5rem">No completed sprints yet</p>' : ''}
          </div>

          <!-- Story Distribution -->
          <div class="chart-card">
            <h3><i class="fa-solid fa-chart-pie"></i> Story Distribution</h3>
            <div class="chart-container" style="max-height:280px">
              <canvas id="distribution-chart"></canvas>
            </div>
          </div>

          <!-- Epic Progress -->
          <div class="chart-card">
            <h3><i class="fa-solid fa-layer-group"></i> Epic Progress</h3>
            <div class="chart-container">
              <canvas id="epic-chart"></canvas>
            </div>
          </div>

          <!-- Sprint Comparison Table -->
          <div class="chart-card full-width">
            <h3><i class="fa-solid fa-table"></i> Sprint Comparison</h3>
            <table class="comparison-table">
              <thead>
                <tr>
                  <th>Sprint</th>
                  <th>Planned Points</th>
                  <th>Completed Points</th>
                  <th>Velocity</th>
                  <th>Duration</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                ${sprints.map(s => {
                  const planned = (s.stories || []).reduce((sum, st) => sum + (st.storyPoints || 0), 0);
                  const done = (s.stories || []).filter(st => st.status === 'done').reduce((sum, st) => sum + (st.storyPoints || 0), 0);
                  const start = new Date(s.startDate);
                  const end = new Date(s.endDate);
                  const days = s.startDate && s.endDate ? Math.ceil((end - start) / (1000 * 60 * 60 * 24)) : '—';
                  return `
                    <tr>
                      <td><strong>${Components._esc(s.name)}</strong></td>
                      <td>${planned}</td>
                      <td>${done}</td>
                      <td>${s.velocity || done}</td>
                      <td>${days} days</td>
                      <td>${Components.sprintStatusBadge(s.status)}</td>
                    </tr>`;
                }).join('')}
                ${sprints.length === 0 ? '<tr><td colspan="6" style="text-align:center;color:var(--text-muted)">No sprints</td></tr>' : ''}
              </tbody>
            </table>
          </div>
        </div>
      </div>`;

    // Render charts after DOM is ready
    setTimeout(() => {
      _renderBurndownChart(activeSprint);
      _renderVelocityChart(sprints);
      _renderDistributionChart(allStories);
      _renderEpicChart(project);
    }, 100);

    // Export button
    document.getElementById('btn-export-report')?.addEventListener('click', () => _exportReport(project));
  }

  function _destroyCharts() {
    Object.values(charts).forEach(c => { try { c.destroy(); } catch(e) {} });
    charts = {};
  }

  function _getChartColors() {
    const isDark = document.body.classList.contains('dark-mode');
    return {
      text: isDark ? '#94a3b8' : '#64748b',
      grid: isDark ? 'rgba(45,49,72,0.5)' : 'rgba(226,232,240,0.5)',
      bg: isDark ? '#222538' : '#ffffff'
    };
  }

  function _renderBurndownChart(sprint) {
    _destroyCharts();
    const canvas = document.getElementById('burndown-chart');
    if (!canvas || !sprint) return;

    const colors = _getChartColors();
    const burndown = sprint.burndownData || [];
    if (burndown.length === 0) return;

    const totalPoints = burndown[0]?.remaining || 0;
    const start = new Date(sprint.startDate);
    const end = new Date(sprint.endDate);
    const totalDays = Math.max(1, Math.ceil((end - start) / (1000 * 60 * 60 * 24)));

    // Ideal line
    const idealData = [];
    for (let i = 0; i <= totalDays; i++) {
      idealData.push({ x: i, y: totalPoints - (totalPoints / totalDays) * i });
    }

    // Actual line
    const actualData = burndown.map(d => ({ x: d.day, y: d.remaining }));

    charts.burndown = new Chart(canvas, {
      type: 'line',
      data: {
        datasets: [
          {
            label: 'Ideal',
            data: idealData,
            borderColor: 'rgba(99,102,241,0.4)',
            borderDash: [5, 5],
            fill: false,
            tension: 0,
            pointRadius: 0
          },
          {
            label: 'Actual',
            data: actualData,
            borderColor: '#6366f1',
            backgroundColor: 'rgba(99,102,241,0.1)',
            fill: true,
            tension: 0.3,
            pointRadius: 4,
            pointBackgroundColor: '#6366f1'
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: 'index', intersect: false },
        scales: {
          x: {
            type: 'linear',
            title: { display: true, text: 'Sprint Day', color: colors.text },
            ticks: { color: colors.text, stepSize: 1 },
            grid: { color: colors.grid }
          },
          y: {
            title: { display: true, text: 'Points Remaining', color: colors.text },
            ticks: { color: colors.text },
            grid: { color: colors.grid },
            beginAtZero: true
          }
        },
        plugins: {
          legend: { labels: { color: colors.text } },
          tooltip: {
            callbacks: {
              label: ctx => `${ctx.dataset.label}: ${Math.round(ctx.parsed.y)} points`
            }
          }
        }
      }
    });
  }

  function _renderVelocityChart(sprints) {
    const canvas = document.getElementById('velocity-chart');
    if (!canvas) return;

    const colors = _getChartColors();
    const completed = sprints.filter(s => s.status === 'completed').slice(-6);
    if (completed.length === 0) return;

    const labels = completed.map(s => s.name);
    const velocities = completed.map(s => s.velocity || 0);
    const avg = velocities.length > 0 ? velocities.reduce((a, b) => a + b, 0) / velocities.length : 0;

    charts.velocity = new Chart(canvas, {
      type: 'bar',
      data: {
        labels,
        datasets: [
          {
            label: 'Velocity',
            data: velocities,
            backgroundColor: 'rgba(99,102,241,0.7)',
            borderColor: '#6366f1',
            borderWidth: 1,
            borderRadius: 4
          },
          {
            label: 'Average',
            data: labels.map(() => avg),
            type: 'line',
            borderColor: '#f59e0b',
            borderDash: [5, 5],
            fill: false,
            pointRadius: 0,
            borderWidth: 2
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: { ticks: { color: colors.text }, grid: { color: colors.grid } },
          y: { ticks: { color: colors.text }, grid: { color: colors.grid }, beginAtZero: true }
        },
        plugins: { legend: { labels: { color: colors.text } } }
      }
    });
  }

  function _renderDistributionChart(stories) {
    const canvas = document.getElementById('distribution-chart');
    if (!canvas) return;

    const colors = _getChartColors();
    const counts = { todo: 0, inprogress: 0, review: 0, done: 0 };
    stories.forEach(s => { if (counts.hasOwnProperty(s.status)) counts[s.status]++; });

    charts.distribution = new Chart(canvas, {
      type: 'doughnut',
      data: {
        labels: ['To Do', 'In Progress', 'In Review', 'Done'],
        datasets: [{
          data: [counts.todo, counts.inprogress, counts.review, counts.done],
          backgroundColor: ['#94a3b8', '#6366f1', '#f59e0b', '#22c55e'],
          borderColor: colors.bg,
          borderWidth: 3
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: 'bottom', labels: { color: colors.text, padding: 15 } }
        }
      }
    });
  }

  function _renderEpicChart(project) {
    const canvas = document.getElementById('epic-chart');
    if (!canvas) return;

    const colors = _getChartColors();
    const epics = Store.getEpics(project);
    if (epics.length === 0) return;

    const labels = epics.map(e => e.name);
    const doneData = [];
    const totalData = [];

    epics.forEach(epic => {
      const stories = Store.getEpicStories(epic.id, project);
      const total = stories.reduce((s, st) => s + (st.storyPoints || 0), 0);
      const done = stories.filter(s => s.status === 'done').reduce((s, st) => s + (st.storyPoints || 0), 0);
      totalData.push(total - done);
      doneData.push(done);
    });

    charts.epic = new Chart(canvas, {
      type: 'bar',
      data: {
        labels,
        datasets: [
          {
            label: 'Done',
            data: doneData,
            backgroundColor: '#22c55e',
            borderRadius: 4
          },
          {
            label: 'Remaining',
            data: totalData,
            backgroundColor: 'rgba(148,163,184,0.3)',
            borderRadius: 4
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: { stacked: true, ticks: { color: colors.text }, grid: { color: colors.grid } },
          y: { stacked: true, ticks: { color: colors.text }, grid: { color: colors.grid }, beginAtZero: true }
        },
        plugins: { legend: { labels: { color: colors.text } } }
      }
    });
  }

  function _exportReport(project) {
    const activeSprint = Store.getActiveSprint(project);
    const sprints = Store.getSprints(project);

    const report = {
      project: project.name,
      exportDate: new Date().toISOString(),
      sprints: sprints.map(s => ({
        name: s.name,
        status: s.status,
        goal: s.goal,
        dates: `${s.startDate} → ${s.endDate}`,
        velocity: s.velocity,
        stories: (s.stories || []).map(st => ({
          title: st.title,
          status: st.status,
          points: st.storyPoints,
          priority: st.priority
        })),
        retrospective: s.retrospective
      }))
    };

    const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `sprint-report-${project.name.replace(/\s+/g, '-')}.json`;
    a.click();
    URL.revokeObjectURL(url);
    Components.showToast('Report exported!', 'success');
  }

  return { render };
})();
