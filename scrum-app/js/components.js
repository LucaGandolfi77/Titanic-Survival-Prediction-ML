/* =====================================================
   COMPONENTS.JS — Reusable render functions
   ===================================================== */

const Components = (() => {

  // ---- Avatar ----
  function avatar(member, size = 'sm') {
    if (!member) return `<span class="avatar avatar-${size}" style="background:var(--border)">?</span>`;
    return `<span class="avatar avatar-${size}" style="background:${member.color || '#6366f1'}" title="${member.name}">${member.avatar || '?'}</span>`;
  }

  // ---- Story Points Badge ----
  function pointsBadge(points) {
    if (!points) return '';
    const cls = `points-${points}`;
    return `<span class="card-points ${cls}">${points}</span>`;
  }

  function pointsBadgeSmall(points) {
    if (!points) return '';
    let bg = 'var(--accent)';
    if (points <= 2) bg = 'var(--success)';
    else if (points <= 5) bg = 'var(--accent)';
    else if (points <= 13) bg = 'var(--warning)';
    else bg = 'var(--danger)';
    return `<span class="backlog-story-points" style="background:${bg}">${points}</span>`;
  }

  // ---- Priority Flag ----
  function priorityFlag(priority) {
    return `<div class="priority-flag ${priority || 'medium'}"></div>`;
  }

  function priorityDot(priority) {
    return `<span class="backlog-priority-dot ${priority || 'medium'}"></span>`;
  }

  function priorityLabel(priority) {
    const labels = { critical: '🔴 Critical', high: '🟠 High', medium: '🟡 Medium', low: '🔵 Low' };
    return labels[priority] || priority;
  }

  // ---- Epic Tag ----
  function epicTag(epic) {
    if (!epic) return '';
    return `<span class="card-epic-tag" style="background:${epic.color}">${epic.name}</span>`;
  }

  // ---- Status Badge ----
  function statusBadge(status) {
    const labels = { todo: 'To Do', inprogress: 'In Progress', review: 'In Review', done: 'Done' };
    const colors = { todo: 'badge-accent', inprogress: 'badge-accent', review: 'badge-warning', done: 'badge-success' };
    return `<span class="badge ${colors[status] || ''}">${labels[status] || status}</span>`;
  }

  // ---- Sprint Status Badge ----
  function sprintStatusBadge(status) {
    return `<span class="sprint-status-badge ${status}">${status.charAt(0).toUpperCase() + status.slice(1)}</span>`;
  }

  // ---- Role Badge ----
  function roleBadge(role) {
    const cls = 'role-' + role.replace(/\s+/g, '');
    return `<span class="team-role-badge ${cls}">${role}</span>`;
  }

  // ---- Subtask Progress ----
  function subtaskProgress(subtasks) {
    if (!subtasks || subtasks.length === 0) return '';
    const done = subtasks.filter(s => s.completed).length;
    const total = subtasks.length;
    const pct = Math.round((done / total) * 100);
    return `
      <span class="card-subtask-bar">
        <span class="mini-progress"><span class="mini-progress-fill" style="width:${pct}%"></span></span>
        ${done}/${total}
      </span>`;
  }

  // ---- Progress Bar ----
  function progressBar(pct, cls = '') {
    return `
      <div class="progress-bar">
        <div class="progress-fill ${cls}" style="width:${Math.min(100, Math.max(0, pct))}%"></div>
      </div>`;
  }

  // ---- Story Card (Board) ----
  function storyCard(story, project) {
    const member = story.assigneeId ? Store.getTeamMember(story.assigneeId, project) : null;
    const epic = story.epicId ? Store.getEpicById(story.epicId, project) : null;
    const tagsHtml = (story.tags || []).map(t => `<span class="card-tag-chip">${t}</span>`).join('');

    return `
      <div class="story-card" draggable="true" data-story-id="${story.id}" data-status="${story.status}">
        ${priorityFlag(story.priority)}
        <div class="card-header">
          <span class="card-title">${_esc(story.title)}</span>
          ${pointsBadge(story.storyPoints)}
        </div>
        <div class="card-footer">
          <div class="card-meta">
            ${epicTag(epic)}
            ${subtaskProgress(story.subtasks)}
          </div>
          ${avatar(member, 'sm')}
        </div>
        ${tagsHtml ? `<div class="card-tags">${tagsHtml}</div>` : ''}
      </div>`;
  }

  // ---- Backlog Story Row ----
  function backlogStoryRow(story, project, options = {}) {
    const member = story.assigneeId ? Store.getTeamMember(story.assigneeId, project) : null;
    const epic = story.epicId ? Store.getEpicById(story.epicId, project) : null;

    return `
      <div class="backlog-story-row" draggable="true" data-story-id="${story.id}" data-sprint-id="${story.sprintId || ''}">
        <span class="backlog-drag-handle"><i class="fa-solid fa-grip-vertical"></i></span>
        <input type="checkbox" class="backlog-story-checkbox" data-story-id="${story.id}">
        ${priorityDot(story.priority)}
        <span class="backlog-story-title" data-story-id="${story.id}">${_esc(story.title)}</span>
        ${pointsBadgeSmall(story.storyPoints)}
        ${member ? avatar(member, 'sm') : '<span class="avatar avatar-sm" style="background:var(--border);opacity:0.4">?</span>'}
        ${epic ? `<span class="backlog-story-epic" style="background:${epic.color}">${_esc(epic.name)}</span>` : ''}
        <span class="backlog-story-actions">
          <button class="btn-edit-story" data-story-id="${story.id}" title="Edit"><i class="fa-solid fa-pen"></i></button>
          <button class="btn-delete-story" data-story-id="${story.id}" title="Delete"><i class="fa-solid fa-trash"></i></button>
        </span>
      </div>`;
  }

  // ---- Time Ago ----
  function timeAgo(dateStr) {
    if (!dateStr) return '';
    const now = new Date();
    const d = new Date(dateStr);
    const diff = Math.floor((now - d) / 1000);
    if (diff < 60) return 'just now';
    if (diff < 3600) return Math.floor(diff / 60) + 'm ago';
    if (diff < 86400) return Math.floor(diff / 3600) + 'h ago';
    if (diff < 604800) return Math.floor(diff / 86400) + 'd ago';
    return d.toLocaleDateString();
  }

  function formatDate(dateStr) {
    if (!dateStr) return '—';
    return new Date(dateStr).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
  }

  function formatDateShort(dateStr) {
    if (!dateStr) return '—';
    return new Date(dateStr).toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
  }

  // ---- Escape HTML ----
  function _esc(str) {
    if (!str) return '';
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
  }

  // ---- Simple Markdown ----
  function renderMarkdown(text) {
    if (!text) return '';
    let html = _esc(text);
    // Headers
    html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');
    html = html.replace(/^## (.+)$/gm, '<h2>$1</h2>');
    html = html.replace(/^# (.+)$/gm, '<h1>$1</h1>');
    // Bold
    html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    // Italic
    html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');
    // Code
    html = html.replace(/`(.+?)`/g, '<code>$1</code>');
    // Lists
    html = html.replace(/^- (.+)$/gm, '<li>$1</li>');
    html = html.replace(/(<li>.*<\/li>)/s, '<ul>$1</ul>');
    // Line breaks
    html = html.replace(/\n/g, '<br>');
    return html;
  }

  // ---- Toast ----
  function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    const icons = { success: 'fa-circle-check', warning: 'fa-triangle-exclamation', danger: 'fa-circle-xmark', info: 'fa-circle-info' };
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.innerHTML = `<i class="fa-solid ${icons[type] || icons.info}"></i><span>${message}</span>`;
    container.appendChild(toast);
    setTimeout(() => toast.remove(), 3000);
  }

  // ---- Autosave indicator ----
  function showSaved() {
    const badge = document.getElementById('autosave-indicator');
    badge.classList.remove('hidden');
    setTimeout(() => badge.classList.add('hidden'), 1500);
  }

  // ---- Confirm Dialog ----
  function confirm(title, message) {
    return new Promise((resolve) => {
      document.getElementById('confirm-title').textContent = title;
      document.getElementById('confirm-message').textContent = message;
      document.getElementById('modal-confirm').classList.remove('hidden');
      
      const yesBtn = document.getElementById('btn-confirm-yes');
      const noBtn = document.getElementById('btn-confirm-no');
      
      const cleanup = () => {
        document.getElementById('modal-confirm').classList.add('hidden');
        yesBtn.removeEventListener('click', onYes);
        noBtn.removeEventListener('click', onNo);
      };
      
      const onYes = () => { cleanup(); resolve(true); };
      const onNo = () => { cleanup(); resolve(false); };
      
      yesBtn.addEventListener('click', onYes);
      noBtn.addEventListener('click', onNo);
    });
  }

  // ---- Days remaining ----
  function daysRemaining(endDate) {
    if (!endDate) return 0;
    const end = new Date(endDate);
    const now = new Date();
    return Math.max(0, Math.ceil((end - now) / (1000 * 60 * 60 * 24)));
  }

  return {
    avatar, pointsBadge, pointsBadgeSmall, priorityFlag, priorityDot, priorityLabel,
    epicTag, statusBadge, sprintStatusBadge, roleBadge, subtaskProgress, progressBar,
    storyCard, backlogStoryRow,
    timeAgo, formatDate, formatDateShort,
    renderMarkdown, showToast, showSaved, confirm, daysRemaining,
    _esc
  };
})();
