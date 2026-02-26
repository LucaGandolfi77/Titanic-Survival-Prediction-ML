/* =====================================================
   STORE.JS — localStorage CRUD (get/set/delete per entity)
   ===================================================== */

const Store = (() => {
  const KEYS = {
    PROJECTS: 'scrumflow_projects',
    ACTIVE_PROJECT: 'scrumflow_active_project',
    ACTIVITY_LOG: 'scrumflow_activity_log',
    DOD_CHECKLIST: 'scrumflow_dod',
    WIP_LIMITS: 'scrumflow_wip_limits',
    THEME: 'scrumflow_theme',
    SIDEBAR: 'scrumflow_sidebar',
    UNDO_STACK: 'scrumflow_undo'
  };

  // ---- Helpers ----
  function _get(key) {
    try {
      const data = localStorage.getItem(key);
      return data ? JSON.parse(data) : null;
    } catch (e) {
      console.error('Store._get error:', e);
      return null;
    }
  }

  function _set(key, value) {
    try {
      localStorage.setItem(key, JSON.stringify(value));
    } catch (e) {
      console.error('Store._set error:', e);
    }
  }

  function _generateId() {
    return Date.now().toString(36) + Math.random().toString(36).substr(2, 9);
  }

  function _now() {
    return new Date().toISOString();
  }

  // ---- Projects ----
  function getProjects() {
    return _get(KEYS.PROJECTS) || [];
  }

  function setProjects(projects) {
    _set(KEYS.PROJECTS, projects);
  }

  function getActiveProjectId() {
    return _get(KEYS.ACTIVE_PROJECT);
  }

  function setActiveProjectId(id) {
    _set(KEYS.ACTIVE_PROJECT, id);
  }

  function getActiveProject() {
    const projects = getProjects();
    const activeId = getActiveProjectId();
    return projects.find(p => p.id === activeId) || projects[0] || null;
  }

  function createProject(name, description = '') {
    const projects = getProjects();
    const project = {
      id: _generateId(),
      name,
      description,
      createdAt: _now(),
      teamMembers: [],
      sprints: [],
      backlog: [],
      epics: []
    };
    projects.push(project);
    setProjects(projects);
    setActiveProjectId(project.id);
    return project;
  }

  function updateProject(projectId, updates) {
    const projects = getProjects();
    const idx = projects.findIndex(p => p.id === projectId);
    if (idx !== -1) {
      projects[idx] = { ...projects[idx], ...updates };
      setProjects(projects);
    }
    return projects[idx];
  }

  function saveProject(project) {
    const projects = getProjects();
    const idx = projects.findIndex(p => p.id === project.id);
    if (idx !== -1) {
      projects[idx] = project;
    } else {
      projects.push(project);
    }
    setProjects(projects);
  }

  function deleteProject(projectId) {
    let projects = getProjects();
    projects = projects.filter(p => p.id !== projectId);
    setProjects(projects);
    if (getActiveProjectId() === projectId) {
      setActiveProjectId(projects[0]?.id || null);
    }
  }

  // ---- Stories ----
  function getAllStories(project) {
    if (!project) project = getActiveProject();
    if (!project) return [];
    const stories = [...(project.backlog || [])];
    (project.sprints || []).forEach(s => {
      (s.stories || []).forEach(st => stories.push(st));
    });
    return stories;
  }

  function getStoryById(storyId, project) {
    if (!project) project = getActiveProject();
    if (!project) return null;
    // Check backlog
    let story = (project.backlog || []).find(s => s.id === storyId);
    if (story) return story;
    // Check sprints
    for (const sprint of (project.sprints || [])) {
      story = (sprint.stories || []).find(s => s.id === storyId);
      if (story) return story;
    }
    return null;
  }

  function createStory(data, project) {
    if (!project) project = getActiveProject();
    if (!project) return null;
    const story = {
      id: _generateId(),
      title: data.title || 'Untitled Story',
      description: data.description || '',
      acceptanceCriteria: data.acceptanceCriteria || '',
      priority: data.priority || 'medium',
      storyPoints: data.storyPoints || 0,
      status: data.status || 'todo',
      sprintId: data.sprintId || null,
      assigneeId: data.assigneeId || null,
      epicId: data.epicId || null,
      tags: data.tags || [],
      subtasks: data.subtasks || [],
      createdAt: _now(),
      updatedAt: _now(),
      completedAt: null,
      activityLog: [{ text: 'Story created', timestamp: _now() }]
    };

    if (story.sprintId) {
      const sprint = (project.sprints || []).find(s => s.id === story.sprintId);
      if (sprint) {
        sprint.stories = sprint.stories || [];
        sprint.stories.push(story);
      } else {
        story.sprintId = null;
        project.backlog = project.backlog || [];
        project.backlog.push(story);
      }
    } else {
      project.backlog = project.backlog || [];
      project.backlog.push(story);
    }

    saveProject(project);
    addActivity(`Created story "${story.title}"`);
    return story;
  }

  function updateStory(storyId, updates, project) {
    if (!project) project = getActiveProject();
    if (!project) return null;

    const oldSprintId = _findStorySprintId(storyId, project);
    let story = null;

    // Find and update story
    const backlogIdx = (project.backlog || []).findIndex(s => s.id === storyId);
    if (backlogIdx !== -1) {
      story = project.backlog[backlogIdx];
    } else {
      for (const sprint of (project.sprints || [])) {
        const idx = (sprint.stories || []).findIndex(s => s.id === storyId);
        if (idx !== -1) {
          story = sprint.stories[idx];
          break;
        }
      }
    }

    if (!story) return null;

    // Track status change
    if (updates.status && updates.status !== story.status) {
      story.activityLog = story.activityLog || [];
      const assignee = getTeamMember(story.assigneeId, project);
      const name = assignee ? assignee.name : 'Someone';
      story.activityLog.push({
        text: `Moved to ${_statusLabel(updates.status)} by ${name}`,
        timestamp: _now()
      });
      if (updates.status === 'done') {
        updates.completedAt = _now();
      }
    }

    Object.assign(story, updates, { updatedAt: _now() });

    // Handle sprint change
    const newSprintId = story.sprintId;
    if (newSprintId !== oldSprintId) {
      _removeStoryFromLocation(storyId, project);
      if (newSprintId) {
        const sprint = (project.sprints || []).find(s => s.id === newSprintId);
        if (sprint) {
          sprint.stories = sprint.stories || [];
          sprint.stories.push(story);
        }
      } else {
        project.backlog = project.backlog || [];
        project.backlog.push(story);
      }
    }

    saveProject(project);
    return story;
  }

  function deleteStory(storyId, project) {
    if (!project) project = getActiveProject();
    if (!project) return;

    const story = getStoryById(storyId, project);
    _removeStoryFromLocation(storyId, project);
    saveProject(project);
    if (story) addActivity(`Deleted story "${story.title}"`);
  }

  function moveStoryToStatus(storyId, newStatus, project) {
    return updateStory(storyId, { status: newStatus }, project);
  }

  function moveStoryToSprint(storyId, sprintId, project) {
    return updateStory(storyId, { sprintId: sprintId }, project);
  }

  function _findStorySprintId(storyId, project) {
    if ((project.backlog || []).find(s => s.id === storyId)) return null;
    for (const sprint of (project.sprints || [])) {
      if ((sprint.stories || []).find(s => s.id === storyId)) return sprint.id;
    }
    return null;
  }

  function _removeStoryFromLocation(storyId, project) {
    project.backlog = (project.backlog || []).filter(s => s.id !== storyId);
    for (const sprint of (project.sprints || [])) {
      sprint.stories = (sprint.stories || []).filter(s => s.id !== storyId);
    }
  }

  function _statusLabel(status) {
    const labels = { todo: 'To Do', inprogress: 'In Progress', review: 'In Review', done: 'Done' };
    return labels[status] || status;
  }

  // ---- Sprints ----
  function getSprints(project) {
    if (!project) project = getActiveProject();
    return project ? (project.sprints || []) : [];
  }

  function getActiveSprint(project) {
    if (!project) project = getActiveProject();
    if (!project) return null;
    return (project.sprints || []).find(s => s.status === 'active') || null;
  }

  function getSprintById(sprintId, project) {
    if (!project) project = getActiveProject();
    if (!project) return null;
    return (project.sprints || []).find(s => s.id === sprintId) || null;
  }

  function createSprint(data, project) {
    if (!project) project = getActiveProject();
    if (!project) return null;
    const sprint = {
      id: _generateId(),
      name: data.name || 'New Sprint',
      goal: data.goal || '',
      startDate: data.startDate || '',
      endDate: data.endDate || '',
      status: 'planned',
      stories: [],
      velocity: 0,
      capacity: data.capacity || 40,
      burndownData: [],
      retrospective: null
    };
    project.sprints = project.sprints || [];
    project.sprints.push(sprint);
    saveProject(project);
    addActivity(`Created sprint "${sprint.name}"`);
    return sprint;
  }

  function updateSprint(sprintId, updates, project) {
    if (!project) project = getActiveProject();
    if (!project) return null;
    const sprint = (project.sprints || []).find(s => s.id === sprintId);
    if (sprint) {
      Object.assign(sprint, updates);
      saveProject(project);
    }
    return sprint;
  }

  function startSprint(sprintId, project) {
    if (!project) project = getActiveProject();
    if (!project) return null;
    // Check no other active sprint
    const active = getActiveSprint(project);
    if (active) return null;
    const sprint = updateSprint(sprintId, { status: 'active' }, project);
    if (sprint) {
      // Initialize burndown
      _initBurndown(sprint, project);
      addActivity(`Started sprint "${sprint.name}"`);
    }
    return sprint;
  }

  function completeSprint(sprintId, retrospective, project) {
    if (!project) project = getActiveProject();
    if (!project) return null;
    const sprint = (project.sprints || []).find(s => s.id === sprintId);
    if (!sprint) return null;

    // Calculate velocity
    const donePoints = (sprint.stories || [])
      .filter(s => s.status === 'done')
      .reduce((sum, s) => sum + (s.storyPoints || 0), 0);
    sprint.velocity = donePoints;
    sprint.status = 'completed';
    sprint.retrospective = retrospective || null;

    // Move undone stories back to backlog
    const undone = (sprint.stories || []).filter(s => s.status !== 'done');
    undone.forEach(s => {
      s.sprintId = null;
      s.activityLog = s.activityLog || [];
      s.activityLog.push({ text: 'Moved back to backlog (sprint completed)', timestamp: _now() });
      project.backlog.push(s);
    });
    sprint.stories = (sprint.stories || []).filter(s => s.status === 'done');

    saveProject(project);
    addActivity(`Completed sprint "${sprint.name}" with ${donePoints} points`);
    return sprint;
  }

  function _initBurndown(sprint, project) {
    const totalPoints = (sprint.stories || [])
      .reduce((sum, s) => sum + (s.storyPoints || 0), 0);
    const start = new Date(sprint.startDate);
    const end = new Date(sprint.endDate);
    const days = Math.max(1, Math.ceil((end - start) / (1000 * 60 * 60 * 24)));
    sprint.burndownData = [{ day: 0, remaining: totalPoints, date: sprint.startDate }];
    saveProject(project);
  }

  function updateBurndown(sprintId, project) {
    if (!project) project = getActiveProject();
    if (!project) return;
    const sprint = (project.sprints || []).find(s => s.id === sprintId);
    if (!sprint || sprint.status !== 'active') return;

    const totalPoints = (sprint.stories || [])
      .reduce((sum, s) => sum + (s.storyPoints || 0), 0);
    const donePoints = (sprint.stories || [])
      .filter(s => s.status === 'done')
      .reduce((sum, s) => sum + (s.storyPoints || 0), 0);
    const remaining = totalPoints - donePoints;

    const start = new Date(sprint.startDate);
    const now = new Date();
    const dayNum = Math.max(0, Math.ceil((now - start) / (1000 * 60 * 60 * 24)));

    sprint.burndownData = sprint.burndownData || [];
    // Update today's entry or add new
    const existing = sprint.burndownData.find(d => d.day === dayNum);
    if (existing) {
      existing.remaining = remaining;
    } else {
      sprint.burndownData.push({ day: dayNum, remaining, date: now.toISOString().split('T')[0] });
    }
    saveProject(project);
  }

  // ---- Team Members ----
  function getTeamMembers(project) {
    if (!project) project = getActiveProject();
    return project ? (project.teamMembers || []) : [];
  }

  function getTeamMember(memberId, project) {
    if (!memberId) return null;
    if (!project) project = getActiveProject();
    if (!project) return null;
    return (project.teamMembers || []).find(m => m.id === memberId) || null;
  }

  function createTeamMember(data, project) {
    if (!project) project = getActiveProject();
    if (!project) return null;
    const member = {
      id: _generateId(),
      name: data.name || 'New Member',
      role: data.role || 'Developer',
      avatar: _getInitials(data.name || 'NM'),
      capacity: data.capacity || 10,
      color: data.color || '#6366f1'
    };
    project.teamMembers = project.teamMembers || [];
    project.teamMembers.push(member);
    saveProject(project);
    addActivity(`Added team member "${member.name}"`);
    return member;
  }

  function updateTeamMember(memberId, updates, project) {
    if (!project) project = getActiveProject();
    if (!project) return null;
    const member = (project.teamMembers || []).find(m => m.id === memberId);
    if (member) {
      Object.assign(member, updates);
      if (updates.name) member.avatar = _getInitials(updates.name);
      saveProject(project);
    }
    return member;
  }

  function deleteTeamMember(memberId, project) {
    if (!project) project = getActiveProject();
    if (!project) return;
    const member = (project.teamMembers || []).find(m => m.id === memberId);
    project.teamMembers = (project.teamMembers || []).filter(m => m.id !== memberId);
    saveProject(project);
    if (member) addActivity(`Removed team member "${member.name}"`);
  }

  function _getInitials(name) {
    return name.split(' ').map(w => w[0]).join('').toUpperCase().substring(0, 2);
  }

  // ---- Epics ----
  function getEpics(project) {
    if (!project) project = getActiveProject();
    return project ? (project.epics || []) : [];
  }

  function getEpicById(epicId, project) {
    if (!epicId) return null;
    if (!project) project = getActiveProject();
    if (!project) return null;
    return (project.epics || []).find(e => e.id === epicId) || null;
  }

  function createEpic(data, project) {
    if (!project) project = getActiveProject();
    if (!project) return null;
    const epic = {
      id: _generateId(),
      name: data.name || 'New Epic',
      description: data.description || '',
      color: data.color || '#6366f1'
    };
    project.epics = project.epics || [];
    project.epics.push(epic);
    saveProject(project);
    addActivity(`Created epic "${epic.name}"`);
    return epic;
  }

  function updateEpic(epicId, updates, project) {
    if (!project) project = getActiveProject();
    if (!project) return null;
    const epic = (project.epics || []).find(e => e.id === epicId);
    if (epic) {
      Object.assign(epic, updates);
      saveProject(project);
    }
    return epic;
  }

  function deleteEpic(epicId, project) {
    if (!project) project = getActiveProject();
    if (!project) return;
    project.epics = (project.epics || []).filter(e => e.id !== epicId);
    // Remove epic from stories
    getAllStories(project).forEach(s => {
      if (s.epicId === epicId) s.epicId = null;
    });
    saveProject(project);
  }

  function getEpicStories(epicId, project) {
    return getAllStories(project).filter(s => s.epicId === epicId);
  }

  // ---- Activity Log ----
  function getActivityLog() {
    return _get(KEYS.ACTIVITY_LOG) || [];
  }

  function addActivity(text) {
    const log = getActivityLog();
    log.unshift({ text, timestamp: _now() });
    // Keep last 50
    if (log.length > 50) log.length = 50;
    _set(KEYS.ACTIVITY_LOG, log);
  }

  function clearActivityLog() {
    _set(KEYS.ACTIVITY_LOG, []);
  }

  // ---- DoD ----
  function getDoDChecklist() {
    return _get(KEYS.DOD_CHECKLIST) || [
      'Code reviewed',
      'Unit tests pass',
      'Acceptance criteria met',
      'Documentation updated',
      'No known bugs'
    ];
  }

  function setDoDChecklist(items) {
    _set(KEYS.DOD_CHECKLIST, items);
  }

  // ---- WIP Limits ----
  function getWipLimits() {
    return _get(KEYS.WIP_LIMITS) || { todo: 0, inprogress: 5, review: 3, done: 0 };
  }

  function setWipLimits(limits) {
    _set(KEYS.WIP_LIMITS, limits);
  }

  // ---- Theme ----
  function getTheme() {
    return _get(KEYS.THEME) || 'dark';
  }

  function setTheme(theme) {
    _set(KEYS.THEME, theme);
  }

  // ---- Sidebar ----
  function getSidebarState() {
    return _get(KEYS.SIDEBAR) || 'expanded';
  }

  function setSidebarState(state) {
    _set(KEYS.SIDEBAR, state);
  }

  // ---- Undo ----
  function pushUndo(action) {
    const stack = _get(KEYS.UNDO_STACK) || [];
    stack.push(action);
    if (stack.length > 20) stack.shift();
    _set(KEYS.UNDO_STACK, stack);
  }

  function popUndo() {
    const stack = _get(KEYS.UNDO_STACK) || [];
    const action = stack.pop();
    _set(KEYS.UNDO_STACK, stack);
    return action || null;
  }

  // ---- Public API ----
  return {
    getProjects, setProjects, getActiveProjectId, setActiveProjectId,
    getActiveProject, createProject, updateProject, saveProject, deleteProject,
    getAllStories, getStoryById, createStory, updateStory, deleteStory,
    moveStoryToStatus, moveStoryToSprint,
    getSprints, getActiveSprint, getSprintById,
    createSprint, updateSprint, startSprint, completeSprint,
    updateBurndown,
    getTeamMembers, getTeamMember, createTeamMember, updateTeamMember, deleteTeamMember,
    getEpics, getEpicById, createEpic, updateEpic, deleteEpic, getEpicStories,
    getActivityLog, addActivity, clearActivityLog,
    getDoDChecklist, setDoDChecklist,
    getWipLimits, setWipLimits,
    getTheme, setTheme,
    getSidebarState, setSidebarState,
    pushUndo, popUndo,
    generateId: _generateId
  };
})();
