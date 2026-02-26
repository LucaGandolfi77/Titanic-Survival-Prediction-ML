/* =====================================================
   SAMPLEDATA.JS — Demo project data
   ===================================================== */

const SampleData = (() => {
  function load() {
    const now = new Date();
    const sprintStart = new Date(now);
    sprintStart.setDate(sprintStart.getDate() - 5);
    const sprintEnd = new Date(now);
    sprintEnd.setDate(sprintEnd.getDate() + 9);

    const sprint2Start = new Date(sprintEnd);
    sprint2Start.setDate(sprint2Start.getDate() + 1);
    const sprint2End = new Date(sprint2Start);
    sprint2End.setDate(sprint2End.getDate() + 13);

    const fmt = d => d.toISOString().split('T')[0];

    const members = [
      { id: 'mem1', name: 'Alice Chen', role: 'Developer', avatar: 'AC', capacity: 13, color: '#6366f1' },
      { id: 'mem2', name: 'Bob Smith', role: 'Developer', avatar: 'BS', capacity: 10, color: '#22c55e' },
      { id: 'mem3', name: 'Carol Davis', role: 'Designer', avatar: 'CD', capacity: 8, color: '#ec4899' },
      { id: 'mem4', name: 'Dan Wilson', role: 'QA', avatar: 'DW', capacity: 8, color: '#f59e0b' }
    ];

    const epics = [
      { id: 'epic1', name: 'User Authentication', description: 'Login, signup, password recovery flows', color: '#6366f1' },
      { id: 'epic2', name: 'Dashboard', description: 'Main dashboard and analytics views', color: '#22c55e' },
      { id: 'epic3', name: 'API Integration', description: 'REST API endpoints and data services', color: '#f59e0b' },
      { id: 'epic4', name: 'Mobile Responsive', description: 'Make all views mobile-friendly', color: '#ec4899' }
    ];

    const ts = () => new Date(now.getTime() - Math.random() * 86400000 * 5).toISOString();

    // Sprint 1 stories (active)
    const sprint1Stories = [
      {
        id: 's1', title: 'Implement login form', description: 'Create login form with email/password fields and validation',
        acceptanceCriteria: '- Form validates email format\n- Shows error on wrong credentials\n- Redirects on success',
        priority: 'critical', storyPoints: 5, status: 'done', sprintId: 'sp1', assigneeId: 'mem1', epicId: 'epic1',
        tags: ['frontend', 'auth'], subtasks: [
          { id: 'st1', text: 'Create form HTML', completed: true },
          { id: 'st2', text: 'Add validation', completed: true },
          { id: 'st3', text: 'Connect to API', completed: true }
        ],
        createdAt: ts(), updatedAt: ts(), completedAt: ts(),
        activityLog: [{ text: 'Story created', timestamp: ts() }, { text: 'Moved to Done by Alice Chen', timestamp: ts() }]
      },
      {
        id: 's2', title: 'Design signup flow wireframes', description: 'Create wireframes for the signup process',
        acceptanceCriteria: '- Include all form fields\n- Show validation states\n- Mobile layout included',
        priority: 'high', storyPoints: 3, status: 'done', sprintId: 'sp1', assigneeId: 'mem3', epicId: 'epic1',
        tags: ['design'], subtasks: [
          { id: 'st4', text: 'Desktop wireframe', completed: true },
          { id: 'st5', text: 'Mobile wireframe', completed: true }
        ],
        createdAt: ts(), updatedAt: ts(), completedAt: ts(),
        activityLog: [{ text: 'Story created', timestamp: ts() }]
      },
      {
        id: 's3', title: 'Build signup API endpoint', description: 'POST /api/signup with validation',
        acceptanceCriteria: '- Validates email uniqueness\n- Hashes password\n- Returns JWT',
        priority: 'critical', storyPoints: 8, status: 'inprogress', sprintId: 'sp1', assigneeId: 'mem1', epicId: 'epic3',
        tags: ['backend', 'api'], subtasks: [
          { id: 'st6', text: 'Create route handler', completed: true },
          { id: 'st7', text: 'Add validation middleware', completed: true },
          { id: 'st8', text: 'Write tests', completed: false },
          { id: 'st9', text: 'Add rate limiting', completed: false }
        ],
        createdAt: ts(), updatedAt: ts(), completedAt: null,
        activityLog: [{ text: 'Story created', timestamp: ts() }, { text: 'Moved to In Progress by Alice Chen', timestamp: ts() }]
      },
      {
        id: 's4', title: 'Dashboard layout component', description: 'Create the main dashboard grid layout',
        acceptanceCriteria: '- Responsive grid\n- Widget placeholders\n- Sidebar integration',
        priority: 'high', storyPoints: 5, status: 'inprogress', sprintId: 'sp1', assigneeId: 'mem2', epicId: 'epic2',
        tags: ['frontend', 'layout'], subtasks: [
          { id: 'st10', text: 'Grid system', completed: true },
          { id: 'st11', text: 'Widget containers', completed: false },
          { id: 'st12', text: 'Responsive breakpoints', completed: false }
        ],
        createdAt: ts(), updatedAt: ts(), completedAt: null,
        activityLog: [{ text: 'Story created', timestamp: ts() }]
      },
      {
        id: 's5', title: 'Password reset flow', description: 'Implement forgot password and reset functionality',
        acceptanceCriteria: '- Send reset email\n- Token validation\n- New password form',
        priority: 'medium', storyPoints: 5, status: 'review', sprintId: 'sp1', assigneeId: 'mem2', epicId: 'epic1',
        tags: ['frontend', 'backend'], subtasks: [
          { id: 'st13', text: 'Reset email template', completed: true },
          { id: 'st14', text: 'Token generation', completed: true },
          { id: 'st15', text: 'Reset form UI', completed: true },
          { id: 'st16', text: 'QA testing', completed: false }
        ],
        createdAt: ts(), updatedAt: ts(), completedAt: null,
        activityLog: [{ text: 'Story created', timestamp: ts() }]
      },
      {
        id: 's6', title: 'QA: Test login scenarios', description: 'Create and execute test cases for login',
        acceptanceCriteria: '- Test all edge cases\n- Report bugs\n- Verify fixes',
        priority: 'high', storyPoints: 3, status: 'todo', sprintId: 'sp1', assigneeId: 'mem4', epicId: 'epic1',
        tags: ['qa', 'testing'], subtasks: [],
        createdAt: ts(), updatedAt: ts(), completedAt: null,
        activityLog: [{ text: 'Story created', timestamp: ts() }]
      },
      {
        id: 's7', title: 'Set up CI/CD pipeline', description: 'Configure GitHub Actions for automated deploys',
        acceptanceCriteria: '- Lint on PR\n- Tests on push\n- Deploy to staging',
        priority: 'medium', storyPoints: 5, status: 'todo', sprintId: 'sp1', assigneeId: 'mem1', epicId: null,
        tags: ['devops'], subtasks: [],
        createdAt: ts(), updatedAt: ts(), completedAt: null,
        activityLog: [{ text: 'Story created', timestamp: ts() }]
      },
      {
        id: 's8', title: 'Mobile nav menu', description: 'Hamburger menu for mobile screens',
        acceptanceCriteria: '- Slide-in animation\n- Close on outside click\n- Accessible',
        priority: 'low', storyPoints: 3, status: 'todo', sprintId: 'sp1', assigneeId: 'mem3', epicId: 'epic4',
        tags: ['frontend', 'mobile'], subtasks: [],
        createdAt: ts(), updatedAt: ts(), completedAt: null,
        activityLog: [{ text: 'Story created', timestamp: ts() }]
      }
    ];

    // Backlog stories
    const backlogStories = [
      {
        id: 's9', title: 'OAuth integration (Google)', description: 'Add Google OAuth 2.0 login option',
        acceptanceCriteria: '- Google sign-in button\n- Token exchange\n- User profile sync',
        priority: 'medium', storyPoints: 8, status: 'todo', sprintId: null, assigneeId: null, epicId: 'epic1',
        tags: ['auth', 'oauth'], subtasks: [], createdAt: ts(), updatedAt: ts(), completedAt: null,
        activityLog: [{ text: 'Story created', timestamp: ts() }]
      },
      {
        id: 's10', title: 'Analytics charts widget', description: 'Build chart widgets for the dashboard',
        acceptanceCriteria: '- Line chart\n- Bar chart\n- Date range selector',
        priority: 'high', storyPoints: 8, status: 'todo', sprintId: null, assigneeId: null, epicId: 'epic2',
        tags: ['frontend', 'charts'], subtasks: [], createdAt: ts(), updatedAt: ts(), completedAt: null,
        activityLog: [{ text: 'Story created', timestamp: ts() }]
      },
      {
        id: 's11', title: 'User profile page', description: 'Account settings and profile customization',
        acceptanceCriteria: '- Edit name/email\n- Change password\n- Upload avatar',
        priority: 'low', storyPoints: 5, status: 'todo', sprintId: null, assigneeId: null, epicId: null,
        tags: ['frontend'], subtasks: [], createdAt: ts(), updatedAt: ts(), completedAt: null,
        activityLog: [{ text: 'Story created', timestamp: ts() }]
      },
      {
        id: 's12', title: 'Email notification service', description: 'Send transactional emails for key actions',
        acceptanceCriteria: '- Welcome email\n- Password reset\n- Weekly digest',
        priority: 'medium', storyPoints: 5, status: 'todo', sprintId: null, assigneeId: null, epicId: 'epic3',
        tags: ['backend', 'email'], subtasks: [], createdAt: ts(), updatedAt: ts(), completedAt: null,
        activityLog: [{ text: 'Story created', timestamp: ts() }]
      },
      {
        id: 's13', title: 'Data export (CSV)', description: 'Allow users to export their data as CSV',
        acceptanceCriteria: '- Select date range\n- Choose data types\n- Download file',
        priority: 'low', storyPoints: 3, status: 'todo', sprintId: null, assigneeId: null, epicId: 'epic3',
        tags: ['backend', 'feature'], subtasks: [], createdAt: ts(), updatedAt: ts(), completedAt: null,
        activityLog: [{ text: 'Story created', timestamp: ts() }]
      },
      {
        id: 's14', title: 'Dark mode toggle', description: 'Implement dark/light mode switching',
        acceptanceCriteria: '- Persist preference\n- Smooth transition\n- All components styled',
        priority: 'low', storyPoints: 3, status: 'todo', sprintId: null, assigneeId: null, epicId: 'epic4',
        tags: ['frontend', 'ui'], subtasks: [], createdAt: ts(), updatedAt: ts(), completedAt: null,
        activityLog: [{ text: 'Story created', timestamp: ts() }]
      },
      {
        id: 's15', title: 'API rate limiting', description: 'Implement rate limiting on all API endpoints',
        acceptanceCriteria: '- 100 req/min per user\n- Return 429 status\n- Redis-based counter',
        priority: 'high', storyPoints: 5, status: 'todo', sprintId: null, assigneeId: null, epicId: 'epic3',
        tags: ['backend', 'security'], subtasks: [], createdAt: ts(), updatedAt: ts(), completedAt: null,
        activityLog: [{ text: 'Story created', timestamp: ts() }]
      },
      {
        id: 's16', title: 'Responsive tables', description: 'Make all data tables work on mobile',
        acceptanceCriteria: '- Horizontal scroll\n- Sticky first column\n- Touch-friendly',
        priority: 'medium', storyPoints: 3, status: 'todo', sprintId: null, assigneeId: null, epicId: 'epic4',
        tags: ['frontend', 'mobile'], subtasks: [], createdAt: ts(), updatedAt: ts(), completedAt: null,
        activityLog: [{ text: 'Story created', timestamp: ts() }]
      },
      {
        id: 's17', title: 'Search functionality', description: 'Global search across all entities',
        acceptanceCriteria: '- Fuzzy matching\n- Results grouped by type\n- Keyboard navigation',
        priority: 'high', storyPoints: 8, status: 'todo', sprintId: null, assigneeId: null, epicId: 'epic2',
        tags: ['frontend', 'backend'], subtasks: [], createdAt: ts(), updatedAt: ts(), completedAt: null,
        activityLog: [{ text: 'Story created', timestamp: ts() }]
      },
      {
        id: 's18', title: 'Error boundary component', description: 'Graceful error handling in UI',
        acceptanceCriteria: '- Catch render errors\n- Show fallback UI\n- Log to service',
        priority: 'medium', storyPoints: 3, status: 'todo', sprintId: null, assigneeId: null, epicId: null,
        tags: ['frontend'], subtasks: [], createdAt: ts(), updatedAt: ts(), completedAt: null,
        activityLog: [{ text: 'Story created', timestamp: ts() }]
      },
      {
        id: 's19', title: 'Onboarding tutorial', description: 'Interactive walkthrough for new users',
        acceptanceCriteria: '- Highlight key features\n- Skip option\n- Track completion',
        priority: 'low', storyPoints: 5, status: 'todo', sprintId: null, assigneeId: null, epicId: 'epic2',
        tags: ['frontend', 'ux'], subtasks: [], createdAt: ts(), updatedAt: ts(), completedAt: null,
        activityLog: [{ text: 'Story created', timestamp: ts() }]
      },
      {
        id: 's20', title: 'WebSocket notifications', description: 'Real-time push notifications',
        acceptanceCriteria: '- Socket.io setup\n- Event system\n- Toast notifications',
        priority: 'medium', storyPoints: 8, status: 'todo', sprintId: null, assigneeId: null, epicId: 'epic3',
        tags: ['backend', 'realtime'], subtasks: [], createdAt: ts(), updatedAt: ts(), completedAt: null,
        activityLog: [{ text: 'Story created', timestamp: ts() }]
      }
    ];

    // Burndown data for sprint 1
    const totalSp1 = sprint1Stories.reduce((s, st) => s + st.storyPoints, 0);
    const burndown = [
      { day: 0, remaining: totalSp1, date: fmt(sprintStart) },
      { day: 1, remaining: totalSp1, date: fmt(new Date(sprintStart.getTime() + 86400000)) },
      { day: 2, remaining: totalSp1 - 3, date: fmt(new Date(sprintStart.getTime() + 86400000 * 2)) },
      { day: 3, remaining: totalSp1 - 5, date: fmt(new Date(sprintStart.getTime() + 86400000 * 3)) },
      { day: 4, remaining: totalSp1 - 8, date: fmt(new Date(sprintStart.getTime() + 86400000 * 4)) },
      { day: 5, remaining: totalSp1 - 8, date: fmt(new Date(sprintStart.getTime() + 86400000 * 5)) }
    ];

    // Completed sprint (historical)
    const oldSprintStart = new Date(sprintStart);
    oldSprintStart.setDate(oldSprintStart.getDate() - 15);
    const oldSprintEnd = new Date(sprintStart);
    oldSprintEnd.setDate(oldSprintEnd.getDate() - 1);

    const completedSprintStories = [
      {
        id: 's_old1', title: 'Project setup', description: 'Initialize repo and toolchain',
        acceptanceCriteria: '', priority: 'critical', storyPoints: 3, status: 'done',
        sprintId: 'sp0', assigneeId: 'mem1', epicId: null, tags: ['devops'],
        subtasks: [], createdAt: ts(), updatedAt: ts(), completedAt: ts(),
        activityLog: [{ text: 'Story created', timestamp: ts() }]
      },
      {
        id: 's_old2', title: 'Design system tokens', description: 'Define colors, fonts, spacing',
        acceptanceCriteria: '', priority: 'high', storyPoints: 5, status: 'done',
        sprintId: 'sp0', assigneeId: 'mem3', epicId: null, tags: ['design'],
        subtasks: [], createdAt: ts(), updatedAt: ts(), completedAt: ts(),
        activityLog: [{ text: 'Story created', timestamp: ts() }]
      },
      {
        id: 's_old3', title: 'Database schema design', description: 'Define tables and relations',
        acceptanceCriteria: '', priority: 'critical', storyPoints: 8, status: 'done',
        sprintId: 'sp0', assigneeId: 'mem2', epicId: 'epic3', tags: ['backend'],
        subtasks: [], createdAt: ts(), updatedAt: ts(), completedAt: ts(),
        activityLog: [{ text: 'Story created', timestamp: ts() }]
      },
      {
        id: 's_old4', title: 'Component library scaffold', description: 'Button, Input, Card components',
        acceptanceCriteria: '', priority: 'high', storyPoints: 5, status: 'done',
        sprintId: 'sp0', assigneeId: 'mem1', epicId: null, tags: ['frontend'],
        subtasks: [], createdAt: ts(), updatedAt: ts(), completedAt: ts(),
        activityLog: [{ text: 'Story created', timestamp: ts() }]
      },
      {
        id: 's_old5', title: 'API boilerplate', description: 'Express setup with middleware',
        acceptanceCriteria: '', priority: 'medium', storyPoints: 3, status: 'done',
        sprintId: 'sp0', assigneeId: 'mem2', epicId: 'epic3', tags: ['backend'],
        subtasks: [], createdAt: ts(), updatedAt: ts(), completedAt: ts(),
        activityLog: [{ text: 'Story created', timestamp: ts() }]
      }
    ];

    const project = {
      id: 'proj_demo',
      name: 'WebApp MVP',
      description: 'Full-stack web application minimum viable product with auth, dashboard, and API',
      createdAt: new Date(now.getTime() - 86400000 * 30).toISOString(),
      teamMembers: members,
      epics: epics,
      sprints: [
        {
          id: 'sp0',
          name: 'Sprint 0 — Setup',
          goal: 'Get the project scaffolding and design foundation in place',
          startDate: fmt(oldSprintStart),
          endDate: fmt(oldSprintEnd),
          status: 'completed',
          stories: completedSprintStories,
          velocity: 24,
          capacity: 30,
          burndownData: [],
          retrospective: {
            well: 'Good team coordination. Tooling setup was smooth.',
            improve: 'Need better story breakdown. Some tasks were too big.',
            actions: ['Create story splitting guidelines', 'Set up code review process']
          }
        },
        {
          id: 'sp1',
          name: 'Sprint 1 — Auth & Dashboard',
          goal: 'Complete core authentication and start dashboard implementation',
          startDate: fmt(sprintStart),
          endDate: fmt(sprintEnd),
          status: 'active',
          stories: sprint1Stories,
          velocity: 0,
          capacity: 40,
          burndownData: burndown,
          retrospective: null
        },
        {
          id: 'sp2',
          name: 'Sprint 2 — API & Polish',
          goal: 'Build remaining API endpoints and refine UX',
          startDate: fmt(sprint2Start),
          endDate: fmt(sprint2End),
          status: 'planned',
          stories: [],
          velocity: 0,
          capacity: 40,
          burndownData: [],
          retrospective: null
        }
      ],
      backlog: backlogStories
    };

    // Save
    Store.setProjects([project]);
    Store.setActiveProjectId(project.id);

    // Activity log
    const activities = [
      { text: 'Loaded demo data', timestamp: now.toISOString() },
      { text: 'Completed sprint "Sprint 0 — Setup"', timestamp: new Date(now.getTime() - 86400000 * 6).toISOString() },
      { text: 'Started sprint "Sprint 1 — Auth & Dashboard"', timestamp: new Date(now.getTime() - 86400000 * 5).toISOString() },
      { text: 'Alice Chen moved "Implement login form" to Done', timestamp: new Date(now.getTime() - 86400000 * 3).toISOString() },
      { text: 'Carol Davis moved "Design signup flow wireframes" to Done', timestamp: new Date(now.getTime() - 86400000 * 2).toISOString() },
      { text: 'Alice Chen started "Build signup API endpoint"', timestamp: new Date(now.getTime() - 86400000).toISOString() },
      { text: 'Bob Smith started "Dashboard layout component"', timestamp: new Date(now.getTime() - 86400000).toISOString() },
      { text: 'Bob Smith moved "Password reset flow" to In Review', timestamp: new Date(now.getTime() - 3600000 * 5).toISOString() }
    ];
    localStorage.setItem('scrumflow_activity_log', JSON.stringify(activities));

    return project;
  }

  return { load };
})();
