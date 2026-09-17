document.addEventListener('DOMContentLoaded', () => {
    loadAnalytics();
    loadRecommendations();
    loadAmbientTheme();
    loadLanguage();
});

function loadLanguage() {
    const switcher = document.getElementById('lang-switcher');
    if (!switcher) return;
    fetch('/api/check-session')
        .then(r => r.json())
        .then(data => {
            if (data.logged_in && data.lang) {
                switcher.value = data.lang;
            }
        })
        .catch(() => {});
}

function changeLang(lang) {
    fetch('/api/set-language', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({language: lang}),
    }).then(() => {
        document.documentElement.lang = lang;
    });
}

async function loadAmbientTheme() {
    try {
        const pos = await getLocation();
        const params = pos ? `?lat=${pos.lat}&lon=${pos.lon}` : '';
        const res = await fetch(`/api/ambient/theme${params}`);
        const data = await res.json();
        if (data.theme) {
            document.body.style.backgroundColor = data.theme.bg;
            document.documentElement.style.setProperty('--accent-pink', data.theme.accent);
        }
    } catch (e) {
        // Use time-based theme as fallback
        const hour = new Date().getHours();
        let theme = 'classic';
        if (hour >= 6 && hour < 12) theme = 'morning';
        else if (hour >= 18 && hour < 22) theme = 'sunset';

        const themes = {
            morning: {bg: '#2e1a1a', accent: '#fbbf24'},
            sunset: {bg: '#3a1a1a', accent: '#f97316'},
            classic: {bg: '#1a1a2e', accent: '#e94560'},
        };
        const t = themes[theme] || themes.classic;
        document.body.style.backgroundColor = t.bg;
        document.documentElement.style.setProperty('--accent-pink', t.accent);
    }
}

function getLocation() {
    return new Promise((resolve) => {
        if (!navigator.geolocation) {
            resolve(null);
            return;
        }
        navigator.geolocation.getCurrentPosition(
            (pos) => resolve({lat: pos.coords.latitude, lon: pos.coords.longitude}),
            () => resolve(null),
            {timeout: 3000}
        );
    });
}

async function loadAnalytics() {
    try {
        const data = await fetchApi('/api/analytics/summary');
        const totalEl = document.getElementById('stat-total');
        if (totalEl) totalEl.textContent = data.total_votes;
        const usersEl = document.getElementById('stat-users');
        if (usersEl) usersEl.textContent = data.users_voted;
        const avgEl = document.getElementById('stat-avg');
        if (avgEl) avgEl.textContent = data.avg_score;
        const posEl = document.getElementById('stat-positive');
        if (posEl) posEl.textContent = data.positive_pct + '%';

        renderScoreDistribution(data.score_distribution);
        renderHourlyActivity(data.hourly_votes);
    } catch (e) {
        console.warn('Failed to load analytics:', e);
    }

    try {
        const sentiment = await fetchApi('/api/analytics/sentiment');
        renderSentiment(sentiment);
    } catch (e) {
        console.warn('Failed to load sentiment:', e);
    }

    try {
        const summary = await fetchApi('/api/analytics/summary');
        renderTopModels(summary.top_models);
    } catch (e) {
        console.warn('Failed to load top models:', e);
    }
}

async function loadRecommendations() {
    try {
        const data = await fetchApi('/api/recommendations');
        const container = document.getElementById('recommendations-list');
        if (!container) return;
        container.innerHTML = '';

        data.models.forEach((model, idx) => {
            const item = document.createElement('div');
            item.className = 'reco-item';
            item.innerHTML = `
                <span class="reco-rank">#${idx + 1}</span>
                <span class="reco-name">${model.username}</span>
                <span class="reco-score">${model.score}</span>
            `;
            container.appendChild(item);
        });
    } catch (e) {
        console.warn('Failed to load recommendations:', e);
    }
}

function renderScoreDistribution(data) {
    const ctx = document.getElementById('chart-distribution');
    if (!ctx || typeof Chart === 'undefined') return;
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: data.map(d => d.score + '/10'),
            datasets: [{
                label: 'Votes',
                data: data.map(d => d.count),
                backgroundColor: data.map(d => {
                    if (d.score >= 8) return '#ffdd59';
                    if (d.score <= 3) return '#e94560';
                    return '#0f3460';
                }),
                borderRadius: 4,
            }],
        },
        options: { responsive: true, plugins: { legend: { display: false } } },
    });
}

function renderHourlyActivity(data) {
    const ctx = document.getElementById('chart-hourly');
    if (!ctx || typeof Chart === 'undefined') return;
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.map(d => d.hour + ':00'),
            datasets: [{
                label: 'Votes',
                data: data.map(d => d.count),
                borderColor: '#e94560',
                backgroundColor: 'rgba(233,69,96,0.2)',
                fill: true,
                tension: 0.4,
            }],
        },
        options: { responsive: true },
    });
}

function renderSentiment(data) {
    const ctx = document.getElementById('chart-sentiment');
    if (!ctx || typeof Chart === 'undefined') return;
    new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Positive', 'Neutral', 'Negative'],
            datasets: [{
                data: [data.positive, data.neutral, data.negative],
                backgroundColor: ['#ffdd59', '#0f3460', '#e94560'],
                borderWidth: 0,
            }],
        },
        options: { responsive: true },
    });
}

function renderTopModels(models) {
    const ctx = document.getElementById('chart-top');
    if (!ctx || typeof Chart === 'undefined') return;
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: models.map(m => m.username),
            datasets: [{
                label: 'Avg Score',
                data: models.map(m => m.avg_score),
                backgroundColor: '#e94560',
                borderRadius: 4,
            }],
        },
        options: {
            responsive: true,
            indexAxis: 'y',
            plugins: { legend: { display: false } },
        },
    });
}

async function fetchApi(path, options = {}) {
    const res = await fetch(path, options);
    if (!res.ok) throw new Error(`API ${res.status}`);
    return res.json();
}
