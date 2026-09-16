/* ===== i18n — externalized UI strings =====
 *
 * Infrastructure for translations. `t(key, params)` looks up the active
 * language with fallback to English; `applyI18n()` localizes static HTML
 * via data-i18n attributes. Language is persisted in GameState.language.
 *
 * Scope: the UI chrome (menus, tabs, panel titles, HUD, overview labels).
 * Game-event messages are English; migrate them here incrementally as needed.
 */

export const STRINGS = {
  en: {
    'menu.title': 'SKATE MANAGER',
    'menu.subtitle': 'Build the ultimate synchronized skating team',
    'menu.newGame': '▶ NEW GAME',
    'menu.continue': '📂 CONTINUE',
    'menu.settings': '⚙️ SETTINGS',

    'setup.heading': 'Create Your Team',
    'setup.teamName': 'Team Name:',
    'setup.teamColor': 'Team Color:',
    'setup.difficulty': 'Starting Difficulty:',
    'setup.difficulty.amateur': 'Amateur',
    'setup.difficulty.semi-pro': 'Semi-Pro',
    'setup.difficulty.elite': 'Elite',
    'setup.start': '⛸️ START SEASON',
    'setup.back': '← BACK',

    'settings.heading': 'Settings',
    'settings.volume': 'Master Volume:',
    'settings.sfx': 'SFX:',
    'settings.autosave': 'Auto-Save:',
    'settings.language': 'Language:',
    'settings.export': '💾 EXPORT SAVE',
    'settings.import': '📥 IMPORT SAVE',
    'settings.back': '← BACK',

    'tab.overview': '🏠 Overview',
    'tab.squad': '👥 Squad',
    'tab.market': '🛒 Market',
    'tab.calendar': '📅 Calendar',
    'tab.sponsors': '💼 Sponsors',
    'tab.standings': '🏆 Standings',
    'tab.stats': '📊 Stats',

    'panel.activeSquad': 'Active Squad ({n}/{max})',
    'panel.reserveBench': 'Reserve Bench ({n}/{max})',
    'panel.cohesionLabel': 'COHESION:',
    'panel.coachingStaff': 'Coaching Staff',
    'panel.seasonCalendar': 'Season Calendar',
    'panel.availableSkaters': 'Available Skaters',
    'panel.listedSkaters': 'Your Listed Skaters',
    'panel.activeDeals': 'Active Deals',
    'panel.availableSponsors': 'Available Sponsors',
    'panel.leaderboard': 'Season Leaderboard',
    'panel.seasonHistory': 'Season History',
    'panel.clubRecords': 'Club Records',
    'panel.scoreProgression': 'Score Progression (this season)',
    'panel.mostFielded': 'Most-Fielded Skaters',
    'panel.hallOfFame': '🏛️ Hall of Fame',

    'ov.avgOverall': 'Avg Overall',
    'ov.avgMorale': 'Avg Morale',
    'ov.cohesion': 'Cohesion',
    'ov.netWeekly': 'Net Weekly',
    'ov.thisWeek': 'This Week:',
    'ov.trainingWeek': 'Training Week',
    'ov.noCompScheduled': 'No competition scheduled',
    'ov.advanceWeek': 'ADVANCE WEEK ▶',
    'ov.competeNow': 'COMPETE NOW 🏆',
    'ov.quickSim': '⚡ QUICK SIM',

    'mg.tempo': '🎵 TEMPO',
    'mg.formations': '💃 FORMATIONS',
    'mg.alerts': '⚠️ ALERTS',
    'mg.judges': '👩‍⚖️ JUDGES',
    'mg.teamMorale': 'TEAM MORALE:',
    'mg.keyHint': '⌨ 1–4 tempo · Q–P formations · ESC pause'
  },

  it: {
    'menu.title': 'SKATE MANAGER',
    'menu.subtitle': 'Costruisci la squadra di pattinaggio sincronizzato definitiva',
    'menu.newGame': '▶ NUOVA PARTITA',
    'menu.continue': '📂 CONTINUA',
    'menu.settings': '⚙️ IMPOSTAZIONI',

    'setup.heading': 'Crea la tua squadra',
    'setup.teamName': 'Nome squadra:',
    'setup.teamColor': 'Colore squadra:',
    'setup.difficulty': 'Difficoltà iniziale:',
    'setup.difficulty.amateur': 'Dilettante',
    'setup.difficulty.semi-pro': 'Semipro',
    'setup.difficulty.elite': 'Elite',
    'setup.start': '⛸️ INIZIA STAGIONE',
    'setup.back': '← INDIETRO',

    'settings.heading': 'Impostazioni',
    'settings.volume': 'Volume principale:',
    'settings.sfx': 'Effetti sonori:',
    'settings.autosave': 'Salvataggio automatico:',
    'settings.language': 'Lingua:',
    'settings.export': '💾 ESPORTA SALVATAGGIO',
    'settings.import': '📥 IMPORTA SALVATAGGIO',
    'settings.back': '← INDIETRO',

    'tab.overview': '🏠 Panoramica',
    'tab.squad': '👥 Squadra',
    'tab.market': '🛒 Mercato',
    'tab.calendar': '📅 Calendario',
    'tab.sponsors': '💼 Sponsor',
    'tab.standings': '🏆 Classifica',
    'tab.stats': '📊 Statistiche',

    'panel.activeSquad': 'Squadra titolare ({n}/{max})',
    'panel.reserveBench': 'Panchina ({n}/{max})',
    'panel.cohesionLabel': 'COESIONE:',
    'panel.coachingStaff': 'Staff tecnico',
    'panel.seasonCalendar': 'Calendario della stagione',
    'panel.availableSkaters': 'Pattinatrici disponibili',
    'panel.listedSkaters': 'Le tue pattinatrici in vendita',
    'panel.activeDeals': 'Contratti attivi',
    'panel.availableSponsors': 'Sponsor disponibili',
    'panel.leaderboard': 'Classifica della stagione',
    'panel.seasonHistory': 'Storico stagioni',
    'panel.clubRecords': 'Record del club',
    'panel.scoreProgression': 'Progressione punteggi (stagione corrente)',
    'panel.mostFielded': 'Pattinatrici più impiegate',
    'panel.hallOfFame': '🏛️ Hall of Fame',

    'ov.avgOverall': 'Media overall',
    'ov.avgMorale': 'Media morale',
    'ov.cohesion': 'Coesione',
    'ov.netWeekly': 'Netto settimanale',
    'ov.thisWeek': 'Questa settimana:',
    'ov.trainingWeek': 'Settimana di allenamento',
    'ov.noCompScheduled': 'Nessuna competizione programmata',
    'ov.advanceWeek': 'AVANZA SETTIMANA ▶',
    'ov.competeNow': 'COMPETI ORA 🏆',
    'ov.quickSim': '⚡ SIM VELOCE',

    'mg.tempo': '🎵 TEMPO',
    'mg.formations': '💃 FORMAZIONI',
    'mg.alerts': '⚠️ ALLERTE',
    'mg.judges': '👩‍⚖️ GIUDICI',
    'mg.teamMorale': 'MORALE SQUADRA:',
    'mg.keyHint': '⌨ 1–4 tempo · Q–P formazioni · ESC pausa'
  }
};

let activeLang = 'en';

export function getLang() {
  return activeLang;
}

export function setLang(lang) {
  if (STRINGS[lang]) activeLang = lang;
  return activeLang;
}

/**
 * Look up a UI string in the active language (fallback: English, then the key).
 * @param {string} key
 * @param {Object<string, string|number>} [params]  {name} placeholders
 */
export function t(key, params = {}) {
  let s = STRINGS[activeLang][key] ?? STRINGS.en[key] ?? key;
  for (const [k, v] of Object.entries(params)) {
    s = s.replaceAll(`{${k}}`, String(v));
  }
  return s;
}

/** Localize all static HTML elements carrying a data-i18n attribute. */
export function applyI18n() {
  document.querySelectorAll('[data-i18n]').forEach(el => {
    el.textContent = t(el.dataset.i18n);
  });
}
