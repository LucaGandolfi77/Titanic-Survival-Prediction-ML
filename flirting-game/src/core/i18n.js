// Translation layer: UI chrome + mood lines in 4 languages (en, it, es, fr).
// Story content stays in the story-pack language; Intl handles locale-aware
// formatting. Auto-detects from navigator.language, persisted in prefs.

const DICTS = {
  en: {
    'setup.badge': 'Interactive Story',
    'setup.lead': 'Pick your vibe, answer fast, and unlock extra flirty scenes before the moment slips away.',
    'setup.character': 'Your character',
    'setup.interest': 'You want to date',
    'setup.goal': 'Goal',
    'setup.goalValue': 'Reach chapter 3',
    'setup.bonus': 'Bonus',
    'setup.bonusValue': 'Fast choices unlock secrets',
    'setup.best': 'Personal best',
    'setup.aiMode': 'AI mode',
    'setup.aiModeHint': 'Infinite procedural scenes · custom lines · smart scoring',
    'setup.aiModel': '🧠 Download neural model (~60MB)',
    'setup.aiModelReady': '🧠 Neural sentiment ready',
    'setup.aiModelDownloading': 'Downloading AI model…',
    'setup.reminders': '🔔 Enable reminders',
    'setup.stats': '🔒 Your stats vault',
    'setup.start': 'Start the story',
    'setup.duet': '👯 Duet mode',
    'game.chapter': 'Chapter {n}',
    'game.reaction': 'Reaction boost',
    'game.charm': 'Charm',
    'game.streak': 'Fast streak',
    'game.secrets': 'Secret scenes',
    'game.restart': 'Restart',
    'game.custom': '✍️ Write your own line',
    'game.customPlaceholder': 'Write your own flirty line…',
    'game.send': 'Send',
    'game.historyTitle': 'Your choices so far',
    'game.close': 'Close',
    'game.hint': 'Swipe left · choice history — Tap · skip reply',
    'end.playAgain': 'Play again',
    'end.share': 'Share your score',
    'end.trailer': '🎬 Story trailer',
    'end.finalCharm': 'Final charm',
    'end.bestStreak': 'Best streak',
    'end.unlocked': 'Unlocked scenes',
    'stats.title': 'Your stats vault 🔒',
    'stats.games': 'Games played',
    'stats.bestScore': 'Best score',
    'stats.bestStreak': 'Best streak',
    'stats.endings': 'Endings collected',
    'stats.difficulty': 'Difficulty',
    'stats.avgReaction': 'Avg reaction',
    'stats.ambient': 'Ambient context',
    'stats.device': 'AI backend',
    'stats.achievements': 'Achievements',
    'stats.export': '📤 Export story',
    'stats.import': '📥 Import story / pack',
    'mood.late-night': 'Late night',
    'mood.morning': 'Morning',
    'mood.afternoon': 'Afternoon',
    'mood.evening': 'Evening',
    'mood.night': 'Night',
    'mood.late-night.flavor': 'the city is asleep, and the hour feels dangerous',
    'mood.morning.flavor': 'the light is soft, and the day is full of possibility',
    'mood.afternoon.flavor': 'the sun is high, and everything feels electric',
    'mood.evening.flavor': 'the light is turning gold, and the night is almost here',
    'mood.night.flavor': 'the city lights are on, and this is the right hour for trouble',
    'toast.version': 'New version available',
    'toast.reload': 'Reload',
    'toast.listening': 'Listening… speak a choice',
    'toast.voiceOn': 'Voice mode on 🔊',
    'toast.voiceOff': 'Voice mode off'
  },
  it: {
    'setup.badge': 'Storia Interattiva',
    'setup.lead': 'Scegli il tuo mood, rispondi veloce e sblocca scene flirty extra prima che il momento sfugga.',
    'setup.character': 'Il tuo personaggio',
    'setup.interest': 'Vuoi uscire con',
    'setup.goal': 'Obiettivo',
    'setup.goalValue': 'Arriva al capitolo 3',
    'setup.bonus': 'Bonus',
    'setup.bonusValue': 'Le risposte veloci sbloccano segreti',
    'setup.best': 'Record personale',
    'setup.aiMode': 'Modalità AI',
    'setup.aiModeHint': 'Scene procedurali infinite · righe personalizzate · punteggi smart',
    'setup.aiModel': '🧠 Scarica il modello neurale (~60MB)',
    'setup.aiModelReady': '🧠 Sentiment neurale pronto',
    'setup.aiModelDownloading': 'Scaricamento modello AI…',
    'setup.reminders': '🔔 Attiva i promemoria',
    'setup.stats': '🔒 La tua cassaforte stats',
    'setup.start': 'Inizia la storia',
    'setup.duet': 'Modalità duetto 👯',
    'game.chapter': 'Capitolo {n}',
    'game.reaction': 'Boost di reazione',
    'game.charm': 'Charm',
    'game.streak': 'Serie veloce',
    'game.secrets': 'Scene segrete',
    'game.restart': 'Ricomincia',
    'game.custom': '✍️ Scrivi la tua riga',
    'game.customPlaceholder': 'Scrivi la tua riga flirty…',
    'game.send': 'Invia',
    'game.historyTitle': 'Le tue scelte finora',
    'game.close': 'Chiudi',
    'game.hint': 'Swipe a sinistra · cronologia — Tap · salta risposta',
    'end.playAgain': 'Rigioca',
    'end.share': 'Condividi il punteggio',
    'end.trailer': '🎬 Trailer della storia',
    'end.finalCharm': 'Charm finale',
    'end.bestStreak': 'Serie migliore',
    'end.unlocked': 'Scene sbloccate',
    'stats.title': 'La tua cassaforte stats 🔒',
    'stats.games': 'Partite giocate',
    'stats.bestScore': 'Record',
    'stats.bestStreak': 'Serie migliore',
    'stats.endings': 'Finali sbloccati',
    'stats.difficulty': 'Difficoltà',
    'stats.avgReaction': 'Reazione media',
    'stats.ambient': 'Contesto ambientale',
    'stats.device': 'Backend AI',
    'stats.achievements': 'Obiettivi',
    'stats.export': '📤 Esporta storia',
    'stats.import': '📥 Importa storia / pack',
    'mood.late-night': 'Notte fonda',
    'mood.morning': 'Mattina',
    'mood.afternoon': 'Pomeriggio',
    'mood.evening': 'Sera',
    'mood.night': 'Notte',
    'mood.late-night.flavor': 'la città dorme, e l\u2019ora sembra fatta apposta',
    'mood.morning.flavor': 'la luce è morbida, e la giornata è piena di possibilità',
    'mood.afternoon.flavor': 'il sole è alto, e tutto sembra elettrico',
    'mood.evening.flavor': 'la luce sta diventando oro, e la notte è quasi qui',
    'mood.night.flavor': 'le luci della città sono accese, ed è l\u2019ora giusta per guai',
    'toast.version': 'Nuova versione disponibile',
    'toast.reload': 'Ricarica',
    'toast.listening': 'In ascolto… parla, scegli una risposta',
    'toast.voiceOn': 'Modalità vocale attiva 🔊',
    'toast.voiceOff': 'Modalità vocale off'
  },
  es: {
    'setup.badge': 'Historia Interactiva',
    'setup.lead': 'Elige tu onda, responde rápido y desbloquea escenas coquetas extra antes de que el momento se escape.',
    'setup.character': 'Tu personaje',
    'setup.interest': 'Quieres salir con',
    'setup.goal': 'Objetivo',
    'setup.goalValue': 'Llega al capítulo 3',
    'setup.bonus': 'Bonus',
    'setup.bonusValue': 'Las respuestas rápidas desbloquean secretos',
    'setup.best': 'Récord personal',
    'setup.aiMode': 'Modo IA',
    'setup.aiModeHint': 'Escenas procedurales infinitas · líneas propias · puntuación inteligente',
    'setup.aiModel': '🧠 Descargar modelo neuronal (~60MB)',
    'setup.aiModelReady': '🧠 Sentimiento neuronal listo',
    'setup.aiModelDownloading': 'Descargando modelo de IA…',
    'setup.reminders': '🔔 Activar recordatorios',
    'setup.stats': '🔒 Tu bóveda de stats',
    'setup.start': 'Empezar la historia',
    'setup.duet': 'Modo dúo 👯',
    'game.chapter': 'Capítulo {n}',
    'game.reaction': 'Impulso de reacción',
    'game.charm': 'Encanto',
    'game.streak': 'Racha rápida',
    'game.secrets': 'Escenas secretas',
    'game.restart': 'Reiniciar',
    'game.custom': '✍️ Escribe tu propia línea',
    'game.customPlaceholder': 'Escribe tu línea coqueta…',
    'game.send': 'Enviar',
    'game.historyTitle': 'Tus decisiones hasta ahora',
    'game.close': 'Cerrar',
    'game.hint': 'Desliza a la izquierda · historial — Toca · saltar respuesta',
    'end.playAgain': 'Jugar otra vez',
    'end.share': 'Compartir puntuación',
    'end.trailer': '🎬 Tráiler de la historia',
    'end.finalCharm': 'Encanto final',
    'end.bestStreak': 'Mejor racha',
    'end.unlocked': 'Escenas desbloqueadas',
    'stats.title': 'Tu bóveda de stats 🔒',
    'stats.games': 'Partidas jugadas',
    'stats.bestScore': 'Récord',
    'stats.bestStreak': 'Mejor racha',
    'stats.endings': 'Finales desbloqueados',
    'stats.difficulty': 'Dificultad',
    'stats.avgReaction': 'Reacción media',
    'stats.ambient': 'Contexto ambiental',
    'stats.device': 'Backend de IA',
    'stats.achievements': 'Logros',
    'stats.export': '📤 Exportar historia',
    'stats.import': '📥 Importar historia / pack',
    'mood.late-night': 'Madrugada',
    'mood.morning': 'Mañana',
    'mood.afternoon': 'Tarde',
    'mood.evening': 'Atardecer',
    'mood.night': 'Noche',
    'mood.late-night.flavor': 'la ciudad duerme, y la hora se siente peligrosa',
    'mood.morning.flavor': 'la luz es suave, y el día está lleno de posibilidades',
    'mood.afternoon.flavor': 'el sol está alto, y todo se siente eléctrico',
    'mood.evening.flavor': 'la luz se vuelve dorada, y la noche casi está aquí',
    'mood.night.flavor': 'las luces de la ciudad están encendidas, y es la hora perfecta para problemas',
    'toast.version': 'Nueva versión disponible',
    'toast.reload': 'Recargar',
    'toast.listening': 'Escuchando… habla, elige una respuesta',
    'toast.voiceOn': 'Modo de voz activado 🔊',
    'toast.voiceOff': 'Modo de voz desactivado'
  },
  fr: {
    'setup.badge': 'Histoire Interactive',
    'setup.lead': 'Choisis ton ambiance, réponds vite et débloque des scènes flirty bonus avant que le moment ne s\u2019envole.',
    'setup.character': 'Ton personnage',
    'setup.interest': 'Tu veux sortir avec',
    'setup.goal': 'Objectif',
    'setup.goalValue': 'Arrive au chapitre 3',
    'setup.bonus': 'Bonus',
    'setup.bonusValue': 'Les réponses rapides débloquent des secrets',
    'setup.best': 'Record personnel',
    'setup.aiMode': 'Mode IA',
    'setup.aiModeHint': 'Scènes procédurales infinies · répliques perso · score intelligent',
    'setup.aiModel': '🧠 Télécharger le modèle neural (~60MB)',
    'setup.aiModelReady': '🧠 Sentiment neural prêt',
    'setup.aiModelDownloading': 'Téléchargement du modèle IA…',
    'setup.reminders': '🔔 Activer les rappels',
    'setup.stats': '🔒 Ton coffre de stats',
    'setup.start': 'Commencer l\u2019histoire',
    'setup.duet': 'Mode duo 👯',
    'game.chapter': 'Chapitre {n}',
    'game.reaction': 'Boost de réaction',
    'game.charm': 'Charme',
    'game.streak': 'Série rapide',
    'game.secrets': 'Scènes secrètes',
    'game.restart': 'Recommencer',
    'game.custom': '✍️ Écris ta propre réplique',
    'game.customPlaceholder': 'Écris ta réplique flirty…',
    'game.send': 'Envoyer',
    'game.historyTitle': 'Tes choix jusqu\u2019ici',
    'game.close': 'Fermer',
    'game.hint': 'Swipe à gauche · historique — Tap · passer la réponse',
    'end.playAgain': 'Rejouer',
    'end.share': 'Partager le score',
    'end.trailer': '🎬 Bande-annonce',
    'end.finalCharm': 'Charme final',
    'end.bestStreak': 'Meilleure série',
    'end.unlocked': 'Scènes débloquées',
    'stats.title': 'Ton coffre de stats 🔒',
    'stats.games': 'Parties jouées',
    'stats.bestScore': 'Record',
    'stats.bestStreak': 'Meilleure série',
    'stats.endings': 'Finales débloquées',
    'stats.difficulty': 'Difficulté',
    'stats.avgReaction': 'Réaction moyenne',
    'stats.ambient': 'Contexte ambiant',
    'stats.device': 'Backend IA',
    'stats.achievements': 'Succès',
    'stats.export': '📤 Exporter l\u2019histoire',
    'stats.import': '📥 Importer histoire / pack',
    'mood.late-night': 'Nuit profonde',
    'mood.morning': 'Matin',
    'mood.afternoon': 'Après-midi',
    'mood.evening': 'Soir',
    'mood.night': 'Nuit',
    'mood.late-night.flavor': 'la ville dort, et l\u2019heure semble dangereuse',
    'mood.morning.flavor': 'la lumière est douce, et la journée est pleine de possibilités',
    'mood.afternoon.flavor': 'le soleil est haut, et tout semble électrique',
    'mood.evening.flavor': 'la lumière devient or, et la nuit est presque là',
    'mood.night.flavor': 'les lumières de la ville sont allumées, et c\u2019est la bonne heure pour les ennuis',
    'toast.version': 'Nouvelle version disponible',
    'toast.reload': 'Recharger',
    'toast.listening': 'Écoute… parle, choisis une réplique',
    'toast.voiceOn': 'Mode vocal activé 🔊',
    'toast.voiceOff': 'Mode vocal désactivé'
  }
};

const LANGS = Object.keys(DICTS);
let current = null;

export function detectLanguage() {
  const nav = typeof navigator !== 'undefined' ? navigator.language || '' : '';
  return LANGS.find((l) => nav.toLowerCase().startsWith(l)) || 'en';
}

export function setLanguage(lang) {
  current = DICTS[lang] ? lang : detectLanguage();
  return current;
}

export function getLanguage() {
  if (!current) current = detectLanguage();
  return current;
}

/** Translate a key with {param} substitution, falling back to English. */
export function t(key, params = {}) {
  const dict = DICTS[getLanguage()] || DICTS.en;
  let text = dict[key] ?? DICTS.en[key] ?? key;
  for (const [name, value] of Object.entries(params)) {
    text = text.replaceAll(`{${name}}`, String(value));
  }
  return text;
}

export function availableLanguages() {
  return LANGS;
}

/**
 * Apply translations to the static UI chrome: [data-i18n] → textContent,
 * [data-i18n-placeholder] → placeholder attribute.
 */
export function applyTranslations() {
  const root = typeof document !== 'undefined' ? document : null;
  if (!root) return;
  root.querySelectorAll('[data-i18n]').forEach((el) => {
    el.textContent = t(el.dataset.i18n);
  });
  root.querySelectorAll('[data-i18n-placeholder]').forEach((el) => {
    el.setAttribute('placeholder', t(el.dataset.i18nPlaceholder));
  });
  root.querySelectorAll('[data-i18n-title]').forEach((el) => {
    el.setAttribute('title', t(el.dataset.i18nTitle));
  });
}
