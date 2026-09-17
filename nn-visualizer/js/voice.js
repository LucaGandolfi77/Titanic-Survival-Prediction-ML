const SITZ = 800;

const COMMANDS_EN = {
  'reset':        { aliases: ['restart', 'reboot', 'ripristina'], desc: 'Reset' },
  'zoom in':      { aliases: ['zoom up', 'enlarge', 'magnify', 'bigger', 'aumenta'], desc: 'Zoom In' },
  'zoom out':     { aliases: ['zoom down', 'shrink', 'reduce', 'smaller', 'riduci'], desc: 'Zoom Out' },
  'fit':          { aliases: ['fit screen', 'fit to screen', 'adatta', 'centra'], desc: 'Fit' },
  'play':         { aliases: ['start', 'train', 'training', 'go', 'via', 'inizia', 'suona'], desc: 'Play' },
  'stop':         { aliases: ['pause', 'halt', 'ferma', 'pausa'], desc: 'Stop' },
  'cosmic':       { aliases: ['universe', 'stars', 'galaxy', 'cosmico', 'stellare', 'universo'], desc: 'Cosmic' },
  'rainbow':      { aliases: ['colors', 'colori', 'arcobaleno', 'arc-en-ciel'], desc: 'Rainbow' },
  'story':        { aliases: ['narrate', 'explain', 'racconta', 'spiega', 'storia'], desc: 'Story' },
  'music game':   { aliases: ['neuron music', 'music neurons', 'gioco musicale'], desc: 'Music Game' },
  'next model':   { aliases: ['switch model', 'other model', 'next', 'prossimo', 'altro'], desc: 'Next Model' },
  'undo':         { aliases: ['back', 'reverse', 'annulla'], desc: 'Undo' },
  'redo':         { aliases: ['forward', 'again', 'ripeti', 'ripristina'], desc: 'Redo' },
  'music on':     { aliases: ['sound on', 'audio on', 'musica', 'suono'], desc: 'Music On' },
  'music off':    { aliases: ['sound off', 'audio off', 'silenzio', 'muto'], desc: 'Music Off' },
  'zoom in more': { aliases: ['more zoom', 'further zoom', 'ingrandisci ancora'], desc: 'Zoom In More' },
  'zoom out more':{ aliases: ['less zoom', 'further zoom out', 'riduci ancora'], desc: 'Zoom Out More' },
};

const COMMANDS_IT = {
  'reset':        { aliases: ['resetta', 'ripristina', 'ricomincia'], desc: 'Reset' },
  'zoom in':      { aliases: ['ingrandisci', 'zoom avanti', 'ingrandisci zoom', 'allarga'], desc: 'Zoom In' },
  'zoom out':     { aliases: ['riduci', 'zoom indietro', 'rimpicciolisci', 'diminuisci'], desc: 'Zoom Out' },
  'fit':          { aliases: ['adatta', 'centra', 'adatta allo schermo'], desc: 'Fit' },
  'play':         { aliases: ['suona', 'inizia', 'vai', 'comincia', 'avvia'], desc: 'Play' },
  'stop':         { aliases: ['ferma', 'smetti', 'arresta'], desc: 'Stop' },
  'cosmic':       { aliases: ['cosmico', 'universo', 'stelle', 'stellare', 'galassia'], desc: 'Cosmic' },
  'rainbow':      { aliases: ['colori', 'arcobaleno', 'colorato'], desc: 'Rainbow' },
  'story':        { aliases: ['narrate', 'racconta', 'storia', 'spiega'], desc: 'Story' },
  'music game':   { aliases: ['gioco musicale', 'musica gioco'], desc: 'Music Game' },
  'next model':   { aliases: ['prossimo modello', 'altro modello', 'cambia modello', 'prossimo'], desc: 'Next Model' },
  'undo':         { aliases: ['annulla', 'undo', 'indietro', 'torna'], desc: 'Undo' },
  'redo':         { aliases: ['ripeti', 'redo', 'avanti', 'ripeti azione'], desc: 'Redo' },
  'music on':     { aliases: ['musica', 'suono', 'audio', 'metti musica', 'attiva audio'], desc: 'Music On' },
  'music off':    { aliases: ['silenzio', 'muto', 'togli musica', 'stop audio', 'mute'], desc: 'Music Off' },
};

export function createVoiceEngine(onCommand, onTranscription) {
  let recognition = null;
  let listening = false;
  let isSpeaking = false;
  let lastAction = null;
  let contextTimeout = null;
  let language = 'en-US';
  let lastConfidence = 0;
  let transcriptBuffer = '';

  function getCommands() {
    const all = { ...COMMANDS_EN };
    if (language.includes('it') || language.includes('IT')) {
      for (const [key, val] of Object.entries(COMMANDS_IT)) {
        if (!all[key]) all[key] = val;
        else all[key].aliases = [...new Set([...all[key].aliases, ...val.aliases])];
      }
    }
    return all;
  }

  function matchCommand(transcript) {
    const commands = getCommands();
    const words = transcript.toLowerCase().trim().split(/\s+/);
    let bestMatch = null;
    let bestScore = 0;

    for (const [action, cmd] of Object.entries(commands)) {
      const allTerms = [action, ...cmd.aliases];
      let score = 0;
      for (const term of allTerms) {
        const termWords = term.split(' ');
        const matchCount = termWords.filter(tw =>
          words.some(w => w.includes(tw) || tw.includes(w))
        ).length;
        if (matchCount === termWords.length) {
          score += termWords.length;
        }
      }
      if (score > bestScore) {
        bestScore = score;
        bestMatch = action;
      }
    }

    if (lastAction && (lastAction.includes('zoom') || lastAction.includes('zoom'))) {
      if (transcript.includes('more') || transcript.includes('ancora') || transcript.includes('ancora')) {
        if (lastAction.includes('in')) return 'zoom in more';
        if (lastAction.includes('out')) return 'zoom out more';
      }
    }

    return bestMatch && bestScore >= 1 ? bestMatch : null;
  }

  function speak(text, lang = language) {
    if (!('speechSynthesis' in window)) return;
    try {
      speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.lang = lang.includes('it') ? 'it-IT' : 'en-US';
      utterance.rate = 1.1;
      utterance.pitch = 1;
      isSpeaking = true;
      utterance.onend = () => { isSpeaking = false; };
      utterance.onerror = () => { isSpeaking = false; };
      speechSynthesis.speak(utterance);
    } catch (e) { /* silent */ }
  }

  function setContext(action) {
    lastAction = action;
    clearTimeout(contextTimeout);
    contextTimeout = setTimeout(() => { lastAction = null; }, 5000);
  }

  function onResult(event) {
    const last = event.results.length - 1;
    const result = event.results[last];
    const transcript = result[0].transcript.toLowerCase().trim();
    const confidence = result[0].confidence;
    lastConfidence = confidence;

    if (onTranscription) onTranscription(transcript, confidence);
    transcriptBuffer = transcript;

    if (result.isFinal) {
      const action = matchCommand(transcript);
      if (action) {
        setContext(action);
        onCommand(action, transcript, confidence);
      } else {
        onCommand('unknown', transcript, confidence);
      }
    }
  }

  function onError(event) {
    console.warn('Speech error:', event.error);
    if (event.error === 'not-allowed') {
      listening = false;
    }
    if (event.error === 'no-speech') {
      // retry after short delay
      setTimeout(() => {
        if (listening && recognition) {
          try { recognition.start(); } catch (e) { /* ignore */ }
        }
      }, 500);
    }
  }

  function onEnd() {
    if (listening && recognition) {
      try { recognition.start(); } catch (e) { /* ignore */ }
    }
  }

  function start() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) return false;
    if (listening) return true;

    recognition = new SpeechRecognition();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = language;

    recognition.onresult = onResult;
    recognition.onerror = onError;
    recognition.onend = onEnd;

    try {
      recognition.start();
      listening = true;
      return true;
    } catch (e) {
      console.error('Voice start failed:', e);
      return false;
    }
  }

  function stop() {
    if (recognition && listening) {
      listening = false;
      try { recognition.stop(); } catch (e) { /* ok */ }
      if ('speechSynthesis' in window) {
        try { speechSynthesis.cancel(); } catch (e) { /* ok */ }
      }
      isSpeaking = false;
    }
  }

  function toggle() {
    if (listening) {
      stop();
      return false;
    }
    return start();
  }

  function setLanguage(lang) {
    language = lang;
    if (recognition) recognition.lang = lang;
  }

  function getLanguage() {
    return language;
  }

  function isListening() {
    return listening;
  }

  function getIsSpeaking() {
    return isSpeaking;
  }

  function getLastConfidence() {
    return lastConfidence;
  }

  function getTranscript() {
    return transcriptBuffer;
  }

  function speakAction(action) {
    const descriptions = {
      'reset': 'Reset', 'zoom in': 'Zooming in', 'zoom out': 'Zooming out',
      'fit': 'Fitting', 'play': 'Playing', 'stop': 'Stopping',
      'cosmic': 'Cosmic mode', 'rainbow': 'Rainbow mode', 'story': 'Story mode',
      'music game': 'Music neurons game', 'next model': 'Next model',
      'undo': 'Undoing', 'redo': 'Redoing',
      'music on': 'Music on', 'music off': 'Music off',
      'zoom in more': 'More zoom', 'zoom out more': 'Less zoom',
    };
    speak(descriptions[action] || 'Done');
  }

  return {
    start, stop, toggle, isListening, getIsSpeaking,
    setLanguage, getLanguage,
    getLastConfidence, getTranscript,
    speakAction, speak,
  };
}
