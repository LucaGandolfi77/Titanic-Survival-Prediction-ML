class CommentaryEngine {
  constructor() {
    this.synthesis = null;
    this.enabled = true;
    this.language = 'it-IT';
    this.lastComment = 0;
    this.cooldown = 3000;
    this.queue = [];
    this.speaking = false;
  }

  init() {
    this.synthesis = window.speechSynthesis;
    if (!this.synthesis) {
      this.enabled = false;
      console.warn('Speech Synthesis API not supported');
      return;
    }

    this.language = this._detectLanguage();
  }

  _detectLanguage() {
    const lang = navigator.language || 'en-US';
    const supported = this._getSupportedVoices().map(v => v.lang);
    if (supported.includes(lang)) return lang;
    if (lang.startsWith('it') && supported.includes('it-IT')) return 'it-IT';
    const fallback = supported.find(l => l.startsWith(lang.substring(0, 2)));
    return fallback || 'en-US';
  }

  _getSupportedVoices() {
    if (!this.synthesis) return [];
    return this.synthesis.getVoices();
  }

  async comment(text, priority = 'normal') {
    if (!this.enabled || !this.synthesis) return;

    const now = Date.now();
    if (now - this.lastComment < this.cooldown && priority !== 'high') return;

    this.queue.push({ text, priority, timestamp: now });
    if (!this.speaking) {
      this._processQueue();
    }
  }

  _processQueue() {
    if (this.queue.length === 0) {
      this.speaking = false;
      return;
    }

    this.queue.sort((a, b) => {
      const p = { high: 3, normal: 2, low: 1 };
      return p[b.priority] - p[a.priority];
    });

    const item = this.queue.shift();
    this.lastComment = item.timestamp;
    this.speaking = true;

    const utterance = new SpeechSynthesisUtterance(item.text);
    utterance.lang = this.language;
    utterance.rate = 0.95;
    utterance.pitch = 1.0;
    utterance.volume = 0.8;

    const voices = this._getSupportedVoices();
    const itVoice = voices.find(v => v.lang.startsWith('it'));
    if (itVoice) utterance.voice = itVoice;

    utterance.onend = () => {
      setTimeout(() => this._processQueue(), 200);
    };

    utterance.onerror = () => {
      this.speaking = false;
      this._processQueue();
    };

    this.synthesis.cancel();
    this.synthesis.speak(utterance);
  }

  commentGoal(team = 'home') {
    const messages = team === 'home'
      ? ['Gol! Che golamazza!', 'Rete fantastica!', 'Che gioia!']
      : ['Gol subito!', 'Ingoalato!', 'Che sventola!'];
    this.comment(messages[Math.floor(Math.random() * messages.length)], 'high');
  }

  commentHalfTime() {
    this.comment('Fine primo tempo. Sforza di più!', 'normal');
  }

  commentFullTime(result) {
    const messages = {
      win: ['Partita vinta! Ben giocato!', 'Vittoria! Grande squadra!'],
      lose: ['Sconfitta. Rimbocchiamoci le maniche!', 'Di sicuro si può fare di meglio.'],
      draw: ['Pareggio. Ai punti!', 'Partita equilibrata.']
    };
    const key = result === 'win' ? 'win' : result === 'lose' ? 'lose' : 'draw';
    const msg = messages[key][Math.floor(Math.random() * messages[key].length)];
    this.comment(msg, 'high');
  }

  commentKickoff() {
    this.comment('Via! Inizia la partita!', 'normal');
  }

  commentTackle() {
    const messages = ['Rinvia!', 'Che contrasto!', 'Bravo!', 'Mai dire mai!'];
    this.comment(messages[Math.floor(Math.random() * messages.length)], 'low');
  }

  commentPass() {
    if (Math.random() > 0.7) {
      this.comment('Bel passaggio!', 'low');
    }
  }

  mute() {
    this.enabled = false;
    if (this.synthesis) this.synthesis.cancel();
  }

  unmute() {
    this.enabled = true;
  }

  setLanguage(lang) {
    this.language = lang;
  }
}

export { CommentaryEngine };
