class CoachAI {
  constructor() {
    this.analysis = null;
  }

  analyzeMatch(matchData) {
    const { players, ball, scores, duration, currentTime } = matchData;
    const myTeam = players.filter(p => p.team === 0);
    const opponentTeam = players.filter(p => p.team === 1);

    const analysis = {
      timestamp: new Date().toISOString(),
      scores: { home: scores[0], away: scores[1] },
      duration: duration,
      playedTime: currentTime,
      stats: this._computeStats(myTeam, opponentTeam, ball),
      positions: this._analyzePositions(myTeam, opponentTeam, ball),
      suggestions: [],
      rating: 0
    };

    analysis.suggestions = this._generateSuggestions(analysis);
    analysis.rating = this._computeRating(analysis);

    this.analysis = analysis;
    return analysis;
  }

  _computeStats(myTeam, opponentTeam, ball) {
    const stats = {
      totalPasses: 0,
      totalTackles: 0,
      possessionTime: 0,
      attacks: 0,
      saves: 0,
      goalsConceded: 0,
      avgSpeed: 0,
      maxSpeed: 0
    };

    let speedSum = 0;
    myTeam.forEach(p => {
      const speed = p.vel.mag();
      speedSum += speed;
      stats.maxSpeed = Math.max(stats.maxSpeed, speed);
    });
    stats.avgSpeed = myTeam.length > 0 ? speedSum / myTeam.length : 0;

    return stats;
  }

  _analyzePositions(myTeam, opponentTeam, ball) {
    const analysis = {
      formationCompactness: 0,
      defensiveLine: null,
      attackingPress: null,
      spacing: 0,
      centerOccupation: 0
    };

    if (myTeam.length === 0) return analysis;

    let avgX = 0, avgY = 0;
    myTeam.forEach(p => {
      avgX += p.pos.x;
      avgY += p.pos.y;
    });
    avgX /= myTeam.length;
    avgY /= myTeam.length;

    let spread = 0;
    myTeam.forEach(p => {
      spread += Math.sqrt((p.pos.x - avgX) ** 2 + (p.pos.y - avgY) ** 2);
    });
    analysis.spacing = spread / myTeam.length;

    const centerThreshold = 400;
    const inCenter = myTeam.filter(p => Math.abs(p.pos.x - centerThreshold) < 200).length;
    analysis.centerOccupation = inCenter / myTeam.length;

    let minX = Infinity;
    myTeam.forEach(p => { minX = Math.min(minX, p.pos.x); });
    analysis.defensiveLine = minX;

    return analysis;
  }

  _generateSuggestions(analysis) {
    const suggestions = [];
    const stats = analysis.stats;
    const positions = analysis.positions;

    if (positions.spacing > 250) {
      suggestions.push({
        type: 'formation',
        priority: 'high',
        icon: '📏',
        title: 'Formazione troppo distesa',
        message: 'I tuoi giocatori sono troppo distanti. Prova a mantenere una formazione più compatta per ridurre gli spazi per l\'avversario.'
      });
    }

    if (positions.centerOccupation < 0.3) {
      suggestions.push({
        type: 'positioning',
        priority: 'medium',
        icon: '🎯',
        title: 'Centro campo poco occupato',
        message: 'Meno del 30% dei tuoi giocatori è nel centro campo. Controlla il centro per avere più possesso palla.'
      });
    }

    if (stats.avgSpeed < 50) {
      suggestions.push({
        type: 'tempo',
        priority: 'medium',
        icon: '⏱️',
        title: 'Gioco troppo lento',
        message: 'La velocità media è bassa. Cerca di muoverti di più senza palla per creare spazio.'
      });
    }

    if (analysis.scores.home < analysis.scores.away) {
      suggestions.push({
        type: 'strategy',
        priority: 'high',
        icon: '⚔️',
        title: 'Sotto nel punteggio',
        message: 'Concentra più giocatori in attacco. Usa il PASS per creare opportunità di tiro.'
      });
    }

    if (analysis.scores.home > analysis.scores.away) {
      suggestions.push({
        type: 'strategy',
        priority: 'medium',
        icon: '🛡️',
        title: 'In vantaggio',
        message: 'Ben giocato! Ora controlla il possesso palla e difendi il vantaggio con una linea difensiva compatta.'
      });
    }

    if (positions.defensiveLine > 200) {
      suggestions.push({
        type: 'defense',
        priority: 'high',
        icon: '🚨',
        title: 'Linea difensiva alta',
        message: 'I tuoi difensori sono troppo avanzati. Risch di subire contropiede. Abbassa la linea difensiva.'
      });
    }

    suggestions.push({
      type: 'general',
      priority: 'low',
      icon: '💡',
      title: 'Consiglio generale',
      message: 'Usa il SWITCH per passare al giocatore più vicino alla palla. Il PASS è più efficace del tiro da distanza.'
    });

    return suggestions;
  }

  _computeRating(analysis) {
    let rating = 5;
    const { scores } = analysis;

    if (scores.home > scores.away) rating += 2;
    else if (scores.home < scores.away) rating -= 1;

    if (analysis.stats.avgSpeed > 100) rating += 1;
    if (analysis.positions.spacing < 200) rating += 1;

    return Math.max(1, Math.min(10, Math.round(rating)));
  }

  getCommentarySnippets(language = 'it') {
    const snippets = {
      it: {
        great_defense: 'Partita difensiva impeccabile!',
        excellent_attack: 'Attacco devastante!',
        keep_pressure: 'Mantieni la pressione!',
        tactical_discipline: 'Disciplina tattica eccellente!',
        need_work: 'C\'è margine di miglioramento',
        turning_point: 'Momento decisivo!',
        momentum_shift: 'Cambio di momentum!'
      },
      en: {
        great_defense: 'Impeccable defensive play!',
        excellent_attack: 'Devastating attack!',
        keep_pressure: 'Keep the pressure!',
        tactical_discipline: 'Excellent tactical discipline!',
        need_work: 'Room for improvement',
        turning_point: 'Turning point!',
        momentum_shift: 'Momentum shift!'
      }
    };

    return snippets[language] || snippets.it;
  }
}

export { CoachAI };
