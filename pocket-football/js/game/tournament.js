const TOURNAMENT_FORMATS = {
  ROUND_ROBIN: 'round_robin',
  SINGLE_ELIMINATION: 'single_elimination',
  DOUBLE_ELIMINATION: 'double_elimination'
};

const TOURNAMENT_STATUS = {
  SETUP: 'setup',
  IN_PROGRESS: 'in_progress',
  FINISHED: 'finished'
};

class TournamentManager {
  constructor(storageService) {
    this.storage = storageService;
  }

  async init() {
    await this.storage.ready();
  }

  async createTournament(config) {
    const tournament = {
      id: `t_${Date.now()}`,
      name: config.name || `Torneo #${Math.floor(Math.random() * 10000)}`,
      format: config.format || TOURNAMENT_FORMATS.ROUND_ROBIN,
      players: config.players || [],
      matches: [],
      currentMatchIndex: 0,
      status: TOURNAMENT_STATUS.SETUP,
      createdAt: new Date().toISOString(),
      settings: {
        difficulty: config.difficulty || 'medium',
        duration: config.duration || 3,
        halfDuration: config.halfDuration || 1.5
      }
    };
    await this.storage.saveTournament(tournament);
    return tournament;
  }

  async getTournament(id) {
    return this.storage.getTournament(id);
  }

  async getAllTournaments() {
    return this.storage.getTournaments();
  }

  async addPlayer(tournamentId, playerName, isHuman = true) {
    const tournament = await this.getTournament(tournamentId);
    if (!tournament) return null;
    const player = {
      name: playerName,
      isHuman,
      totalScore: 0,
      matchesPlayed: 0,
      wins: 0,
      currentMatchIndex: 0
    };
    tournament.players.push(player);
    await this.storage.saveTournament(tournament);
    return player;
  }

  async startTournament(tournamentId) {
    const tournament = await this.getTournament(tournamentId);
    if (!tournament || tournament.players.length < 2) return null;
    tournament.status = TOURNAMENT_STATUS.IN_PROGRESS;
    await this.storage.saveTournament(tournament);
    return tournament;
  }

  async recordMatchResult(tournamentId, matchIndex, homePlayerIdx, awayPlayerIdx, homeScore, awayScore) {
    const tournament = await this.getTournament(tournamentId);
    if (!tournament) return null;

    const match = {
      index: matchIndex,
      homePlayerIdx,
      awayPlayerIdx,
      homeScore,
      awayScore,
      winner: homeScore > awayScore ? homePlayerIdx : (awayScore > homeScore ? awayPlayerIdx : -1)
    };

    tournament.matches.push(match);

    const homePlayer = tournament.players[homePlayerIdx];
    const awayPlayer = tournament.players[awayPlayerIdx];

    homePlayer.matchesPlayed++;
    awayPlayer.matchesPlayed++;

    if (match.winner === homePlayerIdx) {
      homePlayer.wins++;
      homePlayer.totalScore += 3;
    } else if (match.winner === awayPlayerIdx) {
      awayPlayer.wins++;
      awayPlayer.totalScore += 3;
    } else {
      homePlayer.totalScore += 1;
      awayPlayer.totalScore += 1;
    }

    tournament.currentMatchIndex = tournament.matches.length;
    await this.storage.saveTournament(tournament);
    return match;
  }

  async getStandings(tournamentId) {
    const tournament = await this.storage.getTournament(tournamentId);
    if (!tournament) return [];
    return [...tournament.players].sort((a, b) => b.totalScore - a.totalScore || b.wins - a.wins);
  }

  async getNextMatch(tournamentId) {
    const tournament = await this.storage.getTournament(tournamentId);
    if (!tournament || tournament.status !== TOURNAMENT_STATUS.IN_PROGRESS) return null;
    if (tournament.currentMatchIndex >= tournament.matches.length) return null;
    return tournament.matches[tournament.currentMatchIndex];
  }

  async isTournamentComplete(tournamentId) {
    const tournament = await this.getTournament(tournamentId);
    if (!tournament) return false;
    if (tournament.format === TOURNAMENT_FORMATS.ROUND_ROBIN) {
      const n = tournament.players.length;
      const totalMatches = (n * (n - 1)) / 2;
      return tournament.matches.filter(m => m.played).length >= totalMatches;
    }
    return false;
  }

  async getWinner(tournamentId) {
    const standings = await this.getStandings(tournamentId);
    return standings.length > 0 ? standings[0] : null;
  }
}

export { TournamentManager, TOURNAMENT_FORMATS, TOURNAMENT_STATUS };
