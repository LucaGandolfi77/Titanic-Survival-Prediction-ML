import { Vector2 } from '../utils.js';

const AI_BACKEND = {
  WEBNN: 'webnn',
  WASM: 'wasm',
  HEURISTIC: 'heuristic'
};

class AIBrain {
  constructor(difficulty) {
    this.difficulty = difficulty;
    this.backend = null;
    this.backendType = AI_BACKEND.HEURISTIC;
    this.patternLearner = new PatternLearner();
    this.initialized = false;
    this._init();
  }

  async _init() {
    try {
      if (await this._detectWebNN()) {
        this.backendType = AI_BACKEND.WEBNN;
        this.backend = await this._createWebNNModel();
      } else {
        this.backendType = AI_BACKEND.HEURISTIC;
        this.backend = null;
      }
    } catch (e) {
      console.warn('WebNN unavailable, using heuristic backend', e);
      this.backendType = AI_BACKEND.HEURISTIC;
      this.backend = null;
    }
    this.initialized = true;
  }

  async _detectWebNN() {
    if (!('ml' in navigator)) return false;
    try {
      const ml = await navigator.ml;
      const devices = await ml.availableDevices();
      return devices.length > 0;
    } catch {
      return false;
    }
  }

  async _createWebNNModel() {
    const ml = await navigator.ml;
    const model = await ml.createModel({
      inputs: [
        { dtype: 'float32', dimensions: [24] }
      ],
      outputs: [
        { dtype: 'float32', dimensions: [5] }
      ]
    });
    return model;
  }

  async predict(gameState) {
    const features = this._extractFeatures(gameState);

    if (this.backendType === AI_BACKEND.WEBNN && this.backend) {
      try {
        const input = new Float32Array(features);
        const feeds = { 'input': input };
        const results = await this.backend.compute(feeds);
        const output = results.output.data;
        return this._interpretOutput(output);
      } catch (e) {
        console.warn('WebNN inference failed, fallback to heuristic', e);
        return this._heuristicDecide(gameState);
      }
    }

    return this._heuristicDecide(gameState);
  }

  _extractFeatures(gameState) {
    const { ball, players, myTeamIndex } = gameState;
    const myTeam = players.filter(p => p.team === myTeamIndex);
    const opponentTeam = players.filter(p => p.team !== myTeamIndex);

    const features = [];

    features.push(ball.pos.x / 800);
    features.push(ball.pos.y / 500);
    features.push(ball.vel.x / 800);
    features.push(ball.vel.y / 500);
    features.push(ball.owner ? ball.owner.team === myTeamIndex ? 1 : -1 : 0);

    myTeam.forEach(p => {
      features.push((p.pos.x - ball.pos.x) / 800);
      features.push((p.pos.y - ball.pos.y) / 500);
      features.push(p.hasBall ? 1 : 0);
      features.push(p.stamina / 100);
    });

    while (features.length < 16) features.push(0);

    const opponentHasBall = opponentTeam.some(p => p.hasBall);
    features.push(opponentHasBall ? 1 : 0);

    opponentTeam.forEach(p => {
      features.push((p.pos.x - ball.pos.x) / 800);
      features.push((p.pos.y - ball.pos.y) / 500);
    });

    while (features.length < 24) features.push(0);

    return features.slice(0, 24);
  }

  _interpretOutput(output) {
    const actions = ['idle', 'move', 'shoot', 'pass', 'tackle'];
    const maxIdx = output.indexOf(Math.max(...output));
    return {
      action: actions[maxIdx] || 'idle',
      confidence: output[maxIdx],
      target: null
    };
  }

  _heuristicDecide(gameState) {
    const { ball, players, myTeamIndex } = gameState;
    const myTeam = players.filter(p => p.team === myTeamIndex);    const config = {
      easy: { reactionTime: 0.4, aggressiveness: 0.4, accuracy: 0.6 },
      medium: { reactionTime: 0.2, aggressiveness: 0.65, accuracy: 0.8 },
      hard: { reactionTime: 0.05, aggressiveness: 0.85, accuracy: 0.95 }
    };
    const cfg = config[this.difficulty] || config.medium;

    const predictions = [];

    myTeam.forEach(p => {
      let action = 'idle';
      let target = new Vector2(0, 0);

      if (p.hasBall) {
        const goalX = myTeamIndex === 0 ? 800 : 0;
        const goalCenter = new Vector2(goalX, 250);
        const distToGoal = p.pos.distanceTo(goalCenter);

        if (distToGoal < 300) {
          action = 'shoot';
          target = goalCenter;
        } else {
          let bestPass = null;
          let bestScore = -Infinity;
          myTeam.forEach(mate => {
            if (mate === p) return;
            const isForward = (myTeamIndex === 0) ? (mate.pos.x > p.pos.x) : (mate.pos.x < p.pos.x);
            if (isForward) {
              const score = mate.pos.distanceTo(goalCenter) * -1;
              if (score > bestScore) { bestScore = score; bestPass = mate; }
            }
          });
          if (bestPass && Math.random() < cfg.accuracy) {
            action = 'pass';
            target = bestPass.pos;
          } else {
            action = 'move';
            target = goalCenter;
          }
        }
      } else if (ball.owner && ball.owner.team !== myTeamIndex) {
        const distToBall = p.pos.distanceTo(ball.pos);
        if (distToBall < 150) {
          action = 'move';
          target = ball.pos;
          if (distToBall < 20 && Math.random() < cfg.aggressiveness) action = 'tackle';
        } else {
          target = this._getFormationPos(p.role, myTeamIndex, ball.pos);
        }
      } else {
        const distToBall = p.pos.distanceTo(ball.pos);
        if (distToBall < 200) {
          action = 'move';
          target = ball.pos;
        } else {
          target = this._getFormationPos(p.role, myTeamIndex, ball.pos);
        }
      }

      predictions.push({ player: p, action, target, confidence: 0.8 });
    });

    return predictions;
  }

  _getFormationPos(role, teamIndex, ballPos) {
    const isHome = teamIndex === 0;
    let baseX = 0, baseY = 250;

    if (role === 0) { baseX = isHome ? 0.05 : 0.95; baseY = 0.5; }
    else if (role === 1) { baseX = isHome ? 0.25 : 0.75; baseY = 0.3; }
    else if (role === 2) { baseX = isHome ? 0.25 : 0.75; baseY = 0.7; }
    else if (role === 3) { baseX = isHome ? 0.45 : 0.55; baseY = 0.4; }
    else if (role === 4) { baseX = isHome ? 0.45 : 0.55; baseY = 0.6; }

    const fieldW = 800, fieldH = 500;
    const ballFactor = ballPos.x / fieldW;
    let shiftX = (ballFactor - 0.5) * 0.2;
    if (!isHome) shiftX *= -1;

    return new Vector2((baseX + shiftX) * fieldW, baseY * fieldH);
  }

  recordDecision(state, action, success) {
    this.patternLearner.record(state, action, success);
  }

  getPatterns() {
    return this.patternLearner.getPatterns();
  }

  async savePatterns() {
    return this.patternLearner.save();
  }
}

class PatternLearner {
  constructor() {
    this.decisions = [];
    this.playerPatterns = new Map();
  }

  record(state, action, success) {
    this.decisions.push({
      timestamp: Date.now(),
      state: this._snapshotState(state),
      action,
      success
    });
    if (this.decisions.length > 1000) {
      this.decisions = this.decisions.slice(-500);
    }
  }

  _snapshotState(state) {
    if (!state || !state.ball) return null;
    return {
      ballX: state.ball.pos.x,
      ballY: state.ball.pos.y,
      ballOwner: state.ball.owner ? state.ball.owner.team : -1,
      ballHasOwner: !!state.ball.owner
    };
  }

  getPatterns() {
    const patterns = {};
    const byAction = {};

    this.decisions.forEach(d => {
      if (!d.state) return;
      const key = `${Math.floor(d.state.ballX / 100)}_${Math.floor(d.state.ballY / 100)}_${d.state.ballOwner}`;
      if (!byAction[key]) byAction[key] = {};
      if (!byAction[key][d.action]) byAction[key][d.action] = { count: 0, success: 0 };
      byAction[key][d.action].count++;
      if (d.success) byAction[key][d.action].success++;
    });

    for (const [key, actions] of Object.entries(byAction)) {
      patterns[key] = {};
      for (const [action, data] of Object.entries(actions)) {
        patterns[key][action] = data.count > 2 ? data.success / data.count : 0;
      }
    }

    return patterns;
  }

  async save() {
    try {
      const patterns = this.getPatterns();
      localStorage.setItem('pf_ai_patterns', JSON.stringify(patterns));
    } catch (e) {
      console.warn('Failed to save AI patterns', e);
    }
  }

  async load() {
    try {
      const data = localStorage.getItem('pf_ai_patterns');
      if (data) return JSON.parse(data);
    } catch (e) {
      console.warn('Failed to load AI patterns', e);
    }
    return null;
  }
}

export { AIBrain, PatternLearner, AI_BACKEND };
