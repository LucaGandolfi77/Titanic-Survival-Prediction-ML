export class ProceduralGenerator {
    constructor() {
        this.minDifficulty = 1;
        this.maxDifficulty = 10;
    }

    generateConfig(level, playerSkill = 0.5) {
        const difficulty = MathUtils.clamp(level, this.minDifficulty, this.maxDifficulty);
        const skillBonus = 1 - playerSkill;

        const ecoCount = MathUtils.clamp(
            Math.round(3 + difficulty * 1.2 + skillBonus * 3),
            1, 15
        );

        const trashCount = MathUtils.clamp(
            Math.round(30 + difficulty * 8 - playerSkill * 10),
            20, 100
        );

        const maxHeight = MathUtils.clamp(
            Math.round(15 + difficulty * 2.5),
            10, 35
        );

        const speedMult = 1 + (difficulty - 1) * 0.08;

        const buildingDensity = MathUtils.clamp(
            0.6 + difficulty * 0.04,
            0.6, 1.0
        );

        const activistChance = MathUtils.clamp(
            0.1 + difficulty * 0.03,
            0.1, 0.4
        );

        return {
            ecoCount,
            trashCount,
            maxHeight,
            speedMult,
            buildingDensity,
            activistChance,
            difficulty,
            seed: this._seed(level)
        };
    }

    _seed(level) {
        let h = level * 2654435761;
        h = (h ^ (h >>> 16)) >>> 0;
        return h;
    }

    seededRandom(seed) {
        const x = Math.sin(seed) * 10000;
        return x - Math.floor(x);
    }

    generateLevelName(level) {
        const prefixes = ['Sunset', 'Midnight', 'Stormy', 'Foggy', 'Crystal', 'Volcanic', 'Frozen', 'Neon', 'Twilight', 'Chaos'];
        const suffixes = ['City', 'Town', 'Metropolis', 'District', 'Zone', 'Sector', 'Borough', 'Quarter'];
        const pIdx = (level - 1) % prefixes.length;
        const sIdx = (level * 3) % suffixes.length;
        return `${prefixes[pIdx]} ${suffixes[sIdx]}`;
    }

    adjustForPerformance(config, isMobile) {
        if (!isMobile) return config;
        return {
            ...config,
            trashCount: Math.round(config.trashCount * 0.7),
            maxHeight: Math.min(config.maxHeight, 20),
            ecoCount: Math.max(1, Math.round(config.ecoCount * 0.6))
        };
    }
}