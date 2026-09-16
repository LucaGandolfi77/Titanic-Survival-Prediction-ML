/* ===== JSDoc type definitions for the data model =====
 *
 * Editor-time checking only (no runtime code). Reference with:
 *   @type {import('./types.js').Skater}
 * or import in a module and use the namespace.
 */

/**
 * One of the five skater attributes.
 * @typedef {'technique'|'stamina'|'rhythm'|'sync'|'charisma'} StatKey
 */

/**
 * @typedef {Object} SkaterStats
 * @property {number} technique
 * @property {number} stamina
 * @property {number} rhythm
 * @property {number} sync
 * @property {number} charisma
 */

/**
 * @typedef {Object} Nationality
 * @property {string} flag   Emoji flag
 * @property {string} country
 */

/**
 * @typedef {Object} SkaterContract
 * @property {number} weeksRemaining
 * @property {number} wage
 */

/**
 * A skater (squad, reserve, market or listed).
 * @typedef {Object} Skater
 * @property {string} id
 * @property {string} name
 * @property {number} age
 * @property {Nationality} nationality
 * @property {string} avatar            Emoji avatar
 * @property {SkaterStats} stats
 * @property {number} overall           Weighted average 0–99
 * @property {number} value             Market value in €
 * @property {number} wage              Weekly wage in €
 * @property {number} form              0–100
 * @property {number} morale            0–100
 * @property {'active'|'reserve'|'market'|'injured'} status
 * @property {number} injuryWeeks
 * @property {SkaterContract} contract
 * @property {number} [askingPrice]     Market/listed skaters only
 * @property {boolean} [scouted]        Scouted market skaters only
 */

/**
 * @typedef {Object} Competition
 * @property {number} week              1-based week number
 * @property {string|null} name         Null = training week
 * @property {number} tier              0 (training) or 1–4
 * @property {boolean} [training]       True for training weeks
 * @property {number} [entryFee]
 * @property {Object<number, number>} [prizes]   placement → €
 * @property {Object<number, number>} [pointsPool] placement → points
 * @property {Object<number, number>} [fameReward] placement → fame
 * @property {number} [minOverall]
 * @property {CompetitionResult|null} competition Result after playing
 */

/**
 * @typedef {Object} Rival
 * @property {string} name
 * @property {number} strength          30–90
 * @property {number} points
 * @property {number} fame
 * @property {number} wins
 * @property {number} money
 */

/**
 * @typedef {Object} CompetitionResult
 * @property {number} season
 * @property {number} week
 * @property {string} competition
 * @property {number} tier
 * @property {number} playerScore
 * @property {number} placement
 * @property {number} prizeMoney
 * @property {number} pointsAwarded
 * @property {number} fameAwarded
 * @property {Array<{team:string, score:number, isPlayer:boolean, placement:number}>} leaderboard
 */

/**
 * @typedef {Object} Sponsor
 * @property {string} id
 * @property {string} name
 * @property {string} icon
 * @property {number} requiredFame
 * @property {number} weeklyIncome
 * @property {number} duration          Weeks
 * @property {string} description
 * @property {{type:'charisma'|'overall', min:number}|null} requirement
 * @property {boolean} [tempoBonus]     CoolBreeze perk
 * @property {number} [fameBonus]       IcyPeak perk
 * @property {number} [winPointsBonus]  NovaSport perk
 * @property {number} [syncBonus]       QuantumIce perk (fraction)
 */

/**
 * @typedef {Object} SponsorDeal
 * @property {Sponsor} sponsor
 * @property {number} weeksRemaining
 * @property {number} breachCount
 */

/**
 * @typedef {Object} Formation
 * @property {string} id
 * @property {string} name
 * @property {string} emoji
 * @property {number} difficulty        Score multiplier
 * @property {number} unlockFame
 * @property {Array<{x:number, y:number}>} positions  16 normalized positions
 */

export {};
