/* ===== Skater card HTML ===== */

import { formatMoney, overallColor, formDotClass } from '../utils.js';

export function skaterCardHTML(skater, mode = 'squad') {
  const s = skater.stats;
  const injured = skater.injuryWeeks > 0;
  const formClass = formDotClass(skater.form);
  const ovClass = overallColor(skater.overall);

  let actionsHTML = '';
  if (mode === 'active') {
    actionsHTML = `
      <div class="card-actions">
        <button class="card-btn demote" data-id="${skater.id}" title="Demote to Reserve">⬇</button>
        <button class="card-btn sell" data-id="${skater.id}" title="List for Sale">🏷️</button>
        <button class="card-btn release" data-id="${skater.id}" title="Release">✖</button>
      </div>`;
  } else if (mode === 'reserve') {
    actionsHTML = `
      <div class="card-actions">
        <button class="card-btn promote" data-id="${skater.id}" title="Promote to Active">⬆</button>
        <button class="card-btn sell" data-id="${skater.id}" title="List for Sale">🏷️</button>
        <button class="card-btn release" data-id="${skater.id}" title="Release">✖</button>
      </div>`;
  } else if (mode === 'market') {
    const price = skater.askingPrice || skater.value;
    actionsHTML = `
      <div class="card-actions">
        <button class="card-btn buy" data-id="${skater.id}" title="Buy">💰 ${formatMoney(price)}</button>
      </div>`;
  } else if (mode === 'listed') {
    actionsHTML = `
      <div class="card-actions">
        <button class="card-btn cancel-listing" data-id="${skater.id}" title="Cancel Listing">↩ Cancel</button>
        <span class="listed-price">${formatMoney(skater.askingPrice)}</span>
      </div>`;
  }

  return `
    <div class="skater-card ${injured ? 'injured' : ''} ${skater.scouted ? 'scouted' : ''}" data-id="${skater.id}">
      <div class="card-top">
        <span class="card-avatar">${skater.avatar}</span>
        <div class="card-info">
          <span class="card-name">${skater.name}</span>
          <span class="card-meta">${skater.nationality.flag} ${skater.age}y</span>
        </div>
        <span class="card-overall ${ovClass}">${skater.overall}</span>
      </div>
      <div class="card-stats">
        <div class="mini-bar" title="Technique ${s.technique}"><div class="mini-fill tech" style="width:${s.technique}%"></div></div>
        <div class="mini-bar" title="Stamina ${s.stamina}"><div class="mini-fill stam" style="width:${s.stamina}%"></div></div>
        <div class="mini-bar" title="Rhythm ${s.rhythm}"><div class="mini-fill rhy" style="width:${s.rhythm}%"></div></div>
        <div class="mini-bar" title="Sync ${s.sync}"><div class="mini-fill syn" style="width:${s.sync}%"></div></div>
        <div class="mini-bar" title="Charisma ${s.charisma}"><div class="mini-fill cha" style="width:${s.charisma}%"></div></div>
      </div>
      <div class="card-bottom">
        <span class="card-wage">${formatMoney(skater.wage)}/wk</span>
        <span class="card-contract" title="Contract weeks remaining">📝 ${skater.contract.weeksRemaining}w</span>
        <span class="card-form">
          <span class="form-dot ${formClass}"></span>
          ${skater.form}%
        </span>
        ${injured ? '<span class="injury-badge">🏥</span>' : ''}
      </div>
      ${actionsHTML}
    </div>
  `;
}
