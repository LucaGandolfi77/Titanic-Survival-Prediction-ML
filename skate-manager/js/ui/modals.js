/* ===== Game modals: skater detail, sell, buy, competition entry ===== */

import { GameState } from '../state.js';
import { formatMoneyFull, overallColor } from '../utils.js';
import { showModal, hideModal } from './feedback.js';

// ===== Skater detail modal =====
export function showSkaterDetail(skater, actions = []) {
  const stats = skater.stats;
  const html = `
    <div class="skater-detail">
      <div class="skater-detail-header">
        <span class="skater-avatar-lg">${skater.avatar}</span>
        <div>
          <h3>${skater.name}</h3>
          <span>${skater.nationality.flag} ${skater.nationality.country} &middot; Age ${skater.age}</span>
        </div>
        <span class="overall-badge ${overallColor(skater.overall)}">${skater.overall}</span>
      </div>
      <div class="skater-detail-stats">
        ${statBar('Technique', stats.technique, '#7dd3fc')}
        ${statBar('Stamina', stats.stamina, '#34d399')}
        ${statBar('Rhythm', stats.rhythm, '#f472b6')}
        ${statBar('Sync', stats.sync, '#a78bfa')}
        ${statBar('Charisma', stats.charisma, '#fbbf24')}
      </div>
      <div class="skater-detail-info">
        <span>💰 Value: ${formatMoneyFull(skater.value)}</span>
        <span>💵 Wage: ${formatMoneyFull(skater.wage)}/wk</span>
        <span>📊 Form: ${skater.form}%</span>
        <span>😊 Morale: ${skater.morale}%</span>
        <span>📝 Contract: ${skater.contract.weeksRemaining} wks</span>
        ${skater.injuryWeeks > 0 ? `<span class="injury-text">🏥 Injured: ${skater.injuryWeeks} wks</span>` : ''}
      </div>
      ${actions.length > 0 ? `<div class="modal-buttons">${actions.map(a =>
        `<button class="modal-btn ${a.class || 'confirm'}" data-action="${a.id}">${a.label}</button>`
      ).join('')}<button class="modal-btn cancel" data-action="close">Close</button></div>` : '<div class="modal-buttons"><button class="modal-btn cancel" data-action="close">Close</button></div>'}
    </div>
  `;
  showModal(html);
  // Wire up buttons
  document.querySelectorAll('#modal-content .modal-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const action = btn.dataset.action;
      if (action === 'close') { hideModal(); return; }
      const handler = actions.find(a => a.id === action);
      if (handler && handler.fn) {
        hideModal();
        handler.fn(skater);
      }
    });
  });
}

function statBar(label, value, color) {
  return `
    <div class="stat-row-detail">
      <span class="stat-label">${label}</span>
      <div class="stat-bar-bg"><div class="stat-bar-fill" style="width:${value}%;background:${color}"></div></div>
      <span class="stat-value">${value}</span>
    </div>
  `;
}

// ===== Sell skater modal =====
export function showSellModal(skater, onConfirm) {
  const minPrice = Math.round(skater.value * 0.5);
  const maxPrice = Math.round(skater.value * 2);
  const defaultPrice = skater.value;
  const html = `
    <h3 class="modal-title">List ${skater.name} for Sale</h3>
    <p class="modal-message">Set your asking price:</p>
    <div class="sell-price-control">
      <input type="range" id="sell-price-slider" min="${minPrice}" max="${maxPrice}" value="${defaultPrice}" step="500" />
      <span id="sell-price-display" class="sell-price">${formatMoneyFull(defaultPrice)}</span>
    </div>
    <p class="modal-message" style="font-size:0.8rem;color:var(--text-dim)">Market value: ${formatMoneyFull(skater.value)}</p>
    <div class="modal-buttons">
      <button class="modal-btn danger" id="modal-sell-confirm">🏷️ List for Sale</button>
      <button class="modal-btn cancel" id="modal-sell-cancel">Cancel</button>
    </div>
  `;
  showModal(html, { persistent: true });
  const slider = document.getElementById('sell-price-slider');
  const display = document.getElementById('sell-price-display');
  slider.addEventListener('input', () => {
    display.textContent = formatMoneyFull(parseInt(slider.value));
  });
  document.getElementById('modal-sell-confirm').addEventListener('click', () => {
    hideModal();
    if (onConfirm) onConfirm(parseInt(slider.value));
  });
  document.getElementById('modal-sell-cancel').addEventListener('click', () => hideModal());
}

// ===== Buy confirmation modal =====
export function showBuyModal(skater, onConfirm) {
  const price = skater.askingPrice || skater.value;
  const html = `
    <h3 class="modal-title">Sign ${skater.name}?</h3>
    <div class="skater-detail-header" style="margin-bottom:12px;">
      <span class="skater-avatar-lg">${skater.avatar}</span>
      <div>
        <span>${skater.nationality.flag} ${skater.nationality.country} &middot; Age ${skater.age}</span>
        <span class="overall-badge ${overallColor(skater.overall)}">${skater.overall}</span>
      </div>
    </div>
    <p class="modal-message">Transfer fee: <strong>${formatMoneyFull(price)}</strong></p>
    <p class="modal-message">Weekly wage: <strong>${formatMoneyFull(skater.wage)}/wk</strong></p>
    <p class="modal-message" style="font-size:0.8rem;color:var(--text-dim)">Your balance: ${formatMoneyFull(GameState.money)}</p>
    <div class="modal-buttons">
      <button class="modal-btn confirm" id="modal-buy-confirm">💰 Sign Player</button>
      <button class="modal-btn cancel" id="modal-buy-cancel">Cancel</button>
    </div>
  `;
  showModal(html, { persistent: true });
  document.getElementById('modal-buy-confirm').addEventListener('click', () => {
    hideModal();
    if (onConfirm) onConfirm();
  });
  document.getElementById('modal-buy-cancel').addEventListener('click', () => hideModal());
}

// ===== Competition entry confirmation =====
export function showCompEntryModal(comp, onConfirm) {
  const html = `
    <h3 class="modal-title">Enter ${comp.name}?</h3>
    <div class="modal-message">
      <p>Tier ${comp.tier} competition</p>
      <p>Entry Fee: <strong>${formatMoneyFull(comp.entryFee)}</strong></p>
      <p>1st Prize: <strong>${formatMoneyFull(comp.prizes[1])}</strong></p>
      <p>Min Overall: <strong>${comp.minOverall}</strong></p>
    </div>
    <div class="modal-buttons">
      <button class="modal-btn gold" id="modal-enter-confirm">🏆 Enter Competition</button>
      <button class="modal-btn cancel" id="modal-enter-cancel">Cancel</button>
    </div>
  `;
  showModal(html, { persistent: true });
  document.getElementById('modal-enter-confirm').addEventListener('click', () => {
    hideModal();
    if (onConfirm) onConfirm();
  });
  document.getElementById('modal-enter-cancel').addEventListener('click', () => hideModal());
}
