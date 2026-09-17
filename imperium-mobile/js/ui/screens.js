const SCREEN_IDS = [
  'screen-loading',
  'screen-menu',
  'screen-campaign',
  'screen-campaign-brief',
  'screen-campaign-result',
  'screen-skirmish',
  'screen-settings',
  'screen-howto',
  'screen-profile',
  'screen-pause',
  'screen-gameover',
  'screen-victory',
];

export const Screens = {
  current: null,

  show(screenId) {
    SCREEN_IDS.forEach(id => {
      const el = document.getElementById(id);
      if (el) el.classList.add('hidden');
    });
    const target = document.getElementById(screenId);
    if (target) {
      target.classList.remove('hidden');
      target.classList.add('fade-in');
      setTimeout(() => target.classList.remove('fade-in'), 300);
    }
    this.current = screenId;
    if (screenId === 'screen-campaign') {
      if (window._imperium && window._imperium.refreshCampaignList) window._imperium.refreshCampaignList();
    }
    if (screenId !== 'screen-pause' && screenId !== 'screen-gameover' && screenId !== 'screen-victory') {
      document.getElementById('hud')?.classList.remove('hidden');
      document.getElementById('touch-controls')?.classList.remove('hidden');
      document.getElementById('mini-map-container')?.classList.remove('hidden');
    }
  },

  hideHud() {
    document.getElementById('hud')?.classList.add('hidden');
    document.getElementById('touch-controls')?.classList.add('hidden');
    document.getElementById('mini-map-container')?.classList.add('hidden');
  },
};
