// PWA-to-PWA co-op flirtation ("Duet"): one player plays the suitor (the
// normal game), the other plays the crush and reacts to every choice —
// their reaction is what the suitor's device shows instead of the local one.
// Same device: BroadcastChannel. Two devices: WebRTC data channel with
// manual copy/paste signaling (no server).

import { setState } from '../../core/state.js';

const CHANNEL = 'speed-crush-duet';

let session = null;

export function sameDeviceSupported() {
  return typeof window !== 'undefined' && typeof window.BroadcastChannel === 'function';
}

export function crossDeviceSupported() {
  return typeof window !== 'undefined' && typeof window.RTCPeerConnection === 'function';
}

export function isConnected() {
  return !!session?.connected;
}

export function getRole() {
  return session?.role || null;
}

/* ---------- transports ---------- */

function createBroadcastTransport() {
  const channel = new BroadcastChannel(CHANNEL);
  const listeners = new Set();
  channel.onmessage = (event) => listeners.forEach((cb) => cb(event.data));
  return {
    kind: 'broadcast',
    signaling: 'auto',
    onMessage(cb) {
      listeners.add(cb);
    },
    offMessage(cb) {
      listeners.delete(cb);
    },
    send(msg) {
      try {
        channel.postMessage(msg);
      } catch { /* channel closed */ }
    },
    close() {
      try {
        channel.close();
      } catch { /* noop */ }
    }
  };
}

async function waitIce(pc, timeoutMs = 2500) {
  if (pc.iceGatheringState === 'complete') return;
  await new Promise((resolve) => {
    const done = () => {
      pc.removeEventListener('icegatheringstatechange', check);
      clearTimeout(timer);
      resolve();
    };
    const check = () => {
      if (pc.iceGatheringState === 'complete') done();
    };
    const timer = setTimeout(done, timeoutMs);
    pc.addEventListener('icegatheringstatechange', check);
  });
}

async function createRtcTransport(onStatus) {
  const pc = new RTCPeerConnection();
  const channel = pc.createDataChannel('duet', { ordered: true });
  const listeners = new Set();
  channel.onmessage = (event) => {
    try {
      const msg = JSON.parse(event.data);
      listeners.forEach((cb) => cb(msg));
    } catch { /* malformed frame */ }
  };
  channel.onopen = () => {
    session.connected = true;
    onStatus?.('connected');
  };
  channel.onclose = () => onStatus?.('disconnected');

  return {
    kind: 'rtc',
    signaling: 'manual',
    onMessage(cb) {
      listeners.add(cb);
    },
    offMessage(cb) {
      listeners.delete(cb);
    },
    send(msg) {
      try {
        if (channel.readyState === 'open') channel.send(JSON.stringify(msg));
      } catch { /* channel closed */ }
    },
    close() {
      try {
        channel.close();
        pc.close();
      } catch { /* noop */ }
    },
    async createOfferCode() {
      const offer = await pc.createOffer();
      await pc.setLocalDescription(offer);
      await waitIce(pc);
      return btoa(JSON.stringify(pc.localDescription));
    },
    async acceptOfferCode(code) {
      await pc.setRemoteDescription(JSON.parse(atob(code)));
      const answer = await pc.createAnswer();
      await pc.setLocalDescription(answer);
      await waitIce(pc);
      return btoa(JSON.stringify(pc.localDescription));
    },
    async acceptAnswerCode(code) {
      await pc.setRemoteDescription(JSON.parse(atob(code)));
    }
  };
}

/* ---------- session ---------- */

export async function createSession({ role, crossDevice = false, onMessage, onStatus } = {}) {
  closeSession();
  const transport = crossDevice ? await createRtcTransport(onStatus) : createBroadcastTransport();
  session = { role, transport, connected: !crossDevice, onMessage, onStatus };
  transport.onMessage((msg) => session?.onMessage?.(msg));
  transport.send({ type: 'hello', role });
  return session;
}

export function send(msg) {
  session?.transport?.send(msg);
}

/**
 * Suitor side: wait for the crush's reaction (up to `timeoutMs`).
 * Resolves with the reaction text, or null on timeout.
 */
export function waitForReaction(timeoutMs = 6000) {
  return new Promise((resolve) => {
    if (!session) {
      resolve(null);
      return;
    }
    let handler = null;
    const timer = setTimeout(() => {
      session?.transport?.offMessage?.(handler);
      resolve(null);
    }, timeoutMs);
    handler = (msg) => {
      if (msg?.type === 'reaction') {
        clearTimeout(timer);
        session?.transport?.offMessage?.(handler);
        resolve(msg.text);
      }
    };
    session.transport.onMessage(handler);
  });
}

export function closeSession() {
  session?.transport?.close();
  session = null;
}

/* ---------- UI (duet overlay) ---------- */

const GUEST_REACTION_LINES = [
  { text: 'laughs — bold move.', tone: 'good' },
  { text: 'nods slowly, curious.', tone: 'safe' },
  { text: 'raises an eyebrow. Recover.', tone: 'risky' }
];

let els = null;
let characterName = 'Your crush';
let pendingChoice = false;

export function initDuet(elsRef) {
  els = elsRef;
  if (!els.duetOverlay) return;
  els.duetClose?.addEventListener('click', closeDuet);
  els.duetTransport?.addEventListener('change', updateTransportUi);
  els.duetRole?.addEventListener('change', updateTransportUi);
  els.duetCodeBtn?.addEventListener('click', createInviteCode);
  els.duetConnectBtn?.addEventListener('click', connectWithCode);
  els.duetStartBtn?.addEventListener('click', startDuet);
  updateTransportUi();
}

export function openDuet() {
  if (!els?.duetOverlay) return;
  els.duetOverlay.classList.add('open');
  els.duetOverlay.setAttribute('aria-hidden', 'false');
  showSetup();
}

function closeDuet() {
  if (!els?.duetOverlay?.classList.contains('open')) return;
  els.duetOverlay.classList.remove('open');
  els.duetOverlay.setAttribute('aria-hidden', 'true');
}

function updateTransportUi() {
  if (!els) return;
  const crossDevice = els.duetTransport?.value === 'rtc';
  els.duetCodeBtn.hidden = !crossDevice;
  els.duetConnectBtn.hidden = !crossDevice || !els.duetCodeIn?.value;
}

function showSetup() {
  els.duetSetup.hidden = false;
  els.duetPlay.hidden = true;
  els.duetStatus.textContent = 'Pick a role to begin.';
}

function showPlay() {
  els.duetSetup.hidden = true;
  els.duetPlay.hidden = false;
}

async function createInviteCode() {
  const role = els.duetRole?.value || 'suitor';
  try {
    els.duetStatus.textContent = 'Creating invite code…';
    await createSession({
      role,
      crossDevice: true,
      onMessage: handleGuestMessage,
      onStatus: (status) => {
        els.duetStatus.textContent = status === 'connected' ? 'Connected ✓' : status;
      }
    });
    els.duetCodeOut.hidden = false;
    els.duetCodeOut.value = await session.transport.createOfferCode();
    els.duetStatus.textContent = 'Share this invite code, then paste the reply below.';
  } catch (err) {
    console.error('Invite code failed', err);
    els.duetStatus.textContent = 'Could not create the invite code.';
  }
}

async function connectWithCode() {
  const code = (els.duetCodeIn?.value || '').trim();
  if (!code) return;
  try {
    els.duetStatus.textContent = 'Connecting…';
    if (!session) {
      await createSession({
        role: els.duetRole?.value || 'character',
        crossDevice: true,
        onMessage: handleGuestMessage,
        onStatus: (status) => {
          els.duetStatus.textContent = status === 'connected' ? 'Connected ✓' : status;
        }
      });
    }
    if (session.role === 'character') {
      els.duetCodeOut.hidden = false;
      els.duetCodeOut.value = await session.transport.acceptOfferCode(code);
      els.duetStatus.textContent = 'Send this reply code back, then wait for the story.';
    } else {
      await session.transport.acceptAnswerCode(code);
      els.duetStatus.textContent = 'Connected ✓ — start the story!';
    }
  } catch (err) {
    console.error('Connect failed', err);
    els.duetStatus.textContent = 'Invalid code — try again.';
  }
}

async function startDuet() {
  const role = els.duetRole?.value || 'suitor';
  if (!isConnected()) {
    // same-device flow: open a session in BOTH tabs, then the host starts.
    if (role === 'character') {
      if (!session) {
        await createSession({ role: 'character', onMessage: handleGuestMessage });
      }
      showPlay();
      els.duetScene.textContent = 'Waiting for the story…';
      return;
    }
    if (!session) {
      await createSession({ role: 'suitor', onMessage: handleGuestMessage });
    }
  }
  if (role !== 'suitor') {
    els.duetStatus.textContent = 'Only the suitor starts the story.';
    return;
  }
  closeDuet();
  const { startGame } = await import('../game/game.js');
  setState({ duetActive: true });
  await startGame();
}

function handleGuestMessage(msg) {
  if (!msg || !els) return;
  if (msg.type === 'hello') {
    els.duetStatus.textContent = 'Crush connected ✓';
    return;
  }
  if (session?.role !== 'character') return; // only the guest renders
  if (msg.type === 'scene') {
    characterName = msg.character || 'Your crush';
    els.duetScene.textContent = msg.text;
    pendingChoice = false;
    els.duetChoice.textContent = '';
    els.duetReactions.innerHTML = '';
  }
  if (msg.type === 'choice') {
    els.duetChoice.textContent = `“${msg.text}”`;
    pendingChoice = true;
    renderReactions();
  }
  if (msg.type === 'ending') {
    els.duetScene.textContent = 'The story ended 💘';
    els.duetChoice.textContent = msg.text || '';
    els.duetReactions.innerHTML = '';
    pendingChoice = false;
  }
}

function renderReactions() {
  if (!els?.duetReactions) return;
  els.duetReactions.innerHTML = '';
  for (const reaction of GUEST_REACTION_LINES) {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = `duet-reaction-btn ${reaction.tone}`;
    btn.textContent = `${characterName} ${reaction.text}`;
    btn.addEventListener('click', () => {
      if (!pendingChoice) return;
      pendingChoice = false;
      send({ type: 'reaction', text: `${characterName} ${reaction.text}` });
      els.duetReactions.innerHTML = '';
      els.duetChoice.textContent = `You reacted: ${characterName} ${reaction.text}`;
    });
    els.duetReactions.appendChild(btn);
  }
}
