// Local passkey unlock for the stats vault: Face ID / Touch ID via WebAuthn.
// Fully local — no server; the platform authenticator signs local challenges.

const CRED_KEY = 'speed-crush:credential';

export function isSupported() {
  try {
    return (
      typeof window !== 'undefined' &&
      typeof window.PublicKeyCredential === 'function' &&
      typeof navigator?.credentials?.create === 'function'
    );
  } catch {
    return false;
  }
}

export function hasPasskey() {
  try {
    return !!localStorage.getItem(CRED_KEY);
  } catch {
    return false;
  }
}

function b64urlEncode(buffer) {
  const bytes = new Uint8Array(buffer);
  let str = '';
  for (const b of bytes) str += String.fromCharCode(b);
  return btoa(str).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
}

function b64urlDecode(str) {
  const padded = str.replace(/-/g, '+').replace(/_/g, '/');
  const raw = atob(padded + '='.repeat((4 - (padded.length % 4)) % 4));
  return Uint8Array.from(raw, (c) => c.charCodeAt(0));
}

function randomChallenge() {
  const bytes = new Uint8Array(32);
  crypto.getRandomValues(bytes);
  return bytes;
}

export async function registerPasskey(name = 'Speed Crush player') {
  const credential = await navigator.credentials.create({
    publicKey: {
      challenge: randomChallenge(),
      rp: { name: 'Speed Crush' },
      user: {
        id: crypto.getRandomValues(new Uint8Array(16)),
        name,
        displayName: name
      },
      pubKeyCredParams: [
        { type: 'public-key', alg: -7 }, // ES256
        { type: 'public-key', alg: -257 } // RS256
      ],
      authenticatorSelection: {
        authenticatorAttachment: 'platform',
        userVerification: 'required',
        residentKey: 'preferred'
      },
      timeout: 60000,
      attestation: 'none'
    }
  });
  const id = b64urlEncode(credential.rawId);
  try {
    localStorage.setItem(CRED_KEY, id);
  } catch {
    /* storage unavailable — the passkey works for this session only */
  }
  return id;
}

export async function verifyPasskey() {
  let id = null;
  try {
    id = localStorage.getItem(CRED_KEY);
  } catch {
    return false;
  }
  if (!id) return false;
  const assertion = await navigator.credentials.get({
    publicKey: {
      challenge: randomChallenge(),
      allowCredentials: [{ type: 'public-key', id: b64urlDecode(id) }],
      userVerification: 'required',
      timeout: 60000
    }
  });
  return !!assertion;
}

export function forgetPasskey() {
  try {
    localStorage.removeItem(CRED_KEY);
  } catch {
    /* noop */
  }
}
