export const LEVEL_SCHEMA_VERSION = 1;

export function validateLevelData(data) {
    if (!data || typeof data !== 'object') return { valid: false, error: 'Invalid data format' };
    if (data.version !== LEVEL_SCHEMA_VERSION) return { valid: false, error: `Unsupported version: ${data.version}` };
    if (!data.name || typeof data.name !== 'string') return { valid: false, error: 'Missing level name' };
    if (!data.settings || typeof data.settings !== 'object') return { valid: false, error: 'Missing settings' };
    if (!Array.isArray(data.buildings)) return { valid: false, error: 'Buildings must be an array' };

    const settings = data.settings;
    if (typeof settings.ecoCount !== 'number' || settings.ecoCount < 0) return { valid: false, error: 'Invalid ecoCount' };
    if (typeof settings.maxHeight !== 'number' || settings.maxHeight < 5) return { valid: false, error: 'Invalid maxHeight' };
    if (typeof settings.trashCount !== 'number' || settings.trashCount < 0) return { valid: false, error: 'Invalid trashCount' };

    for (let i = 0; i < data.buildings.length; i++) {
        const b = data.buildings[i];
        if (typeof b.x !== 'number' || typeof b.z !== 'number') return { valid: false, error: `Building ${i}: missing position` };
        if (Math.abs(b.x) > 95 || Math.abs(b.z) > 95) return { valid: false, error: `Building ${i}: out of bounds` };
        if (!['normal', 'eco', 'trash_spawner'].includes(b.type)) return { valid: false, error: `Building ${i}: invalid type` };
        if (typeof b.w !== 'number' || b.w < 1 || b.w > 15) return { valid: false, error: `Building ${i}: invalid width` };
        if (typeof b.h !== 'number' || b.h < 2 || b.h > 30) return { valid: false, error: `Building ${i}: invalid height` };
        if (typeof b.d !== 'number' || b.d < 1 || b.d > 15) return { valid: false, error: `Building ${i}: invalid depth` };
    }

    if (data.buildings.length > 200) return { valid: false, error: 'Too many buildings (max 200)' };

    return { valid: true };
}

export function encodeLevel(data) {
    try {
        const json = JSON.stringify(data);
        return btoa(unescape(encodeURIComponent(json)));
    } catch {
        return null;
    }
}

export function decodeLevel(encoded) {
    try {
        const json = decodeURIComponent(escape(atob(encoded)));
        return JSON.parse(json);
    } catch {
        return null;
    }
}

export function levelToCityConfig(data) {
    return {
        ecoCount: data.settings.ecoCount || 0,
        maxHeight: data.settings.maxHeight || 22,
        customBuildings: data.buildings || [],
        trashCount: data.settings.trashCount || 40
    };
}
