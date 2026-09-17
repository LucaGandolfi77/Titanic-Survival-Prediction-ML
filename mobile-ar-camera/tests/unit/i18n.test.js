import { describe, it, expect, vi } from 'vitest';
import { createI18n } from '../../src/ui/i18n.js';

describe('createI18n', () => {
  const i18n = createI18n('en');

  it('should return translated strings', () => {
    expect(i18n.t('retry')).toBe('Retry');
    expect(i18n.t('camera_access_needed')).toBe('Camera access needed');
  });

  it('should fallback to English for missing keys', () => {
    expect(i18n.t('nonexistent_key')).toBe('nonexistent_key');
  });

  it('should support parameter interpolation', () => {
    expect(i18n.t('camera_access_description')).toBe('Allow access to use live filters and AR tracking.');
  });

  it('should switch locale', () => {
    i18n.setLocale('it');
    expect(i18n.t('retry')).toBe('Riprova');
    expect(i18n.getLocale()).toBe('it');
  });

  it('should fallback on invalid locale', () => {
    i18n.setLocale('xx');
    expect(i18n.getLocale()).toBe('it');
  });

  it('should detect browser locale', () => {
    const locale = i18n.detectBrowserLocale();
    expect(typeof locale).toBe('string');
    expect(['en', 'it', 'es', 'fr', 'de']).toContain(locale);
  });

  it('should list available locales', () => {
    const locales = i18n.getAvailableLocales();
    expect(locales.length).toBeGreaterThanOrEqual(4);
    expect(locales).toContain('en');
  });

  it('should notify onChange listeners', () => {
    const callback = vi.fn();
    const i18n2 = createI18n('en');
    i18n2.onChange(callback);
    i18n2.setLocale('es');
    expect(callback).toHaveBeenCalledWith('es');
  });

  it('should have Italian translations', () => {
    expect(i18n.locales.it.shutter).toBe('Scatta foto');
    expect(i18n.locales.it.save_share).toBe('Salva / Condividi');
  });
});
