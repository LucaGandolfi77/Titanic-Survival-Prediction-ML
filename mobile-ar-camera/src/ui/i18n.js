export function createI18n(defaultLocale = 'en') {
  const locales = {
    en: {
      camera_access_needed: 'Camera access needed',
      camera_access_description: 'Allow access to use live filters and AR tracking.',
      retry: 'Retry',
      secure_context_required: 'Secure context required',
      secure_context_description: 'Use HTTPS or localhost so camera APIs can work on mobile Safari.',
      webgl_unavailable: 'WebGL unavailable. Shader filters disabled.',
      camera_permission_denied: 'Camera permission denied. Retry to continue.',
      loading_filter: 'Loading filter…',
      ar_unavailable: 'AR unavailable. Shader filters still work.',
      ar_tracking_failed: 'AR tracking failed.',
      shader_filters_unavailable: 'Shader filters are unavailable on this device.',
      shutter: 'Take photo',
      save_share: 'Save / Share',
      discard: 'Discard',
      torch_not_available: 'Torch not available on this camera.',
      share_cancelled: 'Share cancelled or failed.',
      filter_loaded: 'Filter loaded',
      filter_error: 'Filter failed to load',
    },
    it: {
      camera_access_needed: 'Accesso alla fotocamera necessario',
      camera_access_description: 'Consenti l\'accesso per usare filtri live e tracking AR.',
      retry: 'Riprova',
      secure_context_required: 'Contesto sicuro richiesto',
      secure_context_description: 'Usa HTTPS o localhost affinché le API della fotocamera funzionino su Safari mobile.',
      webgl_unavailable: 'WebGL non disponibile. Filtri shader disabilitati.',
      camera_permission_denied: 'Permesso fotocamera negato. Riprova per continuare.',
      loading_filter: 'Caricamento filtro…',
      ar_unavailable: 'AR non disponibile. I filtri shader funzionano ancora.',
      ar_tracking_failed: 'Tracking AR fallito.',
      shader_filters_unavailable: 'I filtri shader non sono disponibili su questo dispositivo.',
      shutter: 'Scatta foto',
      save_share: 'Salva / Condividi',
      discard: 'Scarta',
      torch_not_available: 'Flash non disponibile su questa fotocamera.',
      share_cancelled: 'Condivisione annullata o fallita.',
      filter_loaded: 'Filtro caricato',
      filter_error: 'Il filtro non è stato caricato',
    },
    es: {
      camera_access_needed: 'Acceso a la cámara necesario',
      camera_access_description: 'Permite el acceso para usar filtros en vivo y tracking AR.',
      retry: 'Reintentar',
      secure_context_required: 'Contexto seguro requerido',
      secure_context_description: 'Usa HTTPS o localhost para que las API de cámara funcionen en Safari móvil.',
      webgl_unavailable: 'WebGL no disponible. Filtros shader deshabilitados.',
      camera_permission_denied: 'Permiso de cámara denegado. Reintenta para continuar.',
      loading_filter: 'Cargando filtro…',
      ar_unavailable: 'AR no disponible. Los filtros shader aún funcionan.',
      ar_tracking_failed: 'Tracking AR fallido.',
      shader_filters_unavailable: 'Los filtros shader no están disponibles en este dispositivo.',
      shutter: 'Tomar foto',
      save_share: 'Guardar / Compartir',
      discard: 'Descartar',
      torch_not_available: 'Flash no disponible en esta cámara.',
      share_cancelled: 'Compartir cancelado o fallido.',
      filter_loaded: 'Filtro cargado',
      filter_error: 'El filtro no se pudo cargar',
    },
    fr: {
      camera_access_needed: 'Accès caméra nécessaire',
      camera_access_description: 'Autorisez l\'accès pour utiliser les filtres en direct et le suivi AR.',
      retry: 'Réessayer',
      secure_context_required: 'Contexte sécurisé requis',
      secure_context_description: 'Utilisez HTTPS ou localhost pour que les API caméra fonctionnent sur Safari mobile.',
      webgl_unavailable: 'WebGL indisponible. Filtres shader désactivés.',
      camera_permission_denied: 'Permission caméra refusée. Réessayez pour continuer.',
      loading_filter: 'Chargement du filtre…',
      ar_unavailable: 'AR indisponible. Les filtres shader fonctionnent encore.',
      ar_tracking_failed: 'Suivi AR échoué.',
      shader_filters_unavailable: 'Les filtres shader ne sont pas disponibles sur cet appareil.',
      shutter: 'Prendre une photo',
      save_share: 'Enregistrer / Partager',
      discard: 'Abandonner',
      torch_not_available: 'Flash non disponible sur cet appareil photo.',
      share_cancelled: 'Partage annulé ou échoué.',
      filter_loaded: 'Filtre chargé',
      filter_error: 'Le filtre n\'a pas pu être chargé',
    },
    de: {
      camera_access_needed: 'Kamerazugriff erforderlich',
      camera_access_description: 'Zugriff erlauben für Live-Filter und AR-Tracking.',
      retry: 'Erneut versuchen',
      secure_context_required: 'Sicherer Kontext erforderlich',
      secure_context_description: 'Verwenden Sie HTTPS oder localhost, damit Kamer-APIs auf Safari mobile funktionieren.',
      webgl_unavailable: 'WebGL nicht verfügbar. Shader-Filter deaktiviert.',
      camera_permission_denied: 'Kameraberechtigung verweigert. Erneut versuchen.',
      loading_filter: 'Filter wird geladen…',
      ar_unavailable: 'AR nicht verfügbar. Shader-Filter funktionieren weiterhin.',
      ar_tracking_failed: 'AR-Tracking fehlgeschlagen.',
      shader_filters_unavailable: 'Shader-Filter sind auf diesem Gerät nicht verfügbar.',
      shutter: 'Foto aufnehmen',
      save_share: 'Speichern / Teilen',
      discard: 'Verwerfen',
      torch_not_available: 'Blitz nicht verfügbar auf dieser Kamera.',
      share_cancelled: 'Teilen abgebrochen oder fehlgeschlagen.',
      filter_loaded: 'Filter geladen',
      filter_error: 'Filter konnte nicht geladen werden',
    },
  };

  let currentLocale = defaultLocale;
  const localeCallbacks = [];

  function t(key, params = {}) {
    const dict = locales[currentLocale] || locales.en;
    let text = dict[key] || locales.en[key] || key;
    Object.entries(params).forEach(([k, v]) => {
      text = text.replace(new RegExp(`\\{${k}\\}`, 'g'), v);
    });
    return text;
  }

  function setLocale(locale) {
    if (locales[locale]) {
      currentLocale = locale;
      localeCallbacks.forEach((cb) => cb(locale));
    }
  }

  function getLocale() {
    return currentLocale;
  }

  function getAvailableLocales() {
    return Object.keys(locales);
  }

  function onChange(callback) {
    localeCallbacks.push(callback);
    return () => {
      const idx = localeCallbacks.indexOf(callback);
      if (idx > -1) localeCallbacks.splice(idx, 1);
    };
  }

  function detectBrowserLocale() {
    const browserLang = (navigator.language || 'en').slice(0, 2);
    if (locales[browserLang]) return browserLang;
    return 'en';
  }

  return {
    t,
    setLocale,
    getLocale,
    getAvailableLocales,
    onChange,
    detectBrowserLocale,
    locales,
  };
}
