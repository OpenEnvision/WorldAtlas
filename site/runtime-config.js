(function configureWorldArenaRuntime(global) {
  const defaults = {
    // Example: https://media.world-arena.ai/catalog/2026-07-22
    archiveCatalogBase: "",
    archiveCatalogPageSize: 480,
    // Example: https://media.world-arena.ai
    mediaOrigin: ""
  };

  global.WORLDARENA_CONFIG = Object.freeze({
    ...defaults,
    ...(global.WORLDARENA_CONFIG || {})
  });
})(window);
