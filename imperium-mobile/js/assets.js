export async function preloadAssets() {
  await Promise.all([
    import('./utils/helpers.js'),
    import('./utils/hex-math.js'),
    import('./data/factions.js'),
    import('./data/buildings.js'),
    import('./data/units.js'),
    import('./data/tech-tree.js'),
    import('./data/map-templates.js'),
  ]);
}
