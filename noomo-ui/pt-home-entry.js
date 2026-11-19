const view = new URLSearchParams(location.search).get('view');
const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;
const overview = document.querySelector('.pt-home-overview');
const skip = document.querySelector('.pt-home-skip');

function hasGraphics() {
  try {
    const context = document.createElement('canvas').getContext('webgl2');
    context?.getExtension('WEBGL_lose_context')?.loseContext();
    return Boolean(context);
  } catch {
    return false;
  }
}

function showOverview(reason) {
  document.documentElement.classList.add('pt-home-reading');
  overview.dataset.reason = reason;
  if (!overview.isConnected) document.body.append(overview);
}

if (view === 'overview' || (reducedMotion && view !== 'scene')) {
  showOverview('preference');
} else if (!hasGraphics()) {
  showOverview('graphics');
} else {
  const timeout = setTimeout(() => showOverview('timeout'), 16000);
  window.__ptSceneReady = () => {
    clearTimeout(timeout);
    if (!document.documentElement.classList.contains('pt-home-reading')) {
      if (!skip.isConnected) document.body.append(skip);
      document.body.classList.add('pt-scene-ready');
      window.dispatchEvent(new Event('pt-scene-ready'));
    }
  };

  import('/_nuxt/iNjJ8EkR.js?pt-build=native-20261118-2').catch(() => {
    clearTimeout(timeout);
    showOverview('load');
  });
}
