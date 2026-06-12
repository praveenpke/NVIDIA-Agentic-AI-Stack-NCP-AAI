/* Theme bootstrap + toggle. Include in <head> so the theme applies before first paint. */
(function () {
  var saved = null;
  try { saved = localStorage.getItem('ncp-theme'); } catch (e) {}
  var theme = saved ||
    ((window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches) ? 'light' : 'dark');
  document.documentElement.setAttribute('data-theme', theme);

  function syncIcon() {
    var btn = document.querySelector('.theme-toggle');
    if (!btn) return;
    var current = document.documentElement.getAttribute('data-theme');
    btn.textContent = current === 'light' ? '\u{1F319}' : '☀️'; /* moon : sun */
    btn.setAttribute('aria-label', 'Switch to ' + (current === 'light' ? 'dark' : 'light') + ' mode');
  }

  window.toggleTheme = function () {
    var next = document.documentElement.getAttribute('data-theme') === 'light' ? 'dark' : 'light';
    document.documentElement.setAttribute('data-theme', next);
    try { localStorage.setItem('ncp-theme', next); } catch (e) {}
    syncIcon();
  };

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', syncIcon);
  } else {
    syncIcon();
  }
})();
