/* Theme bootstrap + toggle + site-wide navigation menu.
   Include in <head> so the theme applies before first paint.
   The menu and control cluster are injected on DOMContentLoaded —
   pages only need this script + theme.css to get full site chrome. */
(function () {
  var saved = null;
  try { saved = localStorage.getItem('ncp-theme'); } catch (e) {}
  var theme = saved ||
    ((window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches) ? 'light' : 'dark');
  document.documentElement.setAttribute('data-theme', theme);

  /* ---- site map (single source of truth for the menu) ---- */
  var SITE = [
    { group: 'Start here' },
    { href: 'index.html', label: '🏠 Home — Study Hub' },
    { href: 'study-guide.html', label: '📋 Exam & Study Guide' },
    { href: 'resources.html', label: '🔗 Resources & Links' },
    { href: 'domain-00-builders-pulse.html', label: '📡 Builder’s Pulse (2025-26)' },
    { group: 'Exam domains' },
    { href: 'domain-01-agent-architecture.html', label: 'D1 · Architecture & Design', wt: '15%' },
    { href: 'domain-02-agent-development.html', label: 'D2 · Agent Development', wt: '15%' },
    { href: 'domain-03-evaluation-tuning.html', label: 'D3 · Evaluation & Tuning', wt: '13%' },
    { href: 'domain-04-deployment-scaling.html', label: 'D4 · Deployment & Scaling', wt: '13%' },
    { href: 'domain-05-cognition-planning-memory.html', label: 'D5 · Cognition & Memory', wt: '10%' },
    { href: 'domain-06-knowledge-integration.html', label: 'D6 · Knowledge & RAG', wt: '10%' },
    { href: 'domain-07-nvidia-platform.html', label: 'D7 · NVIDIA Platform', wt: '7%' },
    { href: 'domain-08-run-monitor-maintain.html', label: 'D8 · Run & Monitor', wt: '5%' },
    { href: 'domain-09-safety-ethics-compliance.html', label: 'D9 · Safety & Compliance', wt: '5%' },
    { href: 'domain-10-human-ai-interaction.html', label: 'D10 · Human-AI Oversight', wt: '5%' },
  ];

  function syncIcon() {
    var btn = document.querySelector('.theme-toggle');
    if (!btn) return;
    var current = document.documentElement.getAttribute('data-theme');
    btn.textContent = current === 'light' ? '\u{1F319}' : '☀️';
    btn.setAttribute('aria-label', 'Switch to ' + (current === 'light' ? 'dark' : 'light') + ' mode');
  }

  window.toggleTheme = function () {
    var next = document.documentElement.getAttribute('data-theme') === 'light' ? 'dark' : 'light';
    document.documentElement.setAttribute('data-theme', next);
    try { localStorage.setItem('ncp-theme', next); } catch (e) {}
    syncIcon();
  };

  function buildChrome() {
    /* move any page-authored toggle into a control cluster, add menu button */
    var cluster = document.createElement('div');
    cluster.className = 'site-controls';

    var menuBtn = document.createElement('button');
    menuBtn.className = 'site-menu-btn';
    menuBtn.textContent = '☰';
    menuBtn.title = 'Site menu';
    menuBtn.setAttribute('aria-label', 'Open site menu');
    cluster.appendChild(menuBtn);

    var toggle = document.querySelector('.theme-toggle');
    if (!toggle) {
      toggle = document.createElement('button');
      toggle.className = 'theme-toggle';
      toggle.setAttribute('onclick', 'toggleTheme()');
      toggle.title = 'Toggle light/dark mode';
    } else if (toggle.parentNode) {
      toggle.parentNode.removeChild(toggle);
    }
    cluster.appendChild(toggle);
    document.body.appendChild(cluster);

    /* slide-over menu */
    var overlay = document.createElement('div');
    overlay.className = 'site-menu-overlay';
    var menu = document.createElement('nav');
    menu.className = 'site-menu';
    menu.setAttribute('aria-label', 'Site navigation');

    var here = (location.pathname.split('/').pop() || 'index.html');
    var html = '<div class="menu-head"><span class="menu-title">NCP-AAI STUDY HUB</span>' +
               '<button class="menu-close" aria-label="Close menu">×</button></div>';
    SITE.forEach(function (item) {
      if (item.group) { html += '<div class="menu-group">' + item.group + '</div>'; return; }
      var cur = item.href === here ? ' current' : '';
      html += '<a class="' + cur.trim() + '" href="' + item.href + '"><span>' + item.label + '</span>' +
              (item.wt ? '<span class="wt">' + item.wt + '</span>' : '') + '</a>';
    });
    menu.innerHTML = html;
    document.body.appendChild(overlay);
    document.body.appendChild(menu);

    function openMenu() { overlay.classList.add('open'); menu.classList.add('open'); }
    function closeMenu() { overlay.classList.remove('open'); menu.classList.remove('open'); }
    menuBtn.addEventListener('click', openMenu);
    overlay.addEventListener('click', closeMenu);
    menu.querySelector('.menu-close').addEventListener('click', closeMenu);
    document.addEventListener('keydown', function (e) { if (e.key === 'Escape') closeMenu(); });

    syncIcon();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', buildChrome);
  } else {
    buildChrome();
  }
})();
