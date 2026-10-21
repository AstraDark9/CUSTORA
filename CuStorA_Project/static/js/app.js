(function () {
  // Theme toggle button. The inline head script (kept per-page so it
  // can run before first paint) already set data-theme on <html>;
  // this just wires up the button that flips it afterwards.
  function initThemeToggle() {
    var btn = document.getElementById('themeToggle');
    if (!btn) return;
    btn.addEventListener('click', function () {
      var next = document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
      document.documentElement.setAttribute('data-theme', next);
      localStorage.setItem('custora-theme', next);
    });
  }

  // Off-canvas sidebar for narrow screens. A no-op on pages that
  // don't render a sidebar (login, pipeline status).
  function initMobileNav() {
    var sidebar = document.getElementById('sidebar');
    var toggle = document.getElementById('navToggle');
    var scrim = document.getElementById('sidebarScrim');
    var closeBtn = document.getElementById('sidebarClose');
    if (!sidebar || !toggle || !scrim) return;

    var lastFocused = null;

    function open() {
      lastFocused = document.activeElement;
      sidebar.classList.add('mobile-open');
      scrim.classList.add('open');
      toggle.setAttribute('aria-expanded', 'true');
      document.body.style.overflow = 'hidden';
      if (closeBtn) closeBtn.focus();
    }
    function close() {
      sidebar.classList.remove('mobile-open');
      scrim.classList.remove('open');
      toggle.setAttribute('aria-expanded', 'false');
      document.body.style.overflow = '';
      if (lastFocused && typeof lastFocused.focus === 'function') lastFocused.focus();
    }

    toggle.addEventListener('click', function () {
      if (sidebar.classList.contains('mobile-open')) close();
      else open();
    });
    scrim.addEventListener('click', close);
    if (closeBtn) closeBtn.addEventListener('click', close);
    // Navigating to a new page closes the drawer on the way out, so
    // it doesn't reopen already-expanded on the next full page load.
    sidebar.querySelectorAll('a[href]').forEach(function (link) {
      link.addEventListener('click', close);
    });

    document.addEventListener('keydown', function (e) {
      if (!sidebar.classList.contains('mobile-open')) return;
      if (e.key === 'Escape') {
        close();
        return;
      }
      if (e.key === 'Tab') {
        var focusable = Array.prototype.slice.call(
          sidebar.querySelectorAll('a[href], button:not([disabled])')
        );
        if (!focusable.length) return;
        var first = focusable[0];
        var last = focusable[focusable.length - 1];
        if (e.shiftKey && document.activeElement === first) {
          e.preventDefault();
          last.focus();
        } else if (!e.shiftKey && document.activeElement === last) {
          e.preventDefault();
          first.focus();
        }
      }
    });
  }

  document.addEventListener('DOMContentLoaded', function () {
    initThemeToggle();
    initMobileNav();
  });
})();
