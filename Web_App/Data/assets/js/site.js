/* =============================================================================
   site.js — Edward Turk portfolio
   Behavior contract: plans/website-02-design.md §19. No framework, no build step.

   The site is FULLY USABLE WITH JS OFF. This file only enhances:
     1. Boot on DOMContentLoaded (.js class + stored theme are set pre-paint by the
        inline head script in §6 — not here, so there is no theme flash).
     2. Hamburger: .nav-menu.is-open + aria-expanded; focus first link on open;
        close on link click / outside click / Escape (Escape returns focus to the
        toggle); reset at >= 48rem.
     3. Active nav on the homepage only, via IntersectionObserver.
     4. Theme toggle: effective theme -> aria-pressed + glyph; flip + persist on click.
     5. Video facade: intercept .video__play and swap in the nocookie iframe.
     6. Footer year for every [data-year].
     7. PDF lightbox: intercept .pdf-trigger and open its href in a modal iframe
        (Escape closes, Tab is trapped, focus returns to the trigger). With JS
        off the trigger stays an ordinary link straight to the PDF.

   Every handler feature-detects and no-ops when its target is absent.
   ============================================================================= */
(function () {
  'use strict';

  var NAV_BREAKPOINT_PX = 768; /* 48rem at the 16px root default */

  /* --- small helpers ------------------------------------------------------ */
  function $(sel, root) { return (root || document).querySelector(sel); }
  function $$(sel, root) { return Array.prototype.slice.call((root || document).querySelectorAll(sel)); }
  function store(key, val) { try { localStorage.setItem(key, val); } catch (e) { /* private mode */ } }
  function read(key) { try { return localStorage.getItem(key); } catch (e) { return null; } }

  /* ---------------------------------------------------------------------- */
  /* 2. Hamburger                                                            */
  /* ---------------------------------------------------------------------- */
  function initNav() {
    var toggle = $('.nav-toggle');
    var menu = $('#nav-menu');
    if (!toggle || !menu) return;

    function isOpen() { return menu.classList.contains('is-open'); }

    function open() {
      menu.classList.add('is-open');
      toggle.setAttribute('aria-expanded', 'true');
      var first = $('a', menu);
      if (first) first.focus();
    }

    function close(returnFocus) {
      if (!isOpen()) return;
      menu.classList.remove('is-open');
      toggle.setAttribute('aria-expanded', 'false');
      if (returnFocus) toggle.focus();
    }

    toggle.addEventListener('click', function () {
      if (isOpen()) { close(false); } else { open(); }
    });

    /* close on link click (in-page anchors would otherwise leave the menu open) */
    menu.addEventListener('click', function (e) {
      if (e.target.closest && e.target.closest('a')) close(false);
    });

    /* close on outside click */
    document.addEventListener('click', function (e) {
      if (!isOpen()) return;
      if (menu.contains(e.target) || toggle.contains(e.target)) return;
      close(false);
    });

    /* Escape closes AND returns focus to the toggle (§19.2) */
    document.addEventListener('keydown', function (e) {
      if (e.key === 'Escape' || e.key === 'Esc') close(true);
    });

    /* resize past the collapse boundary resets the menu */
    window.addEventListener('resize', function () {
      if (window.innerWidth >= NAV_BREAKPOINT_PX) {
        menu.classList.remove('is-open');
        toggle.setAttribute('aria-expanded', 'false');
      }
    });
  }

  /* ---------------------------------------------------------------------- */
  /* 3. Active nav — homepage only                                           */
  /* ---------------------------------------------------------------------- */
  function initActiveNav() {
    if (!('IntersectionObserver' in window)) return;

    /* Project pages set .is-active on Work statically; JS must do nothing there. */
    var path = window.location.pathname;
    var isHome = path === '/' || path === '' || path === '/index.html';
    if (!isHome) return;

    var ids = ['hero', 'work', 'about', 'contact']; /* #thread removed in 2026-09-01 homepage rework */
    var sections = ids
      .map(function (id) { return document.getElementById(id); })
      .filter(Boolean);
    if (!sections.length) return;

    function linkFor(id) { return $('.nav-menu a[href="/#' + id + '"]'); }

    var observer = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (!entry.isIntersecting) return;
        $$('.nav-menu a').forEach(function (a) { a.classList.remove('is-active'); });
        var link = linkFor(entry.target.id);
        if (link) link.classList.add('is-active');
      });
    }, { rootMargin: '-45% 0px -45% 0px', threshold: 0 });

    sections.forEach(function (s) { observer.observe(s); });
  }

  /* ---------------------------------------------------------------------- */
  /* 4. Theme toggle                                                         */
  /* ---------------------------------------------------------------------- */
  function initTheme() {
    var btn = $('.theme-toggle');
    if (!btn) return;
    var icon = $('.theme-toggle__icon', btn);

    function effectiveTheme() {
      var attr = document.documentElement.getAttribute('data-theme');
      if (attr === 'dark' || attr === 'light') return attr;
      if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) return 'dark';
      return 'light';
    }

    function paint(theme) {
      var dark = theme === 'dark';
      btn.setAttribute('aria-pressed', dark ? 'true' : 'false');
      /* ☀ when dark is active, ☾ when light. The aria-label stays the stable
         "Dark theme" — never rewritten to an action (§6 theme-toggle contract). */
      if (icon) icon.textContent = dark ? '☀' : '☾';
    }

    paint(effectiveTheme());

    btn.addEventListener('click', function () {
      var next = effectiveTheme() === 'dark' ? 'light' : 'dark';
      document.documentElement.setAttribute('data-theme', next);
      store('theme', next);
      paint(next);
    });

    /* Follow the system only while no explicit choice is stored. */
    if (window.matchMedia) {
      var mq = window.matchMedia('(prefers-color-scheme: dark)');
      var onChange = function () { if (!read('theme')) paint(effectiveTheme()); };
      if (mq.addEventListener) { mq.addEventListener('change', onChange); }
      else if (mq.addListener) { mq.addListener(onChange); }
    }
  }

  /* ---------------------------------------------------------------------- */
  /* 5. Video facade — click-to-load nocookie iframe                         */
  /* ---------------------------------------------------------------------- */
  function initVideo() {
    $$('.video__play').forEach(function (anchor) {
      anchor.addEventListener('click', function (e) {
        var id = anchor.getAttribute('data-video-id');
        if (!id) return; /* no id -> leave the plain link working */
        e.preventDefault();

        var frame = document.createElement('iframe');
        frame.src = 'https://www.youtube-nocookie.com/embed/' + encodeURIComponent(id) + '?autoplay=1';
        frame.title = anchor.getAttribute('aria-label') || 'Video';
        frame.setAttribute('loading', 'lazy');
        frame.setAttribute('allow', 'autoplay; fullscreen; picture-in-picture; encrypted-media');
        frame.setAttribute('allowfullscreen', '');
        frame.setAttribute('referrerpolicy', 'strict-origin-when-cross-origin');

        anchor.replaceWith(frame);
        frame.focus();
      });
    });
  }

  /* ---------------------------------------------------------------------- */
  /* 7. PDF lightbox — .pdf-trigger opens its own href in a modal viewer      */
  /*    Same contract as the video facade (§5): the markup ships as a REAL    */
  /*    link to the PDF, so with JS off the click just opens the file. This   */
  /*    only intercepts the click and builds the dialog; nothing here exists  */
  /*    in the HTML, so there is no dead dialog when JS is absent.            */
  /* ---------------------------------------------------------------------- */
  function initPdfLightbox() {
    var triggers = $$('.pdf-trigger');
    if (!triggers.length) return;

    var openState = null; /* { overlay, trigger, onKeydown } while open */

    function focusables(root) {
      return $$('a[href],button:not([disabled]),iframe,[tabindex]:not([tabindex="-1"])', root);
    }

    function close() {
      if (!openState) return;
      var s = openState;
      openState = null;
      document.removeEventListener('keydown', s.onKeydown, true);
      document.body.classList.remove('has-lightbox');
      s.overlay.remove();
      s.trigger.focus();
    }

    function open(trigger) {
      if (openState) return;

      var href = trigger.getAttribute('href');
      if (!href) return; /* no target -> leave the plain link working */
      var title = trigger.getAttribute('data-pdf-title') || 'Document';
      var titleId = 'pdf-lightbox-title';

      var overlay = document.createElement('div');
      overlay.className = 'pdf-lightbox';
      overlay.setAttribute('role', 'dialog');
      overlay.setAttribute('aria-modal', 'true');
      overlay.setAttribute('aria-labelledby', titleId);

      var panel = document.createElement('div');
      panel.className = 'pdf-lightbox__panel';

      var bar = document.createElement('div');
      bar.className = 'pdf-lightbox__bar';

      var heading = document.createElement('p');
      heading.className = 'pdf-lightbox__title';
      heading.id = titleId;
      heading.textContent = title;

      var actions = document.createElement('div');
      actions.className = 'pdf-lightbox__actions';

      /* Escape hatch for browsers with no inline PDF viewer (most mobile). */
      var openLink = document.createElement('a');
      openLink.className = 'btn btn--ghost btn--sm';
      openLink.href = href;
      openLink.target = '_blank';
      openLink.rel = 'noopener';
      openLink.textContent = 'Open in a new tab';

      var closeBtn = document.createElement('button');
      closeBtn.className = 'btn btn--ghost btn--sm';
      closeBtn.type = 'button';
      closeBtn.textContent = 'Close';

      var frame = document.createElement('iframe');
      frame.className = 'pdf-lightbox__frame';
      frame.src = href;
      frame.title = title;

      actions.appendChild(openLink);
      actions.appendChild(closeBtn);
      bar.appendChild(heading);
      bar.appendChild(actions);
      panel.appendChild(bar);
      panel.appendChild(frame);
      overlay.appendChild(panel);

      closeBtn.addEventListener('click', close);
      /* click on the backdrop, never on the panel */
      overlay.addEventListener('click', function (e) { if (e.target === overlay) close(); });

      /* Escape closes; Tab is trapped inside the panel. Capture phase so the
         hamburger's own Escape handler never sees this one. */
      function onKeydown(e) {
        if (e.key === 'Escape' || e.key === 'Esc') {
          e.stopPropagation();
          e.preventDefault();
          close();
          return;
        }
        if (e.key !== 'Tab') return;
        var items = focusables(panel);
        if (!items.length) return;
        var first = items[0];
        var last = items[items.length - 1];
        if (e.shiftKey && (document.activeElement === first || !panel.contains(document.activeElement))) {
          e.preventDefault();
          last.focus();
        } else if (!e.shiftKey && document.activeElement === last) {
          e.preventDefault();
          first.focus();
        }
      }

      document.body.classList.add('has-lightbox');
      document.body.appendChild(overlay);
      document.addEventListener('keydown', onKeydown, true);
      openState = { overlay: overlay, trigger: trigger, onKeydown: onKeydown };
      closeBtn.focus();
    }

    triggers.forEach(function (trigger) {
      trigger.addEventListener('click', function (e) {
        /* let modified clicks (new tab/window, download) behave normally */
        if (e.metaKey || e.ctrlKey || e.shiftKey || e.altKey || e.button !== 0) return;
        e.preventDefault();
        open(trigger);
      });
    });
  }

  /* ---------------------------------------------------------------------- */
  /* 6. Footer year                                                          */
  /* ---------------------------------------------------------------------- */
  function initYear() {
    var year = String(new Date().getFullYear());
    $$('[data-year]').forEach(function (el) { el.textContent = year; });
  }

  /* ---------------------------------------------------------------------- */
  /* 1. Boot                                                                 */
  /* ---------------------------------------------------------------------- */
  function boot() {
    initNav();
    initActiveNav();
    initTheme();
    initVideo();
    initPdfLightbox();
    initYear();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot);
  } else {
    boot();
  }
})();
