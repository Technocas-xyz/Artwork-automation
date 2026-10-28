/* ============================================================================
   shell.js — product-shell chrome (dark navy sidebar, top bar, page header).
   VISUAL/STRUCTURAL ADDITION ONLY.

   Rules honoured:
   - Does NOT touch app.py, src/, or the existing inline <script> / its logic.
   - Renames/removes NO existing id or data attribute, changes NO API call.
   - The existing <header>, its .wf-tab buttons, and the .columns block are
     MOVED (re-parented) into the new layout — moving a node preserves all its
     event listeners, so the original tab-switching logic keeps working.
   - Workflow switching is NOT reimplemented: sidebar nav items dispatch a real
     .click() on the matching existing .wf-tab. A MutationObserver mirrors the
     real tabs' .active state back into the new UI.
   ============================================================================ */
(function () {
  "use strict";

  function ready(fn) {
    if (document.readyState !== "loading") fn();
    else document.addEventListener("DOMContentLoaded", fn);
  }

  var APP_VERSION = "v1.0.0";   // decorative footer label

  // Tabler icon glyphs (webfont classes). Icon names map to ti-<name>.
  var WF_ICONS = {
    text: "typography",
    textimage: "photo-edit",
    mockup: "scissors",
    artwork: "wand",
    custom: "adjustments",
    identify: "list-search",
    printready: "rosette-discount-check",
    colorway: "palette",
    geometry: "ruler-measure",
    similarity: "git-compare",
    nextcloud: "cloud"
  };
  // Workflows that belong in the "CREATE ARTWORK" group (everything that
  // generates/analyses art). nextcloud is shown separately below.
  var CREATE_WORKFLOWS = ["text", "textimage", "mockup", "artwork", "custom",
                          "identify", "printready", "colorway", "geometry", "similarity"];

  var PAGE_DESC = {
    text: "Generate finished artwork from wording the client supplies.",
    textimage: "The client supplies an image — use it as an element, as style, or replace its wording.",
    mockup: "Pull individual designs out of a mockup or contact sheet.",
    artwork: "Batch clean-up and regeneration of supplied artwork.",
    custom: "Apply a chosen sequence of operations to one artwork.",
    identify: "List the objects in an artwork, then regenerate it with your changes applied.",
    printready: "Check whether an artwork is ready for DTF printing, with a score and a printer\u2019s assessment.",
    colorway: "Find the garment colours an artwork suits, preview it on them, and adapt it where needed.",
    geometry: "Measure an artwork's geometry, regenerate it at the target DTF size, and measure the result.",
    similarity: "Upload two images and get their similarity \u2014 shape, edges, colour and detail \u2014 with an overall score.",
    nextcloud: "Browse the Leads 2.0 vault and send artwork to any workflow."
  };
  var PAGE_TITLE = {
    text: "Text",
    textimage: "Text + Image",
    mockup: "Artwork Extraction",
    artwork: "Artwork Generation",
    custom: "Custom Operation",
    identify: "Artwork Identification",
    printready: "Print Ready QA",
    colorway: "Colorways",
    geometry: "Pixel Geometry",
    similarity: "Similarity Check",
    nextcloud: "Nextcloud Vault"
  };

  ready(function () {
    var header = document.querySelector("header");
    var columns = document.querySelector(".columns");
    var realTabs = Array.prototype.slice.call(document.querySelectorAll(".wf-tab"));
    if (!header || !columns || !realTabs.length) return; // markup not as expected — bail safely

    var tabByWf = {};
    realTabs.forEach(function (t) { tabByWf[t.dataset.wf] = t; });

    // ---- Shell scaffold ----
    var shell = document.createElement("div");
    shell.className = "app-shell";

    var sidebar = document.createElement("aside");
    sidebar.className = "app-sidebar";

    var main = document.createElement("div");
    main.className = "app-main";

    // ======================= SIDEBAR =======================================
    sidebar.innerHTML =
      '<div class="sb-brand">' +
        '<span class="sb-logo">A</span>' +
        '<span class="sb-brand-text">' +
          '<div class="sb-brand-name">Artwork Studio</div>' +
          '<div class="sb-brand-sub">AI Powered Designs</div>' +
        '</span>' +
      '</div>' +
      '<nav class="sb-nav" id="sbNav"></nav>' +
      '<div class="sb-foot">' +
        '<span class="sb-version">' + APP_VERSION + '</span>' +
        '<button class="sb-collapse" id="sbCollapse" type="button" title="Collapse sidebar">' +
          '<i class="ti ti-chevrons-left"></i></button>' +
      '</div>';

    var nav = sidebar.querySelector("#sbNav");

    function navItem(icon, label, opts) {
      opts = opts || {};
      var el = document.createElement("button");
      el.type = "button";
      el.className = "sb-item" + (opts.active ? " active" : "");
      el.innerHTML = '<i class="ti ti-' + icon + '"></i><span>' + label + '</span>';
      el.title = label;
      return el;
    }
    function sectionLabel(text) {
      var d = document.createElement("div");
      d.className = "sb-section";
      d.textContent = text;
      return d;
    }
    function clearPlaceholderActive() {
      nav.querySelectorAll(".sb-item.active").forEach(function (x) {
        if (!x.dataset.wfLink) x.classList.remove("active");
      });
    }

    // A sidebar item that opens a real .wf-tab. Keyed by workflow via
    // data-wfLink so syncActive() can mirror the active state onto it.
    var wfNavItems = {};     // wf -> primary sidebar item (for title/mirror)
    function wfLink(wf, icon, label) {
      var it = navItem(icon, label);
      var tab = tabByWf[wf];
      it.dataset.wfLink = wf;
      it.addEventListener("click", function () {
        if (!tab || tab.disabled) return;
        tab.click();                 // reuse existing switching behaviour
      });
      return it;
    }

    // --- Dashboard (decorative landing highlight) ---
    var dashItem = navItem("layout-dashboard", "Dashboard");
    dashItem.addEventListener("click", function () {
      clearPlaceholderActive();
      dashItem.classList.add("active");
    });
    nav.appendChild(dashItem);

    // --- Nextcloud (main starting point) + Artwork Vault ---
    // Both open the single real nextcloud tab. "Nextcloud" is the browse/import
    // starting point; "Artwork Vault" is our saved outputs — same backend tab
    // today, kept as distinct entries per the information architecture. They are
    // mirrored together when that tab is active.
    // Both "Nextcloud" and "Artwork Vault" open the same nextcloud tab, but only
    // ONE sidebar item may look active at a time. ncActiveItem tracks the one the
    // operator actually clicked (default: Nextcloud); syncActive highlights only
    // that one and clears the other.
    var ncTabItems = [];
    var ncActiveItem = null;
    if (tabByWf.nextcloud) {
      var ncItem = wfLink("nextcloud", "cloud", "Nextcloud");
      var vaultItem = wfLink("nextcloud", "photo", "Artwork Vault");
      wfNavItems.nextcloud = ncItem;      // the one that drives the page title
      ncTabItems.push(ncItem, vaultItem);
      ncActiveItem = ncItem;
      ncItem.addEventListener("click", function () { ncActiveItem = ncItem; });
      vaultItem.addEventListener("click", function () { ncActiveItem = vaultItem; });
      nav.appendChild(ncItem);
      nav.appendChild(vaultItem);
    }

    // --- Workflows ▾ (collapsible group) ---
    var WF_OPEN_KEY = "sbWorkflowsOpen";
    function wfGroupStored() {
      try { return localStorage.getItem(WF_OPEN_KEY); } catch (e) { return null; }
    }
    var wfGroupHead = document.createElement("button");
    wfGroupHead.type = "button";
    wfGroupHead.className = "sb-group-head";
    wfGroupHead.innerHTML = '<i class="ti ti-stack-2"></i><span>Workflows</span>' +
      '<i class="ti ti-chevron-down sb-caret"></i>';
    var wfGroupBody = document.createElement("div");
    wfGroupBody.className = "sb-group-body";
    nav.appendChild(wfGroupHead);
    nav.appendChild(wfGroupBody);

    CREATE_WORKFLOWS.forEach(function (wf) {
      var tab = tabByWf[wf];
      if (!tab) return;
      var it = wfLink(wf, WF_ICONS[wf] || "square", PAGE_TITLE[wf] || tab.textContent.trim());
      it.classList.add("sb-sub");
      wfNavItems[wf] = it;
      wfGroupBody.appendChild(it);
    });

    function setWfGroupOpen(open, persist) {
      wfGroupHead.classList.toggle("open", open);
      wfGroupBody.classList.toggle("open", open);
      if (persist) { try { localStorage.setItem(WF_OPEN_KEY, open ? "1" : "0"); } catch (e) {} }
    }
    wfGroupHead.addEventListener("click", function () {
      setWfGroupOpen(!wfGroupBody.classList.contains("open"), true);
    });
    // Initial state: remembered preference, else closed (opened on demand when a
    // workflow page is active — see syncActive).
    setWfGroupOpen(wfGroupStored() === "1", false);

    // --- Settings (bottom of the nav) → admin user modal (modal gates by role) ---
    nav.appendChild(sectionLabel(""));   // spacer pushes Settings down a touch
    var usersItem = navItem("settings", "Settings");
    usersItem.addEventListener("click", function () {
      clearPlaceholderActive();
      usersItem.classList.add("active");
      if (typeof window.openAdminUsers === "function") window.openAdminUsers();
    });
    nav.appendChild(usersItem);

    // Collapse toggle.
    sidebar.querySelector("#sbCollapse").addEventListener("click", function () {
      shell.classList.toggle("sb-collapsed");
    });

    // ======================= TOP BAR =======================================
    var topbar = document.createElement("div");
    topbar.className = "app-topbar";

    var search = document.createElement("div");
    search.className = "topbar-search";
    search.innerHTML = '<i class="ti ti-search"></i>' +
      '<input type="text" placeholder="Search artworks, jobs, or tools…" aria-label="Search">';

    var userSlot = document.createElement("div");
    userSlot.className = "app-user-slot";
    userSlot.id = "shellUserSlot";

    // "How it works?" moved into the top bar (was in the page header) to keep
    // the header a single compact line.
    var howBtn = document.createElement("button");
    howBtn.type = "button";
    howBtn.className = "app-howto";
    howBtn.id = "shellHowTo";
    howBtn.innerHTML = '<i class="ti ti-help-circle"></i> How it works?';
    howBtn.addEventListener("click", function () { openAgentPanel(); });

    // Agent status pill (fed by /api/status polling below).
    var agentPill = document.createElement("div");
    agentPill.className = "agent-pill agent-pill-off";
    agentPill.id = "shellAgentPill";
    agentPill.innerHTML = '<span class="agent-dot"></span><span class="agent-text">Checking agent…</span>';

    // Download-agent button.
    var dlBtn = document.createElement("button");
    dlBtn.type = "button";
    dlBtn.className = "agent-dl-btn";
    dlBtn.id = "shellAgentDownload";
    dlBtn.innerHTML = '<i class="ti ti-download"></i> Download agent';
    dlBtn.addEventListener("click", function () { openAgentPanel(); });

    // Notification bell (decorative) with an unread badge.
    var bell = document.createElement("button");
    bell.type = "button";
    bell.className = "topbar-bell";
    bell.setAttribute("aria-label", "Notifications");
    bell.innerHTML = '<i class="ti ti-bell"></i><span class="bell-badge"></span>';

    // Avatar + name + chevron, with a dropdown holding the existing
    // #userInfo / session indicator (which carry the real logout etc.).
    var userMenu = document.createElement("div");
    userMenu.className = "app-user-menu";
    userMenu.id = "shellUserMenu";
    userMenu.setAttribute("role", "button");
    userMenu.setAttribute("tabindex", "0");
    userMenu.innerHTML =
      '<span class="app-avatar" id="shellAvatar">AD</span>' +
      '<span class="app-user-name" id="shellUserName">admin</span>' +
      '<i class="ti ti-chevron-down app-user-caret"></i>';
    var dropdown = document.createElement("div");
    dropdown.className = "app-user-dropdown";
    dropdown.id = "shellUserDropdown";
    dropdown.innerHTML = '<div class="dd-name" id="shellDdName">Signed in</div>' +
      '<div class="dd-sub" id="shellDdSub">Decoinks</div>' +
      '<div class="dd-slot" id="shellDdSlot"></div>';
    userMenu.appendChild(dropdown);
    userMenu.addEventListener("click", function (e) {
      e.stopPropagation();
      dropdown.classList.toggle("open");
    });
    document.addEventListener("click", function () { dropdown.classList.remove("open"); });

    userSlot.appendChild(howBtn);
    userSlot.appendChild(agentPill);
    userSlot.appendChild(dlBtn);
    userSlot.appendChild(bell);
    userSlot.appendChild(userMenu);

    topbar.appendChild(search);
    topbar.appendChild(userSlot);

    // ======================= PAGE HEADER ===================================
    var pagehead = document.createElement("div");
    pagehead.className = "app-pagehead";
    // ONE-LINE header: title + subtitle on the same row (no breadcrumb). The
    // "How it works?" control lives in the top bar now (added above), freeing
    // vertical space so a workflow fits a 1366×768 screen without scrolling.
    pagehead.innerHTML =
      '<div class="app-page-title" id="shellPageTitle"></div>' +
      '<div class="app-page-desc" id="shellPageDesc"></div>';

    // ---- Assemble: move existing header + columns into the main area ----
    header.parentNode.insertBefore(shell, header);
    shell.appendChild(sidebar);
    shell.appendChild(main);
    main.appendChild(topbar);
    main.appendChild(pagehead);
    main.appendChild(header);    // header kept (hidden via CSS) as tab source-of-truth
    main.appendChild(columns);   // columns + all col contents unchanged

    // ---- ONE left scroll container ----
    // #col1 (inputs) and #col2 (options + action button) used to be two separate
    // grid rows that scrolled independently and overlapped. Wrap them in a single
    // .col-left once at startup — their ids are untouched, so every renderer that
    // writes into #col1 / #col2 keeps working. Inside the wrapper they flow
    // normally one after the other and only the wrapper scrolls; #col3 (Results)
    // sits beside it. Pixel Geometry (geo-full) hides this wrapper and uses #col3
    // full-width, so it is unaffected.
    (function wrapLeftColumn() {
      var c1 = document.getElementById("col1");
      var c2 = document.getElementById("col2");
      var c3 = document.getElementById("col3");
      if (!c1 || !c2 || !columns) return;
      var left = document.createElement("div");
      left.className = "col-left";
      left.id = "colLeft";
      columns.insertBefore(left, c1);   // put the wrapper where col1 was
      left.appendChild(c1);
      left.appendChild(c2);
      if (c3) columns.appendChild(c3);  // keep Results after the wrapper
    })();

    // Relocate the existing user + session indicators into the dropdown slot.
    var userInfo = document.getElementById("userInfo");
    var sessionIndicator = document.getElementById("sessionIndicator");
    var ddSlot = document.getElementById("shellDdSlot");
    if (sessionIndicator && ddSlot) ddSlot.appendChild(sessionIndicator);
    if (userInfo && ddSlot) ddSlot.appendChild(userInfo);

    // ---- Mirror active state from the real tabs (source of truth) ----
    function activeIndex() {
      for (var i = 0; i < realTabs.length; i++) {
        if (realTabs[i].classList.contains("active")) return i;
      }
      return 0;
    }
    function syncActive() {
      var idx = activeIndex();
      var wf = realTabs[idx] ? realTabs[idx].dataset.wf : "text";
      Object.keys(wfNavItems).forEach(function (k) {
        wfNavItems[k].classList.toggle("active", k === wf);
      });
      // Nextcloud is reached from TWO sidebar items; highlight ONLY the one the
      // operator clicked (ncActiveItem) so a single item is ever active.
      ncTabItems.forEach(function (it) {
        it.classList.toggle("active", wf === "nextcloud" && it === ncActiveItem);
      });
      // A real workflow is active → clear the decorative Dashboard highlight.
      if (wfNavItems[wf] || wf === "nextcloud") {
        dashItem.classList.remove("active");
      }
      var t = document.getElementById("shellPageTitle");
      var d = document.getElementById("shellPageDesc");
      var c = document.getElementById("shellCrumbPage");
      if (t) t.textContent = PAGE_TITLE[wf] || "";
      if (d) d.textContent = PAGE_DESC[wf] || "";
      if (c) c.textContent = PAGE_TITLE[wf] || "Workflow";
    }
    var mo = new MutationObserver(syncActive);
    realTabs.forEach(function (tab) {
      mo.observe(tab, { attributes: true, attributeFilter: ["class"] });
    });
    syncActive();

    // Fill avatar initials, top-bar name and dropdown name from #userInfo text.
    function _initials(name) {
      var parts = name.split(/\s+/).filter(Boolean);
      if (!parts.length) return "AD";
      if (parts.length === 1) return parts[0].slice(0, 2).toUpperCase();
      return (parts[0].charAt(0) + parts[parts.length - 1].charAt(0)).toUpperCase();
    }
    function setUserBits() {
      var ui = document.getElementById("userInfo");
      var name = ui ? (ui.textContent || "").replace(/\s+/g, " ").trim() : "";
      // #userInfo often holds extra chrome; take the first non-empty token line.
      var display = name ? name.split("•")[0].trim() : "";
      var avatarEl = document.getElementById("shellAvatar");
      var nameEl = document.getElementById("shellUserName");
      var ddName = document.getElementById("shellDdName");
      if (avatarEl) avatarEl.textContent = display ? _initials(display) : "AD";
      if (nameEl && display) nameEl.textContent = display.split(" ")[0] || display;
      if (ddName && display) ddName.textContent = display;
    }
    setUserBits();
    if (userInfo) {
      var uo = new MutationObserver(setUserBits);
      uo.observe(userInfo, { childList: true, subtree: true, characterData: true });
    }

    // --- Single agent-setup modal entry point (defined in index.html) ---
    function openAgentPanel() {
      if (typeof window.openAgentSetup === "function") window.openAgentSetup();
    }
    window.openAgentPanel = openAgentPanel;

    // Prominent no-agent banner above the columns.
    var banner = document.createElement("div");
    banner.className = "agent-banner";
    banner.id = "agentBanner";
    banner.style.display = "none";
    banner.innerHTML = '<span>Download and run the agent to start generating.</span>' +
      '<button type="button" class="agent-banner-btn" id="agentBannerBtn">Download agent</button>';
    if (columns && columns.parentNode) columns.parentNode.insertBefore(banner, columns);
    document.getElementById("agentBannerBtn").addEventListener("click", function () { openAgentPanel(); });

    // ======================= REUSABLE STEPPER ==============================
    // window.buildStepper([{name,desc}...], currentIndex) -> HTML string of a
    // numbered horizontal stepper (steps before current = done). Pure helper;
    // nothing calls it from workflow logic, so it changes no behaviour.
    window.buildStepper = function (steps, current) {
      steps = steps || []; current = current || 0;
      var h = '<div class="stepper">';
      for (var i = 0; i < steps.length; i++) {
        if (i) h += '<div class="st-sep"></div>';
        var cls = i < current ? "done" : (i === current ? "current" : "");
        var s = steps[i] || {};
        h += '<div class="step ' + cls + '"><div class="st-num">' + (i + 1) + '</div>' +
          '<div class="st-txt"><div class="st-name">' + (s.name || "") + '</div>' +
          (s.desc ? '<div class="st-desc">' + s.desc + '</div>' : '') + '</div></div>';
      }
      return h + '</div>';
    };

    // --- Agent connection status: poll /api/status every 5s ---
    function _agoText(epochSec) {
      if (!epochSec) return "never seen";
      var s = Math.max(0, Math.round(Date.now() / 1000 - epochSec));
      if (s < 10) return "just now";
      if (s < 60) return s + "s ago";
      if (s < 3600) return Math.round(s / 60) + "m ago";
      if (s < 86400) return Math.round(s / 3600) + "h ago";
      return Math.round(s / 86400) + "d ago";
    }
    function pollAgentStatus() {
      fetch("/api/status", { headers: { "Accept": "application/json" } })
        .then(function (r) { return r.ok ? r.json() : null; })
        .then(function (s) {
          if (!s) return;
          var pill = document.getElementById("shellAgentPill");
          var dl = document.getElementById("shellAgentDownload");
          var bnr = document.getElementById("agentBanner");
          if (!pill) return;
          var textEl = pill.querySelector(".agent-text");
          if (s.agent_connected) {
            pill.className = "agent-pill agent-pill-on";
            var nm = s.agent_name || "agent";
            var ver = s.agent_version ? "  v" + s.agent_version : "";
            var seenTxt = "";
            var la = (s.agents || []).filter(function (a) { return a.name === nm; })[0];
            if (la && la.last_seen) seenTxt = "  (" + _agoText(la.last_seen) + ")";
            textEl.textContent = "Agent connected — " + nm + ver + seenTxt;
            if (dl) dl.classList.remove("agent-dl-prominent");
            if (bnr) bnr.style.display = "none";
          } else {
            pill.className = "agent-pill agent-pill-off";
            textEl.textContent = "No agent running";
            if (dl) dl.classList.add("agent-dl-prominent");
            if (bnr) bnr.style.display = "flex";
          }
        })
        .catch(function () {});
    }
    pollAgentStatus();
    setInterval(pollAgentStatus, 5000);

    // Show which agent picked up each job, on job cards.
    function annotateJobAgents() {
      fetch("/api/jobs", { headers: { "Accept": "application/json" } })
        .then(function (r) { return r.ok ? r.json() : null; })
        .then(function (list) {
          if (!list) return;
          list.forEach(function (j) {
            if (!j.claimed_by_name) return;
            var host = document.querySelector('[data-job="' + j.id + '"]');
            if (!host) return;
            var card = host.closest(".history-item, .stage-result, #resultArea") || host.parentElement;
            if (!card || card.querySelector(".agent-badge")) return;
            var badge = document.createElement("div");
            badge.className = "agent-badge";
            badge.textContent = "Picked up by " + j.claimed_by_name;
            card.appendChild(badge);
          });
        })
        .catch(function () {});
    }
    setInterval(annotateJobAgents, 5000);
  });
})();
