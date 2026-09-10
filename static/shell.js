/* ============================================================================
   shell.js — product-shell chrome (sidebar, top bar, action bar).
   VISUAL/STRUCTURAL ADDITION ONLY.

   Rules honoured:
   - Does NOT touch app.py, src/, or the existing inline <script>.
   - Renames/removes NO existing id or data attribute.
   - The existing <header>, its .wf-tab buttons, and the .columns block are
     MOVED (re-parented) into the new layout — moving a node preserves all its
     event listeners, so the original tab-switching logic keeps working.
   - Workflow switching is NOT reimplemented: sidebar nav items dispatch a
     real .click() on the matching existing .wf-tab. A MutationObserver mirrors
     the real tabs' .active state back into the new UI.
   ============================================================================ */
(function () {
  "use strict";

  function ready(fn) {
    if (document.readyState !== "loading") fn();
    else document.addEventListener("DOMContentLoaded", fn);
  }

  // Tabler icon glyphs (webfont classes). Icon names map to ti-<name>.
  var NAV_TOP = [
    { icon: "layout-dashboard", label: "Dashboard" },
    { icon: "folders", label: "Projects" },
    { icon: "photo", label: "Artwork Vault" }
  ];
  var WF_ICONS = {
    text: "typography",
    mockup: "scissors",
    artwork: "wand",
    custom: "adjustments"
  };
  var PAGE_DESC = {
    text: "Generate finished artwork from a line of design text.",
    mockup: "Pull individual designs out of a mockup or contact sheet.",
    artwork: "Batch clean-up and regeneration of supplied artwork.",
    custom: "Apply a chosen sequence of operations to one artwork."
  };
  var PAGE_TITLE = {
    text: "Text",
    mockup: "Artwork Extraction",
    artwork: "Artwork Generation",
    custom: "Custom Operation"
  };

  ready(function () {
    var header = document.querySelector("header");
    var columns = document.querySelector(".columns");
    var realTabs = Array.prototype.slice.call(document.querySelectorAll(".wf-tab"));
    if (!header || !columns || !realTabs.length) return; // markup not as expected — bail safely

    // --- Build the shell scaffold ---
    var shell = document.createElement("div");
    shell.className = "app-shell";

    var sidebar = document.createElement("aside");
    sidebar.className = "app-sidebar";

    var main = document.createElement("div");
    main.className = "app-main";

    // Top bar: page title + description on the left, agent status + user on the
    // right. The numbered workflow breadcrumb was removed — the sidebar is the
    // single place a workflow is chosen, and the numbering wrongly implied the
    // workflows are sequential steps.
    var topbar = document.createElement("div");
    topbar.className = "app-topbar";

    var titleWrap = document.createElement("div");
    titleWrap.className = "app-title-wrap";
    titleWrap.innerHTML = '<div class="app-page-title" id="shellPageTitle"></div>' +
      '<div class="app-page-desc" id="shellPageDesc"></div>';

    // User block moved visually to the top bar (clone-follow, see below).
    var userSlot = document.createElement("div");
    userSlot.className = "app-user-slot";
    userSlot.id = "shellUserSlot";

    // Agent connection status pill (fed by /api/status polling below).
    var agentPill = document.createElement("div");
    agentPill.className = "agent-pill agent-pill-off";
    agentPill.id = "shellAgentPill";
    agentPill.innerHTML = '<span class="agent-dot"></span><span class="agent-text">Checking agent…</span>';

    // "Download agent" button — opens the setup panel.
    var dlBtn = document.createElement("button");
    dlBtn.type = "button";
    dlBtn.className = "agent-dl-btn";
    dlBtn.id = "shellAgentDownload";
    dlBtn.innerHTML = '<i class="ti ti-download"></i> Download agent';
    dlBtn.addEventListener("click", function () { openAgentPanel(); });

    topbar.appendChild(titleWrap);
    userSlot.appendChild(agentPill);
    userSlot.appendChild(dlBtn);
    topbar.appendChild(userSlot);

    // --- Sidebar contents ---
    sidebar.innerHTML =
      '<div class="sb-brand"><span class="sb-logo">A</span><span class="sb-brand-name">Artwork Studio</span></div>' +
      '<button class="sb-newproject" type="button"><i class="ti ti-plus"></i> New Project</button>' +
      '<nav class="sb-nav" id="sbNav"></nav>' +
      '<div class="sb-help"><div class="sb-help-title">Need help?</div>' +
      '<div class="sb-help-body">Check the workflow guide or reach out.</div>' +
      '<a class="sb-help-link" href="#" onclick="return false">View guide</a></div>';

    var nav = sidebar.querySelector("#sbNav");

    function navItem(icon, label, opts) {
      opts = opts || {};
      var el = document.createElement(opts.href ? "a" : "button");
      el.className = "sb-item" + (opts.active ? " active" : "");
      if (!opts.href) el.type = "button";
      el.innerHTML = '<i class="ti ti-' + icon + '"></i><span>' + label + '</span>';
      return el;
    }
    function sectionLabel(text) {
      var d = document.createElement("div");
      d.className = "sb-section";
      d.textContent = text;
      return d;
    }

    // Top nav (decorative destinations — not wired to backend routes)
    NAV_TOP.forEach(function (n, i) {
      var it = navItem(n.icon, n.label, { active: i === 0 });
      it.addEventListener("click", function () {
        // Visual selection only for these placeholder destinations.
        clearSidebarActive();
        it.classList.add("active");
      });
      nav.appendChild(it);
    });

    nav.appendChild(sectionLabel("Workflow"));

    // Workflow nav items — each dispatches a click on the matching real .wf-tab.
    var wfNavItems = {};
    realTabs.forEach(function (tab) {
      var wf = tab.dataset.wf;
      var it = navItem(WF_ICONS[wf] || "square", tab.textContent.trim());
      it.dataset.wfLink = wf;
      it.addEventListener("click", function () {
        if (tab.disabled) return;
        tab.click(); // reuse existing switching behaviour — do NOT reimplement
      });
      wfNavItems[wf] = it;
      nav.appendChild(it);
    });

    nav.appendChild(sectionLabel("Settings"));
    var settingsItem = navItem("settings", "Settings", {});
    settingsItem.addEventListener("click", function () {
      clearSidebarActive();
      settingsItem.classList.add("active");
    });
    nav.appendChild(settingsItem);

    // "New Project" resets to the first workflow tab.
    var newBtn = sidebar.querySelector(".sb-newproject");
    newBtn.addEventListener("click", function () {
      if (realTabs[0] && !realTabs[0].disabled) realTabs[0].click();
    });

    function clearSidebarActive() {
      nav.querySelectorAll(".sb-item.active").forEach(function (x) {
        // Keep the active workflow item highlighted; only clear placeholder nav.
        if (!x.dataset.wfLink) x.classList.remove("active");
      });
    }

    // --- Assemble: move existing header + columns into the main area ---
    // Insert shell where the header currently is, then re-parent nodes.
    header.parentNode.insertBefore(shell, header);
    shell.appendChild(sidebar);
    shell.appendChild(main);
    main.appendChild(topbar);
    main.appendChild(header);   // header keeps its listeners; hidden via CSS
    main.appendChild(columns);  // columns and all col contents unchanged

    // Relocate the existing user + session indicators into the top bar slot.
    var userInfo = document.getElementById("userInfo");
    var sessionIndicator = document.getElementById("sessionIndicator");
    if (sessionIndicator) userSlot.appendChild(sessionIndicator);
    if (userInfo) {
      var avatar = document.createElement("span");
      avatar.className = "app-avatar";
      avatar.id = "shellAvatar";
      userSlot.appendChild(avatar);
      userSlot.appendChild(userInfo);
    }

    // --- Mirror active state from the real tabs (source of truth) ---
    function activeIndex() {
      for (var i = 0; i < realTabs.length; i++) {
        if (realTabs[i].classList.contains("active")) return i;
      }
      return 0;
    }
    function syncActive() {
      var idx = activeIndex();
      var wf = realTabs[idx] ? realTabs[idx].dataset.wf : "text";

      // Sidebar workflow items — the single place a workflow is highlighted.
      Object.keys(wfNavItems).forEach(function (k) {
        wfNavItems[k].classList.toggle("active", k === wf);
      });

      // Page title + description
      var t = document.getElementById("shellPageTitle");
      var d = document.getElementById("shellPageDesc");
      if (t) t.textContent = PAGE_TITLE[wf] || "";
      if (d) d.textContent = PAGE_DESC[wf] || "";
    }

    // Observe class changes on the real tabs to mirror them.
    var mo = new MutationObserver(syncActive);
    realTabs.forEach(function (tab) {
      mo.observe(tab, { attributes: true, attributeFilter: ["class"] });
    });
    syncActive();

    // Fill avatar initial once userInfo is populated by the existing script.
    var avatarEl = document.getElementById("shellAvatar");
    if (avatarEl) {
      var setInitial = function () {
        var ui = document.getElementById("userInfo");
        var name = ui ? (ui.textContent || "").trim() : "";
        avatarEl.textContent = name ? name.charAt(0).toUpperCase() : "";
      };
      setInitial();
      var uo = new MutationObserver(setInitial);
      if (userInfo) uo.observe(userInfo, { childList: true, subtree: true, characterData: true });
    }

    // --- Agent setup: there is a SINGLE panel now, the "Agent setup" modal
    // defined in index.html (window.openAgentSetup). It asks for the designer's
    // per-machine name, then shows that machine's token, the download link and
    // the three setup steps. This shell no longer builds its own duplicate
    // modal (which called /api/my-agent-token with no name and showed a blank
    // token). All entry points delegate to that one modal.
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

    // --- Agent connection status: poll /api/status every 5s ---
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
            textEl.textContent = "Agent connected — " + nm;
            if (dl) dl.classList.remove("agent-dl-prominent");
            if (bnr) bnr.style.display = "none";
          } else {
            pill.className = "agent-pill agent-pill-off";
            textEl.textContent = "No agent running — start the agent on your PC to generate";
            if (dl) dl.classList.add("agent-dl-prominent");
            if (bnr) bnr.style.display = "flex";
          }
        })
        .catch(function () {});
    }
    pollAgentStatus();
    setInterval(pollAgentStatus, 5000);

    // Show which agent picked up each job on the job cards. The existing render
    // writes job status text; we annotate any element that carries a job id.
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
