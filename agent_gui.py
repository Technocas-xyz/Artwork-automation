"""Small tkinter + tray wrapper around agent.py.

This is a THIN wrapper: it collects settings, drives sign-in, and starts/stops
the agent's existing run loop on a background thread. It does not change the job
loop or any workflow logic — it only calls agent.configure(), agent.register()
and agent.run_loop().

Config (server URL + token) is saved to %APPDATA%\\ArtworkAgent\\config.json so
the designer enters it once.
"""
from __future__ import annotations

import json
import os
import queue
import sys
import threading
import traceback
from pathlib import Path

import tkinter as tk
from tkinter import ttk, messagebox

import agent  # the existing agent, reused unchanged

# Error log + errors ring for the agent window. Guarded so a missing module
# never stops the GUI from launching.
try:
    from src import agent_logging as _agent_logging
except Exception:
    _agent_logging = None

import logging as _logging

# Every user-facing failure (a messagebox, the red status line, a failed
# sign-in / start) is funnelled through here so it ALSO reaches errors.log and
# the Errors box — not just the popup. Throttled like the claim loop: the same
# message in a row is logged once, then counted, so a repeated failure does not
# flood the file. exc_info=True attaches the current traceback when inside an
# except block.
_gui_log_lock = __import__("threading").Lock()
_gui_last_msg = {"text": None}


def _gui_log(message: str, level: int = _logging.ERROR, exc: bool = False) -> None:
    """Log a user-facing GUI failure at WARNING/ERROR (default ERROR) with the
    same 'repeat logged once' throttle used by the claim loop. Never raises."""
    try:
        msg = str(message or "").strip()
        if not msg:
            return
        with _gui_log_lock:
            if _gui_last_msg["text"] == msg:
                return   # identical to the previous message — don't re-log
            _gui_last_msg["text"] = msg
        _logging.getLogger("agent.gui").log(level, "%s", msg, exc_info=exc)
    except Exception:
        pass

# --- Config file in AppData ---
_APPDATA = Path(os.getenv("APPDATA") or (Path.home() / "AppData" / "Roaming"))
_CONFIG_DIR = _APPDATA / "ArtworkAgent"
_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
_CONFIG_FILE = _CONFIG_DIR / "config.json"

DEFAULT_SERVER = os.getenv("SERVER_URL", "http://127.0.0.1:8000")


def load_config() -> dict:
    if _CONFIG_FILE.exists():
        try:
            return json.loads(_CONFIG_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_config(cfg: dict) -> None:
    try:
        _CONFIG_FILE.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    except Exception as exc:
        print(f"[gui] could not save config: {exc}")


# --- Password obfuscation (NOT encryption) ---------------------------------
# The manager accepted storing the password on the designer's machine (see the
# task summary). We at least keep it out of plain sight in the config file with
# a reversible XOR+base64 scramble against a fixed local key. This is
# obfuscation, not security: anyone with the file and this code can recover it.
# It is deliberately never printed or logged.
import base64 as _b64

_OBF_KEY = b"artwork-agent-local-obfuscation-key"


def _xor(data: bytes) -> bytes:
    k = _OBF_KEY
    return bytes(b ^ k[i % len(k)] for i, b in enumerate(data))


def obfuscate_password(plain: str) -> str:
    if not plain:
        return ""
    return _b64.b64encode(_xor(plain.encode("utf-8"))).decode("ascii")


def deobfuscate_password(scrambled: str) -> str:
    if not scrambled:
        return ""
    try:
        return _xor(_b64.b64decode(scrambled.encode("ascii"))).decode("utf-8")
    except Exception:
        return ""


# --- Tray icon drawing (coloured dot) ---
def _make_icon(color: str):
    from PIL import Image, ImageDraw
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse((8, 8, 56, 56), fill=color)
    return img


_TRAY_COLORS = {
    "connected": "#16a34a",     # green
    "running": "#16a34a",       # green (busy but healthy)
    "not_signed_in": "#d97706", # amber
    "error": "#dc2626",         # red
    "stopped": "#9ca3af",       # grey
}


class AgentGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        # Show the running code version in the title so a designer can confirm a
        # fix has landed without opening code_version.txt on their machine.
        self._code_version = getattr(agent, "AGENT_CODE_VERSION", "unknown")
        self.root.title(f"Artwork Agent  v{self._code_version}")
        self.root.geometry("440x440")
        self.root.resizable(False, False)

        cfg = load_config()
        self.server_var = tk.StringVar(value=cfg.get("server_url", DEFAULT_SERVER))
        # Agent now signs in with the same email + password as the website.
        self.email_var = tk.StringVar(value=cfg.get("email", ""))
        self.password_var = tk.StringVar(value=deobfuscate_password(cfg.get("password_obf", "")))
        # The token is obtained at Start by authenticating; not shown or entered.
        self._token = ""

        self._events: queue.Queue = queue.Queue()
        self._stop_event = threading.Event()
        self._agent_id = None
        self._tray = None
        self._state = "stopped"

        # --- Single browser owner ---------------------------------------
        # Playwright's sync API is thread-affine: the context/page can only be
        # used from the thread that created them. So ONE dedicated browser
        # thread owns the context for the whole GUI lifetime and processes
        # commands (sign-in / start / stop) from this queue. Nothing else ever
        # launches a context, so the profile is locked exactly once.
        self._cmd_queue: queue.Queue = queue.Queue()
        self._pw = None               # Playwright driver, owned by the browser thread only
        self._context = None          # owned by the browser thread only
        self._page = None             # owned by the browser thread only
        self._signed_in = False       # set True after a successful sign-in while the context stays open
        self._browser_thread = threading.Thread(target=self._browser_loop, daemon=True)
        self._browser_thread.start()
        # Playwright's sync API is thread-bound: the context can ONLY be closed
        # on the browser thread that created it. So ALL closes happen on
        # _browser_thread — never from the Tk thread or atexit/signal handlers.
        # "Restart to apply" just sets this flag; the browser thread acts on it
        # (in run_loop's idle branch when running, or via the queue when stopped).
        self._restart_requested = threading.Event()
        # The browser thread closes the context before a self-update re-exec.
        # agent._apply_and_restart calls this, and it only ever runs on the
        # browser thread, so the close is always same-thread.
        agent.set_browser_closer(self._close_context)

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(200, self._drain_events)

    # -------------------------------------------------- UI
    def _build_ui(self):
        pad = {"padx": 12, "pady": 4}
        frm = ttk.Frame(self.root)
        frm.pack(fill="both", expand=True, padx=8, pady=8)

        ttk.Label(frm, text="Server URL").grid(row=0, column=0, sticky="w", **pad)
        ttk.Entry(frm, textvariable=self.server_var, width=40).grid(row=0, column=1, columnspan=2, sticky="we", **pad)

        ttk.Label(frm, text="Email").grid(row=1, column=0, sticky="w", **pad)
        ttk.Entry(frm, textvariable=self.email_var, width=40).grid(row=1, column=1, columnspan=2, sticky="we", **pad)

        ttk.Label(frm, text="Password").grid(row=2, column=0, sticky="w", **pad)
        ttk.Entry(frm, textvariable=self.password_var, width=40, show="*").grid(row=2, column=1, columnspan=2, sticky="we", **pad)

        btns = ttk.Frame(frm)
        btns.grid(row=3, column=0, columnspan=3, sticky="we", pady=(10, 4))
        self.signin_btn = ttk.Button(btns, text="Sign in to ChatGPT", command=self._on_signin)
        self.signin_btn.pack(side="left", padx=4)
        self.start_btn = ttk.Button(btns, text="Start", command=self._on_start)
        self.start_btn.pack(side="left", padx=4)
        self.stop_btn = ttk.Button(btns, text="Stop", command=self._on_stop, state="disabled")
        self.stop_btn.pack(side="left", padx=4)

        self.status_dot = tk.Canvas(frm, width=14, height=14, highlightthickness=0)
        self.status_dot.grid(row=4, column=0, sticky="e", pady=(12, 4))
        self._dot = self.status_dot.create_oval(2, 2, 12, 12, fill="#9ca3af", outline="")
        self.status_var = tk.StringVar(value="Stopped")
        ttk.Label(frm, textvariable=self.status_var).grid(row=4, column=1, columnspan=2, sticky="w", pady=(12, 4))

        self.hint = ttk.Label(frm, text="Enter your token, sign in, then Start. The window minimises to the tray.",
                              foreground="#6b7280", wraplength=400)
        self.hint.grid(row=5, column=0, columnspan=3, sticky="w", padx=12, pady=(8, 0))

        # --- Self-update row (hidden until there is something to say) ---------
        self.update_frame = ttk.Frame(frm)
        self.update_frame.grid(row=6, column=0, columnspan=3, sticky="we", padx=12, pady=(6, 0))
        self.update_var = tk.StringVar(value="")
        self.update_label = ttk.Label(self.update_frame, textvariable=self.update_var, wraplength=320)
        self.update_label.pack(side="left")
        self.restart_btn = ttk.Button(self.update_frame, text="Restart to apply", command=self._on_restart_update)
        # Not packed until an update is actually staged.
        self.update_frame.grid_remove()

        # Always-visible running code version — the one place a designer can read
        # it off without opening code_version.txt.
        ttk.Label(frm, text=f"Code version v{self._code_version}",
                  foreground="#9ca3af").grid(row=7, column=0, columnspan=3,
                                             sticky="w", padx=12, pady=(10, 0))

        # --- Browser profile location: shown so a designer/support can verify
        #     where the ChatGPT session is stored, and open it directly. ---
        prof_row = ttk.Frame(frm)
        prof_row.grid(row=8, column=0, columnspan=3, sticky="we", padx=12, pady=(8, 0))
        try:
            from src.self_update import profiles_root
            self._profile_path = str(profiles_root() / "acct1")
        except Exception:
            self._profile_path = "(unknown)"
        ttk.Label(prof_row, text="ChatGPT session:", foreground="#6b7280").pack(side="left")
        ttk.Button(prof_row, text="Show profile folder",
                   command=self._open_profile_folder).pack(side="right")
        ttk.Label(frm, text=self._profile_path, foreground="#9ca3af",
                  wraplength=400).grid(row=9, column=0, columnspan=3,
                                       sticky="w", padx=12, pady=(0, 0))

        # --- Warning banner: the exe is running from a temp/zip-extraction path,
        #     so the profile can never persist. Only shown when detected. ---
        try:
            from src.self_update import running_from_temp_location
            temp_run = running_from_temp_location()
        except Exception:
            temp_run = False
        if temp_run:
            ttk.Label(
                frm,
                text=("⚠ The agent is running from a temporary folder (opened "
                      "straight from the .zip). Your ChatGPT login will NOT be "
                      "remembered. Extract the agent to a normal folder (e.g. "
                      "Documents) and run it from there."),
                foreground="#b45309", wraplength=400,
            ).grid(row=10, column=0, columnspan=3, sticky="w", padx=12, pady=(8, 0))

        # --- Errors area: the last 5 errors + buttons to open the log folder
        #     and copy the most recent full traceback to the clipboard. Fed from
        #     the agent_logging errors ring (WARNING+). ---
        err_head = ttk.Frame(frm)
        err_head.grid(row=11, column=0, columnspan=3, sticky="we", padx=12, pady=(14, 0))
        ttk.Label(err_head, text="Recent errors", foreground="#6b7280").pack(side="left")
        ttk.Button(err_head, text="Open log folder", command=self._open_log_folder).pack(side="right")
        ttk.Button(err_head, text="Copy last error", command=self._copy_last_error).pack(side="right", padx=(0, 6))

        # Read-only, small, monospace-ish box showing the last 5 one-line errors.
        self.errors_text = tk.Text(frm, height=5, width=52, wrap="none",
                                   font=("Consolas", 8), background="#f9fafb",
                                   foreground="#374151", relief="solid", borderwidth=1)
        self.errors_text.grid(row=12, column=0, columnspan=3, sticky="we", padx=12, pady=(4, 0))
        self.errors_text.configure(state="disabled")
        self._errors_sig = None   # last rendered ring signature, to avoid needless redraws
        self._last_err_count = 0

        frm.columnconfigure(1, weight=1)
        self._refresh_errors()

    def _set_state(self, state: str, detail: str = ""):
        self._state = state
        colors = {"connected": "#16a34a", "running": "#16a34a", "not_signed_in": "#d97706",
                  "error": "#dc2626", "stopped": "#9ca3af"}
        self.status_dot.itemconfig(self._dot, fill=colors.get(state, "#9ca3af"))
        labels = {"connected": "Connected — waiting for jobs", "running": "Running a job",
                  "not_signed_in": "Not signed in to ChatGPT", "error": "Error", "stopped": "Stopped"}
        line = detail or labels.get(state, state)
        self.status_var.set(line)
        # The red "error" status line is a user-facing failure — log it so it
        # reaches errors.log and the Errors box, not just the window. This is the
        # single funnel every error path (sign-in, start, loop) passes through.
        if state == "error":
            _gui_log(line, level=_logging.ERROR)
        self._update_tray(state)

    # -------------------------------------------------- config
    def _persist(self):
        # Persist the server URL + email, and the password OBFUSCATED (never in
        # plain text). The password is not logged anywhere.
        save_config({"server_url": self.server_var.get().strip(),
                     "email": self.email_var.get().strip(),
                     "password_obf": obfuscate_password(self.password_var.get())})

    def _sign_in_get_token(self) -> bool:
        """Authenticate email+password against the server and store the token.

        Returns True on success. On failure shows the server's PLAIN message
        (bad credentials vs disabled account) or a connection error, and returns
        False. Persists settings (email + obfuscated password) regardless so the
        designer does not retype them next launch."""
        server = self.server_var.get().strip()
        email = self.email_var.get().strip()
        password = self.password_var.get()
        if not email or not password:
            _gui_log("Sign in: email and password are required.", level=_logging.WARNING)
            messagebox.showwarning("Sign in", "Enter your email and password first.")
            return False
        self._persist()
        try:
            self._token = agent.authenticate(server, email, password)
        except agent.AuthError as exc:
            self._token = ""
            # Log WITH the traceback first (server-reachability failures like
            # WinError 10061 carry the cause in the traceback), then show the UI.
            # _set_state below sees the same message and the throttle dedupes it.
            _gui_log(f"Server sign-in failed: {exc}", level=_logging.ERROR, exc=True)
            self._set_state("error", str(exc))
            messagebox.showerror("Sign-in failed", str(exc))
            return False
        # Configure the agent with the freshly obtained token. The display name
        # is the email (identity comes from the token; the name is just a label).
        agent.configure(server_url=server, token=self._token, name=email)
        return True

    # -------------------------------------------------- actions
    def _on_signin(self):
        # Authenticate first (email+password -> token), then open ChatGPT.
        if not self._sign_in_get_token():
            return
        self.signin_btn.config(state="disabled")
        self.status_var.set("Opening ChatGPT — sign in, then close the browser…")
        # Route to the single browser thread — never open a context on another thread.
        self._cmd_queue.put(("signin", None))

    def _on_start(self):
        # Authenticate first (email+password -> token). A disabled account or
        # wrong password is rejected here with a plain message, before any work.
        if not self._sign_in_get_token():
            return
        self._stop_event.clear()
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        # Hand the work to the single browser thread — do NOT touch the context here.
        self._cmd_queue.put(("start", None))

    def _on_stop(self):
        # Ask the claim loop to stop, THEN fully tear the browser down on its
        # owning thread. A full teardown (context + Playwright driver) lets a
        # designer recover from a wedged session with Stop -> Start, without
        # killing the whole process. The next Start relaunches cleanly.
        self._stop_event.set()
        self.stop_btn.config(state="disabled")
        self.start_btn.config(state="normal")
        self._set_state("stopped", "Stopped")
        self._cmd_queue.put(("teardown", None))

    # -------------------------------------------------- single browser thread
    def _browser_loop(self):
        """Owns the ONE Playwright context/page for the GUI's whole lifetime.

        All browser calls (launch, session check, sign-in, claim loop) happen
        here, on this one thread, so Playwright's thread affinity is respected
        and the profile is locked exactly once.
        """
        while True:
            try:
                cmd, _ = self._cmd_queue.get()
            except Exception:
                continue
            if cmd == "quit":
                self._close_context()
                return
            elif cmd == "restart":
                # Restart requested while the agent is STOPPED (we are idle in
                # this loop, so no job is running). Apply the staged update on
                # THIS owning thread: _apply_and_restart closes the context and
                # re-execs. If nothing is staged / apply fails, we fall through.
                self._do_restart_when_idle()
            elif cmd == "signin":
                self._do_signin()
            elif cmd == "start":
                self._do_start()
            elif cmd == "teardown":
                # Full teardown on the OWNING thread so a designer can recover
                # (Stop) without killing the process. Next signin/start relaunches.
                self._close_context()
                self._events.put(("log", "browser fully torn down; ready to start again"))

    def _do_restart_when_idle(self):
        """Apply a staged update + re-exec, running ON THE BROWSER THREAD (the
        stopped-agent path, reached via the 'restart' queue command). Closing
        the context happens inside agent._apply_and_restart via the registered
        browser-closer, so it is same-thread. Returns (without re-exec) if
        nothing is staged or the apply fails."""
        try:
            if not agent.self_update.has_staged_update():
                self._events.put(("log", "restart requested but no staged update to apply"))
                self._events.put(("update", {"reason": "up_to_date"}))
                return
            self._events.put(("log", "applying staged update on browser thread (agent stopped)…"))
            # _apply_and_restart closes the context (browser-closer) then re-execs.
            agent._apply_and_restart(
                log=lambda m: self._events.put(("log", m)),
                status=lambda s, d="": self._events.put(("status", (s, d))),
            )
            # If we reach here, apply did NOT re-exec (nothing staged / failed).
            self._events.put(("log", "restart did not occur (apply failed or nothing staged)"))
        except Exception as exc:
            self._events.put(("log", f"restart-when-idle error: {exc}"))

    def _ensure_context(self):
        """Launch the context exactly once; reuse it forever after.

        Holds the Playwright driver (self._pw) so teardown can stop it — not
        stopping it leaks the sync event loop onto this thread and breaks the
        next launch."""
        if self._context is None:
            self._pw, self._context, self._page = agent.open_browser_context()
        return self._page

    def _do_signin(self):
        try:
            first_time = self._context is None
            page = self._ensure_context()
            if not first_time:
                # Context already open — just surface it and reload ChatGPT.
                try:
                    page.bring_to_front()
                    page.goto("https://chatgpt.com", wait_until="domcontentloaded")
                except Exception:
                    pass
            logged_in = agent.is_logged_in(page)
            self._signed_in = bool(logged_in)
            # Prove whether the session reached disk: log the Cookies DB stat
            # right after a sign-in. Compare this with the pre-launch line on
            # the NEXT start to confirm the session persisted.
            try:
                from src.self_update import cookies_file_stat
                self._events.put(("log", f"cookies (post-signin): {cookies_file_stat('acct1')}"))
            except Exception:
                pass
            self._events.put(("signin_done", logged_in))
        except Exception as exc:
            traceback.print_exc()
            _gui_log(f"ChatGPT sign-in failed: {exc}", level=_logging.ERROR, exc=True)
            # A failed sign-in must NOT leave a half-open context / leaked
            # Playwright driver on this thread — that is what made every retry
            # fail with 'Sync API inside the async loop'. Tear everything down
            # here (we are on the browser thread) so the next attempt is clean.
            self._close_context()
            self._events.put(("signin_error", str(exc)))

    def _do_start(self):
        try:
            # Apply any update staged by a previous session BEFORE claiming a
            # job (provably idle here). On success this re-execs and never
            # returns; on failure we keep the current code and carry on.
            try:
                if agent.self_update.has_staged_update():
                    self._events.put(("log", "Applying staged update before start…"))
                    res = agent.apply_pending_update()  # re-execs on success; returns dict on failure
                    # If we're still here, the apply did NOT re-exec — surface why
                    # rather than starting silently on the old code.
                    if isinstance(res, dict) and res.get("error"):
                        self._events.put(("log", f"staged update did not apply: {res.get('error')}"))
                        self._events.put(("update", {"reason": "error", "error":
                            "A downloaded update could not be applied — still on the current version. " + str(res.get("error"))}))
            except Exception as exc:
                self._events.put(("log", f"startup update skipped: {exc}"))

            cold = self._context is None
            page = self._ensure_context()
            # Only check the session when starting cold (no prior sign-in on the
            # still-open context). After a successful sign-in we trust it and
            # proceed, so Start cannot flip green -> amber.
            if self._signed_in and not cold:
                logged_in = True
            else:
                logged_in = agent.is_logged_in(page)
                self._signed_in = bool(logged_in)

            self._agent_id = agent.register(logged_in)
            self._events.put(("log", f"Registered as {self._agent_id}"))
            agent.run_loop(
                page, self._agent_id, stop_event=self._stop_event,
                restart_requested=self._restart_requested,
                on_status=lambda s, d="": self._events.put(("status", (s, d))),
                on_log=lambda m: self._events.put(("log", m)),
                # GUI does NOT auto-restart: a discovered update is staged while
                # idle and the designer clicks Restart to apply it. auto_restart
                # stays False so an update never interrupts the session silently.
                on_update=lambda st: self._events.put(("update", st)),
                auto_restart=False,
            )
            self._events.put(("status", ("stopped", "Stopped")))
        except Exception as exc:
            traceback.print_exc()
            _gui_log(f"Start failed (browser launch / profile / register): {exc}",
                     level=_logging.ERROR, exc=True)
            self._events.put(("status", ("error", str(exc))))

    def _close_context(self):
        """FULL teardown: close the context AND stop the Playwright driver.

        Runs only on the browser thread. Stopping pw is what actually clears the
        sync driver's event loop from this thread; closing the context alone
        leaves it attached and the next launch fails with the async-loop error.
        Always resets state so a subsequent launch starts from a clean slate,
        even if one of the teardown steps raised."""
        try:
            if self._context is not None:
                try:
                    self._context.close()
                except Exception:
                    pass
            if self._pw is not None:
                try:
                    self._pw.stop()
                except Exception:
                    pass
        finally:
            self._pw = None
            self._context = None
            self._page = None
            self._signed_in = False
            # Clear browser.py's module-level live refs so a later atexit/signal
            # _close_live sees nothing to do (avoids a redundant cross-thread
            # close attempt after we already closed on the owning thread).
            try:
                from src import browser as _browser_mod
                _browser_mod._clear_live_refs()
            except Exception:
                pass

    # -------------------------------------------------- event pump
    def _drain_events(self):
        try:
            while True:
                kind, payload = self._events.get_nowait()
                if kind == "status":
                    s, d = payload
                    self._set_state(s, d)
                elif kind == "log":
                    print(f"[gui] {payload}")
                elif kind == "signin_done":
                    self.signin_btn.config(state="normal")
                    if payload:
                        self._set_state("connected", "Signed in to ChatGPT")
                    else:
                        self._set_state("not_signed_in", "Not signed in — try again")
                elif kind == "signin_error":
                    self.signin_btn.config(state="normal")
                    self._set_state("error", f"Sign-in failed: {payload}")
                elif kind == "update":
                    self._on_update_event(payload)
        except queue.Empty:
            pass
        # Refresh the Errors area from the logging ring (cheap; only redraws on
        # change). Driven here so it updates without its own timer.
        self._refresh_errors()
        self.root.after(200, self._drain_events)

    # -------------------------------------------------- self-update
    def _on_update_event(self, st: dict):
        """Reflect an update-status dict from the agent's idle self-update.

        Presentation is driven by `reason`, not by string-matching, so the tone
        matches what actually happened while a job keeps running fine:
          staged        -> amber "Update available — restart to apply" + button
          error         -> red (a real update problem worth attention)
          unreachable   -> MUTED grey note (agent works; only the check failed)
          not_supported -> say NOTHING (older server without the endpoints)
          up_to_date/ok -> hide the row
        Red is reserved for failures that actually stop work — never for a
        background update check that couldn't reach the server."""
        MUTED = "#6b7280"
        AMBER = "#b45309"
        RED = "#dc2626"
        reason = st.get("reason") or ("ok" if not st.get("error") else "error")

        def hide_restart():
            try:
                self.restart_btn.pack_forget()
            except Exception:
                pass

        if st.get("staged"):
            sv = st.get("server_version") or "new"
            self.update_frame.grid()
            self.update_var.set(f"Update available ({sv}) — restart to apply.")
            self.update_label.config(foreground=AMBER)
            try:
                self.restart_btn.pack(side="left", padx=(8, 0))
            except Exception:
                pass
            return

        if reason == "not_supported":
            # An older server that doesn't offer self-update is not an error;
            # show the designer nothing at all.
            hide_restart()
            self.update_frame.grid_remove()
            return

        if reason == "unreachable":
            # The agent is working; only the update check failed. Muted, calm.
            hide_restart()
            self.update_frame.grid()
            self.update_var.set("Couldn't check for updates right now — still running the current version.")
            self.update_label.config(foreground=MUTED)
            return

        if reason == "error" or st.get("error"):
            # A genuine update problem (bad download / apply failed). Worth
            # attention — but the agent keeps its current, working code.
            hide_restart()
            self.update_frame.grid()
            self.update_var.set(f"Update problem — still running current version. {st.get('error') or ''}".strip())
            self.update_label.config(foreground=RED)
            return

        # up_to_date / ok with nothing staged — nothing to act on.
        hide_restart()
        self.update_frame.grid_remove()

    def _on_restart_update(self):
        """Request a restart to apply the staged update — WITHOUT touching the
        browser context on the Tk thread (Playwright is thread-bound).

        The actual close + re-exec always happen on the BROWSER thread:
          * agent RUNNING  -> set self._restart_requested; run_loop's idle branch
            applies it after the current job finishes (jobs are never cut off).
          * agent STOPPED  -> the browser loop isn't in run_loop, so post a
            'restart' command to its queue; it applies on the browser thread.
        Either way the Tk thread only signals; it never closes or re-execs."""
        self.restart_btn.config(state="disabled")
        running = self._state in ("connected", "running", "not_signed_in")
        if running:
            self._restart_requested.set()
            self.update_var.set("Restarting to apply update (after the current job, if any)…")
        else:
            self._cmd_queue.put(("restart", None))
            self.update_var.set("Restarting to apply update…")
        self.update_label.config(foreground="#b45309")

    # -------------------------------------------------- tray
    def _ensure_tray(self):
        if self._tray is not None:
            return
        try:
            import pystray
            menu = pystray.Menu(
                pystray.MenuItem("Show window", self._tray_show, default=True),
                pystray.MenuItem("Stop agent", lambda: self._on_stop()),
                pystray.MenuItem("Quit", self._tray_quit),
            )
            self._tray = pystray.Icon("artwork_agent", _make_icon("#9ca3af"), "Artwork Agent", menu)
            threading.Thread(target=self._tray.run, daemon=True).start()
        except Exception as exc:
            print(f"[gui] tray unavailable: {exc}")

    def _update_tray(self, state: str):
        if self._tray is None:
            return
        try:
            self._tray.icon = _make_icon(_TRAY_COLORS.get(state, "#9ca3af"))
            self._tray.title = f"Artwork Agent — {state}"
        except Exception:
            pass

    def _tray_show(self, *_):
        self.root.after(0, self._restore_window)

    def _restore_window(self):
        self.root.deiconify()
        self.root.state("normal")
        self.root.lift()

    def _tray_quit(self, *_):
        self._stop_event.set()
        # Close the context on its owning (browser) thread.
        self._cmd_queue.put(("quit", None))
        if self._tray:
            self._tray.stop()
        self.root.after(0, self.root.destroy)

    # -------------------------------------------------- errors area
    def _refresh_errors(self):
        """Redraw the Errors box from the agent_logging ring (last 5). Cheap and
        idempotent — only rewrites when the ring actually changed, so it is safe
        to call from the 200ms event pump."""
        if _agent_logging is None or not hasattr(self, "errors_text"):
            return
        try:
            errs = _agent_logging.last_errors(5)
        except Exception:
            errs = []
        sig = tuple((e.get("time"), e.get("job_id"), e.get("message")) for e in errs)
        if sig == self._errors_sig:
            return
        self._errors_sig = sig
        self._last_err_count = len(errs)
        if errs:
            lines = []
            for e in errs:
                jid = (e.get("job_id") or "")[:8]
                jtag = f" job {jid}" if jid else ""
                lines.append(f"{e.get('time','')}{jtag}  {e.get('message','')}")
            text = "\n".join(lines)
        else:
            text = "No errors logged this session."
        try:
            self.errors_text.configure(state="normal")
            self.errors_text.delete("1.0", "end")
            self.errors_text.insert("1.0", text)
            self.errors_text.see("end")
            self.errors_text.configure(state="disabled")
        except Exception:
            pass

    def _open_log_folder(self):
        """Open the resolved log folder (logs/agent.log + logs/errors.log)."""
        try:
            path = _agent_logging.log_dir() if _agent_logging is not None else Path("logs")
            Path(path).mkdir(parents=True, exist_ok=True)
            if sys.platform.startswith("win"):
                os.startfile(str(path))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                import subprocess
                subprocess.Popen(["open", str(path)])
            else:
                import subprocess
                subprocess.Popen(["xdg-open", str(path)])
        except Exception as exc:
            messagebox.showinfo("Log folder",
                                f"Log folder:\n{path}\n\n(Could not open it automatically: {exc})")

    def _copy_last_error(self):
        """Copy the most recent error's FULL traceback to the clipboard."""
        full = ""
        try:
            if _agent_logging is not None:
                full = _agent_logging.last_error_full() or ""
        except Exception:
            full = ""
        if not full:
            messagebox.showinfo("Copy last error", "No error has been logged this session.")
            return
        try:
            self.root.clipboard_clear()
            self.root.clipboard_append(full)
            # Keep it on the clipboard after the window is interacted with.
            self.root.update()
            messagebox.showinfo("Copy last error",
                                "The last error's full details were copied to the clipboard.")
        except Exception as exc:
            messagebox.showinfo("Copy last error", f"Could not copy: {exc}")

    def _open_profile_folder(self):
        """Open the browser-profile folder in Explorer so support can inspect a
        designer's saved session (and whether Cookies exists) directly."""
        try:
            from src.self_update import profiles_root
            path = profiles_root() / "acct1"
            path.mkdir(parents=True, exist_ok=True)
            if sys.platform.startswith("win"):
                os.startfile(str(path))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                import subprocess
                subprocess.Popen(["open", str(path)])
            else:
                import subprocess
                subprocess.Popen(["xdg-open", str(path)])
        except Exception as exc:
            messagebox.showinfo("Profile folder", f"Profile location:\n{self._profile_path}\n\n({exc})")

    def _on_close(self):
        # Minimise to tray instead of quitting.
        self._ensure_tray()
        self.root.withdraw()


def main():
    # Belt-and-braces: importing agent already set logging up, but call it here
    # too (idempotent) so the log files + excepthooks are in place even if the
    # import order ever changes. Never let a logging problem stop the GUI.
    try:
        if _agent_logging is not None:
            _agent_logging.setup_logging(route_print=True)
            _agent_logging.install_excepthooks()
    except Exception:
        pass
    root = tk.Tk()
    gui = AgentGUI(root)
    gui._set_state("stopped", "Stopped")
    root.mainloop()


if __name__ == "__main__":
    main()
