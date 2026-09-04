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
import threading
import traceback
from pathlib import Path

import tkinter as tk
from tkinter import ttk, messagebox

import agent  # the existing agent, reused unchanged

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
        self.root.title("Artwork Agent")
        self.root.geometry("440x300")
        self.root.resizable(False, False)

        cfg = load_config()
        self.server_var = tk.StringVar(value=cfg.get("server_url", DEFAULT_SERVER))
        self.token_var = tk.StringVar(value=cfg.get("token", ""))
        self.name_var = tk.StringVar(value=cfg.get("name", os.getenv("COMPUTERNAME", "designer")))

        self._events: queue.Queue = queue.Queue()
        self._stop_event = threading.Event()
        self._worker_thread: threading.Thread | None = None
        self._context = None
        self._page = None
        self._agent_id = None
        self._tray = None
        self._state = "stopped"

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

        ttk.Label(frm, text="Agent token").grid(row=1, column=0, sticky="w", **pad)
        ttk.Entry(frm, textvariable=self.token_var, width=40, show="*").grid(row=1, column=1, columnspan=2, sticky="we", **pad)

        ttk.Label(frm, text="Your name").grid(row=2, column=0, sticky="w", **pad)
        ttk.Entry(frm, textvariable=self.name_var, width=40).grid(row=2, column=1, columnspan=2, sticky="we", **pad)

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

        frm.columnconfigure(1, weight=1)

    def _set_state(self, state: str, detail: str = ""):
        self._state = state
        colors = {"connected": "#16a34a", "running": "#16a34a", "not_signed_in": "#d97706",
                  "error": "#dc2626", "stopped": "#9ca3af"}
        self.status_dot.itemconfig(self._dot, fill=colors.get(state, "#9ca3af"))
        labels = {"connected": "Connected — waiting for jobs", "running": "Running a job",
                  "not_signed_in": "Not signed in to ChatGPT", "error": "Error", "stopped": "Stopped"}
        self.status_var.set(detail or labels.get(state, state))
        self._update_tray(state)

    # -------------------------------------------------- config
    def _persist(self):
        save_config({"server_url": self.server_var.get().strip(),
                     "token": self.token_var.get().strip(),
                     "name": self.name_var.get().strip()})

    def _apply_config_to_agent(self):
        self._persist()
        agent.configure(server_url=self.server_var.get().strip(),
                        token=self.token_var.get().strip(),
                        name=self.name_var.get().strip())

    # -------------------------------------------------- actions
    def _on_signin(self):
        if not self.token_var.get().strip():
            messagebox.showwarning("Token needed", "Paste your agent token first.")
            return
        self._apply_config_to_agent()
        self.signin_btn.config(state="disabled")
        self.status_var.set("Opening ChatGPT — sign in, then close the browser…")
        threading.Thread(target=self._signin_worker, daemon=True).start()

    def _signin_worker(self):
        try:
            if self._context is None:
                self._context, self._page = agent.open_browser_context()
            else:
                try:
                    self._page.bring_to_front()
                    self._page.goto("https://chatgpt.com", wait_until="domcontentloaded")
                except Exception:
                    pass
            logged_in = agent.is_logged_in(self._page)
            self._events.put(("signin_done", logged_in))
        except Exception as exc:
            traceback.print_exc()
            self._events.put(("signin_error", str(exc)))

    def _on_start(self):
        if not self.token_var.get().strip():
            messagebox.showwarning("Token needed", "Paste your agent token first.")
            return
        self._apply_config_to_agent()
        self._stop_event.clear()
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self._worker_thread = threading.Thread(target=self._run_worker, daemon=True)
        self._worker_thread.start()

    def _run_worker(self):
        try:
            if self._context is None:
                self._context, self._page = agent.open_browser_context()
            logged_in = agent.is_logged_in(self._page)
            self._agent_id = agent.register(logged_in)
            self._events.put(("log", f"Registered as {self._agent_id}"))
            agent.run_loop(
                self._page, self._agent_id, stop_event=self._stop_event,
                on_status=lambda s, d="": self._events.put(("status", (s, d))),
                on_log=lambda m: self._events.put(("log", m)),
            )
            self._events.put(("status", ("stopped", "Stopped")))
        except Exception as exc:
            traceback.print_exc()
            self._events.put(("status", ("error", str(exc))))

    def _on_stop(self):
        self._stop_event.set()
        self.stop_btn.config(state="disabled")
        self.start_btn.config(state="normal")
        self._set_state("stopped", "Stopped")

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
        except queue.Empty:
            pass
        self.root.after(200, self._drain_events)

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
        if self._tray:
            self._tray.stop()
        self.root.after(0, self.root.destroy)

    def _on_close(self):
        # Minimise to tray instead of quitting.
        self._ensure_tray()
        self.root.withdraw()


def main():
    root = tk.Tk()
    gui = AgentGUI(root)
    gui._set_state("stopped", "Stopped")
    root.mainloop()


if __name__ == "__main__":
    main()
