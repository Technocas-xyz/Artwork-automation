"""Live view of the Nextcloud vault.

The operator should not have to press Refresh: a design dropped into a
customer's folder has to show up in the browser on its own. Nextcloud's
webhook_listeners app is registered on this server but its callbacks are never
delivered (verified by live test), so push cannot be relied on. Instead one
background thread asks Nextcloud a single WebDAV SEARCH every few seconds —
"every file under Leads 2.0 modified since <cursor>" — which is a sub-second
request whose usual answer is zero rows.

Every change bumps a monotonic `revision`. The browser long-polls
`/api/nextcloud/changes?since=<revision>`; the request parks on a condition
variable and returns the moment something lands, so the UI updates about as
fast as the poll interval rather than on a UI timer.
"""

from __future__ import annotations

import os
import threading
import time
from collections import deque

from src import nextcloud as nc

# The SEARCH window deliberately re-reads a little history: getlastmodified has
# one-second granularity, so a file written in the same second as the cursor
# would otherwise be missed.
OVERLAP_MS = 120_000
BATCH = 400
RECENT_MAX = 250
DIR_REV_MAX = 800
CUSTOMER_TTL = 300.0          # seconds; a new customer folder invalidates early


def _poll_interval() -> float:
    return max(2.0, float(os.environ.get("NEXTCLOUD_POLL_SECONDS", "4") or 4))


def _initial_tail_ms() -> int:
    hours = float(os.environ.get("NEXTCLOUD_INITIAL_TAIL_HOURS", "24") or 24)
    return int(max(0.0, hours) * 3600 * 1000)


class VaultWatcher:
    def __init__(self) -> None:
        self._cond = threading.Condition()
        self.revision = 0
        self.recent: deque[dict] = deque(maxlen=RECENT_MAX)   # newest first
        self.dir_rev: dict[str, int] = {}                     # folder -> revision it last changed
        self.cursor_ms = 0
        self.last_poll_at = 0.0
        self.last_error = ""
        self.started = False
        self._seen_etags: dict[str, str] = {}
        self._thread: threading.Thread | None = None
        self._customers: list[dict] = []
        self._customers_at = 0.0
        self._customers_lock = threading.Lock()

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> bool:
        if self._thread or os.environ.get("NEXTCLOUD_WATCH", "").lower() == "false":
            return False
        if not nc.get_config().configured:
            return False
        self._thread = threading.Thread(target=self._run, name="nc-watcher", daemon=True)
        self._thread.start()
        self.started = True
        return True

    def _run(self) -> None:
        first = True
        while True:
            interval = _poll_interval()
            try:
                self.poll_once(first=first)
                first = False
            except Exception as exc:            # a bad tick must not kill the thread
                self.last_error = str(exc)
                interval = max(interval, 10.0)  # back off while Nextcloud is unhappy
            time.sleep(interval)

    # -- the tick ----------------------------------------------------------

    def poll_once(self, first: bool = False) -> int:
        """One SEARCH. Returns the number of changed files it published."""
        cfg = nc.get_config()
        if not cfg.configured:
            return 0

        if not self.cursor_ms:
            # A cold start looks back a little so the feed is not empty on boot
            # and a restart cannot silently skip files that landed while down.
            self.cursor_ms = int(time.time() * 1000) - _initial_tail_ms()

        since = max(0, self.cursor_ms - OVERLAP_MS)
        try:
            entries = nc.search_modified_since(since, limit=BATCH)
            self.last_error = ""
        except nc.NextcloudError as exc:
            self.last_error = str(exc)
            raise

        self.last_poll_at = time.time()
        files = [e for e in entries if not e.is_dir and e.path]

        # Advance over everything the window returned, changed or not. Results
        # are ascending, so a batch that hits the cap resumes exactly where it
        # stopped instead of skipping the remainder.
        newest = max((e.modified_ms for e in entries if e.modified_ms), default=0)
        if newest > self.cursor_ms:
            self.cursor_ms = newest

        # The overlap re-reads the last two minutes, so most ticks hand back
        # files we already published. Remembering etags turns those into a no-op.
        changed = [e for e in files if self._seen_etags.get(e.path) != e.etag]
        self._seen_etags = {e.path: e.etag for e in files}
        if not changed:
            return 0

        # A folder that appears for the first time means a new customer.
        touched_customers = {nc.customer_folder(e.path, cfg) for e in changed}
        known = {c["folder"] for c in self._customers}
        if self._customers and (touched_customers - known - {""}):
            self._customers_at = 0.0

        with self._cond:
            self.revision += 1
            rev = self.revision
            for entry in changed:
                folder = entry.path.rsplit("/", 1)[0] if "/" in entry.path else cfg.root
                self.dir_rev[folder] = rev
                customer = nc.customer_folder(entry.path, cfg)
                item = entry.as_dict()
                item.update({
                    "customer": customer,
                    "customer_label": nc.display_name(customer),
                    "folder": folder,
                    "revision": rev,
                    # Files found by the cold-start look-back are history, not
                    # arrivals — they must not all flash "new" on first paint.
                    "is_new": not first,
                    "seen_at": time.time(),
                })
                self.recent.appendleft(item)
            self._prune_dir_rev()
            self._cond.notify_all()
        return len(changed)

    def _prune_dir_rev(self) -> None:
        if len(self.dir_rev) <= DIR_REV_MAX:
            return
        keep = sorted(self.dir_rev.items(), key=lambda kv: kv[1], reverse=True)[:DIR_REV_MAX // 2]
        self.dir_rev = dict(keep)

    # -- readers -----------------------------------------------------------

    def wait_for_change(self, since: int, timeout: float) -> int:
        """Park until the revision moves past `since`, or the timeout expires."""
        deadline = time.time() + max(0.0, timeout)
        with self._cond:
            while self.revision <= since:
                remaining = deadline - time.time()
                if remaining <= 0:
                    break
                self._cond.wait(remaining)
            return self.revision

    def snapshot(self, since: int = 0, limit: int = 40) -> dict:
        with self._cond:
            recent = [dict(item) for item in list(self.recent)[:limit]]
            changed_dirs = sorted(
                (path for path, rev in self.dir_rev.items() if rev > since))
            revision = self.revision
        return {
            "revision": revision,
            "changed_dirs": changed_dirs,
            "recent": recent,
            "new_since": sum(1 for item in recent if item["revision"] > since),
            "watching": self.started,
            "last_poll_at": self.last_poll_at,
            "stale": bool(self.started and self.last_poll_at
                          and time.time() - self.last_poll_at > _poll_interval() * 4),
            "error": self.last_error,
            "poll_seconds": _poll_interval(),
        }

    # -- customer folders --------------------------------------------------

    def customers(self, refresh: bool = False) -> list[dict]:
        """The customer folders under the root, cached.

        There are ~1 300 of them and the list changes rarely, so listing it on
        every keystroke of the dropdown filter would be pure waste. The cache is
        dropped early whenever a tick sees a folder we have not heard of.
        """
        with self._customers_lock:
            fresh = self._customers and (time.time() - self._customers_at) < CUSTOMER_TTL
            if fresh and not refresh:
                return self._customers
            cfg = nc.get_config()
            entries = nc.list_folder(cfg.root)
            items = [
                {"folder": e.name, "path": e.path, "label": nc.display_name(e.name),
                 "modified_ms": e.modified_ms}
                for e in entries if e.is_dir
            ]
            items.sort(key=lambda c: c["label"].lower())
            self._customers = items
            self._customers_at = time.time()
            return items


watcher = VaultWatcher()
