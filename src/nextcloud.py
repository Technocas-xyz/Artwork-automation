"""Nextcloud WebDAV client for the shop's artwork vault.

The operator's source artwork lives in Nextcloud under `Leads 2.0/<customer>/…`.
This module is the read side of that: list a folder, search the change feed,
fetch a thumbnail, download a file. Nothing here writes to Nextcloud — pulling a
file into a job copies it, the customer's folder is never modified.

Config comes from the environment so credentials never touch the codebase:

    NEXTCLOUD_URL           e.g. https://cloud.example.com
    NEXTCLOUD_USER          bot account (an app account, not a person)
    NEXTCLOUD_APP_PASSWORD  Settings -> Security -> App password
    NEXTCLOUD_ROOT          folder the browser is scoped to (default "Leads 2.0")
    NEXTCLOUD_TIMEOUT       per-request timeout in seconds (default 20)
    NEXTCLOUD_RETRIES       transient-error retries (default 2)

Every request goes through `_request`, which adds Basic auth, a hard timeout and
retry-with-backoff on transient failures. Auth and not-found errors fail fast —
retrying those only makes the operator wait.
"""

from __future__ import annotations

import os
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import quote, unquote
from xml.sax.saxutils import escape as xml_escape, unescape as xml_unescape

import requests
from requests.auth import HTTPBasicAuth


class NextcloudError(Exception):
    """A Nextcloud request failed. `status` is what the API layer should return."""

    def __init__(self, message: str, status: int = 502):
        super().__init__(message)
        self.status = status


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NextcloudConfig:
    base_url: str = ""
    user: str = ""
    app_password: str = ""
    root: str = "Leads 2.0"
    timeout: float = 20.0
    retries: int = 2

    @property
    def configured(self) -> bool:
        return bool(self.base_url and self.user and self.app_password)

    @property
    def dav_root(self) -> str:
        return f"{self.base_url}/remote.php/dav/files/{quote(self.user, safe='@.')}"


def get_config() -> NextcloudConfig:
    return NextcloudConfig(
        base_url=os.environ.get("NEXTCLOUD_URL", "").strip().rstrip("/"),
        user=os.environ.get("NEXTCLOUD_USER", "").strip(),
        app_password=os.environ.get("NEXTCLOUD_APP_PASSWORD", "").strip(),
        root=os.environ.get("NEXTCLOUD_ROOT", "Leads 2.0").strip().strip("/") or "Leads 2.0",
        timeout=max(5.0, float(os.environ.get("NEXTCLOUD_TIMEOUT", "20") or 20)),
        retries=max(0, int(os.environ.get("NEXTCLOUD_RETRIES", "2") or 0)),
    )


# One pooled session for the whole process. The live watcher fires a request
# every few seconds, so keep-alive is worth having.
_session = requests.Session()
_session_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def encode_path(rel_path: str) -> str:
    """Percent-encode each segment but keep the slashes."""
    parts = [p for p in str(rel_path or "").strip("/").split("/") if p]
    return "/".join(quote(p, safe="") for p in parts)


def dav_url(cfg: NextcloudConfig, rel_path: str) -> str:
    enc = encode_path(rel_path)
    return f"{cfg.dav_root}/{enc}" if enc else cfg.dav_root


def safe_rel(rel_path: str, cfg: NextcloudConfig | None = None) -> str:
    """Normalise a client-supplied path and confine it to the configured root.

    Every path that reaches Nextcloud passes through here, so a crafted
    `../../` can neither escape `Leads 2.0` nor reach another user's files.
    """
    cfg = cfg or get_config()
    raw = str(rel_path or "").replace("\\", "/").strip()
    parts = [p for p in raw.split("/") if p and p != "."]
    if any(p == ".." for p in parts):
        raise NextcloudError("Path is not allowed.", 400)
    clean = "/".join(parts)
    root = cfg.root
    if not clean:
        return root
    if clean == root or clean.startswith(root + "/"):
        return clean
    # Accept paths given relative to the root as a convenience.
    return f"{root}/{clean}"


def customer_folder(rel_path: str, cfg: NextcloudConfig | None = None) -> str:
    """The `<customer>` segment of a vault path, or "" if the path is the root."""
    cfg = cfg or get_config()
    rest = rel_path[len(cfg.root):].strip("/") if rel_path.startswith(cfg.root) else rel_path.strip("/")
    return rest.split("/")[0] if rest else ""


_LEAD_PREFIX = re.compile(r"^\d{6}_")


def display_name(folder: str) -> str:
    """`260902_Jeremy_Brown` -> `Jeremy Brown`.

    Vault folders are `YYMMDD_First_Last`; the date prefix is filing metadata,
    not part of the client's name, so it is stripped for the Client field.
    """
    name = _LEAD_PREFIX.sub("", str(folder or "").strip())
    return name.replace("_", " ").strip() or str(folder or "").strip()


# ---------------------------------------------------------------------------
# Request wrapper
# ---------------------------------------------------------------------------


_TRANSIENT = (requests.Timeout, requests.ConnectionError)


def _request(cfg: NextcloudConfig, method: str, url: str, *, headers=None,
             body=None, stream: bool = False, retries: int | None = None,
             timeout: float | None = None) -> requests.Response:
    if not cfg.configured:
        raise NextcloudError(
            "Nextcloud is not configured. Set NEXTCLOUD_URL, NEXTCLOUD_USER and "
            "NEXTCLOUD_APP_PASSWORD in the .env file.", 503)

    attempts = cfg.retries if retries is None else retries
    last: Exception | None = None
    for attempt in range(attempts + 1):
        try:
            res = _session.request(
                method, url,
                auth=HTTPBasicAuth(cfg.user, cfg.app_password),
                headers=headers or {},
                data=body.encode("utf-8") if isinstance(body, str) else body,
                timeout=timeout or cfg.timeout,
                stream=stream,
                allow_redirects=True,
            )
        except _TRANSIENT as exc:
            last = exc
            if attempt < attempts:
                time.sleep(0.3 * (attempt + 1))
                continue
            raise NextcloudError(f"Nextcloud is unreachable: {exc}", 504) from exc
        except requests.RequestException as exc:
            raise NextcloudError(f"Nextcloud request failed: {exc}", 502) from exc

        if res.status_code == 401:
            raise NextcloudError("Nextcloud rejected the credentials — check NEXTCLOUD_APP_PASSWORD.", 401)
        if res.status_code == 403:
            raise NextcloudError("Nextcloud denied access — the bot user needs share access to this folder.", 403)
        if res.status_code == 404:
            raise NextcloudError("That path no longer exists in Nextcloud.", 404)
        if res.status_code == 429 or res.status_code >= 500:
            res.close()
            last = NextcloudError(f"Nextcloud returned {res.status_code}", 502)
            if attempt < attempts:
                time.sleep(0.3 * (attempt + 1))
                continue
            raise last
        return res

    raise NextcloudError(f"Nextcloud request failed: {last}", 502)


# ---------------------------------------------------------------------------
# PROPFIND / SEARCH parsing
# ---------------------------------------------------------------------------

# Nextcloud answers with hrefs keyed to the *user id* (".../files/<uid>/…") even
# when we authenticate with the e-mail alias, so an exact prefix match on
# dav_root is not reliable. Strip whatever user segment comes back.
_DAV_PREFIX = re.compile(r"^.*?/remote\.php/dav/files/[^/]+/")
_RESPONSE_BLOCK = re.compile(r"<[^>]*:response[\s>].*?</[^>]*:response>", re.S | re.I)
_COLLECTION = re.compile(r"<[^>]*:collection\s*/?>", re.I)

PROPS = """<d:getlastmodified/><d:getetag/><d:getcontenttype/><d:getcontentlength/><d:resourcetype/><oc:fileid/>"""

PROPFIND_BODY = f"""<?xml version="1.0"?>
<d:propfind xmlns:d="DAV:" xmlns:oc="http://owncloud.org/ns">
  <d:prop>{PROPS}</d:prop>
</d:propfind>"""


def _tag(block: str, name: str) -> str | None:
    """First `<*:name>…</*:name>` in the block, entity-decoded.

    Decoding matters beyond tidiness: etags come back as `&quot;abc&quot;`, and a
    filename containing `&` arrives as `&amp;`. Comparing or re-requesting the
    raw form would quietly break both the change feed and the file path.
    """
    m = re.search(rf"<[^>]*{name}[^>]*>(.*?)</[^>]*{name}>", block, re.S | re.I)
    if not m:
        return None
    return xml_unescape(m.group(1) or "", {"&quot;": '"', "&apos;": "'"}).strip() or None


def _to_epoch_ms(http_date: str | None) -> int:
    if not http_date:
        return 0
    try:
        return int(parsedate_to_datetime(http_date).timestamp() * 1000)
    except Exception:
        return 0


@dataclass
class Entry:
    path: str
    name: str
    is_dir: bool
    size: int = 0
    etag: str = ""
    mime_type: str = ""
    modified: str = ""
    modified_ms: int = 0
    fileid: str = ""

    def as_dict(self) -> dict:
        return {
            "path": self.path, "name": self.name, "is_dir": self.is_dir,
            "size": self.size, "size_kb": round(self.size / 1024, 1),
            "etag": self.etag, "mime_type": self.mime_type,
            "modified": self.modified, "modified_ms": self.modified_ms,
            "fileid": self.fileid,
        }


def parse_multistatus(xml: str) -> list[Entry]:
    entries: list[Entry] = []
    for block in _RESPONSE_BLOCK.findall(xml or ""):
        href_raw = _tag(block, "href")
        if not href_raw:
            continue
        href = unquote(href_raw)
        rel = _DAV_PREFIX.sub("", href).strip("/")
        if not rel:
            continue
        modified = _tag(block, "getlastmodified") or ""
        entries.append(Entry(
            path=rel,
            name=rel.split("/")[-1],
            is_dir=bool(_COLLECTION.search(block)),
            size=int(_tag(block, "getcontentlength") or 0),
            etag=(_tag(block, "getetag") or "").replace('"', ""),
            mime_type=_tag(block, "getcontenttype") or "",
            modified=modified,
            modified_ms=_to_epoch_ms(modified),
            fileid=_tag(block, "fileid") or "",
        ))
    return entries


# ---------------------------------------------------------------------------
# Operations
# ---------------------------------------------------------------------------


def test_connection() -> dict:
    """PROPFIND the configured root — proves credentials, reachability and share."""
    cfg = get_config()
    if not cfg.configured:
        return {
            "configured": False, "ok": False, "root": cfg.root,
            "message": "Set NEXTCLOUD_URL, NEXTCLOUD_USER and NEXTCLOUD_APP_PASSWORD in .env",
        }
    started = time.time()
    try:
        res = _request(cfg, "PROPFIND", dav_url(cfg, cfg.root),
                       headers={"Depth": "0", "Content-Type": "application/xml"},
                       body=PROPFIND_BODY)
        ok = res.status_code in (200, 207)
        return {
            "configured": True, "ok": ok, "status": res.status_code,
            "latency_ms": int((time.time() - started) * 1000),
            "base_url": cfg.base_url, "user": cfg.user, "root": cfg.root,
            "message": f"Connected to {cfg.root}" if ok else f"Unexpected status {res.status_code}",
        }
    except NextcloudError as exc:
        return {"configured": True, "ok": False, "root": cfg.root,
                "base_url": cfg.base_url, "user": cfg.user, "message": str(exc)}


def list_folder(rel_path: str = "") -> list[Entry]:
    """One folder, Depth 1. The folder itself is dropped — children only."""
    cfg = get_config()
    target = safe_rel(rel_path, cfg)
    res = _request(cfg, "PROPFIND", dav_url(cfg, target),
                   headers={"Depth": "1", "Content-Type": "application/xml"},
                   body=PROPFIND_BODY)
    if res.status_code not in (200, 207):
        raise NextcloudError(f"Listing failed with status {res.status_code}")
    return [e for e in parse_multistatus(res.text) if e.path != target]


def search_modified_since(since_ms: int, limit: int = 400, rel_root: str = "") -> list[Entry]:
    """Every file under the root modified after `since_ms`, oldest first.

    This is the change feed. Nextcloud's webhooks are registered on this server
    but never delivered (verified), and re-walking a 1 200-customer tree every
    few seconds is far too expensive — one WebDAV SEARCH answers the same
    question in a single sub-second request.
    """
    cfg = get_config()
    root = safe_rel(rel_root, cfg) if rel_root else cfg.root
    scope = "/".join(p for p in ["/files", cfg.user, root] if p).replace("//", "/")
    if not scope.startswith("/"):
        scope = "/" + scope
    stamp = datetime.fromtimestamp(max(0, since_ms) / 1000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    body = f"""<?xml version="1.0" encoding="UTF-8"?>
<d:searchrequest xmlns:d="DAV:" xmlns:oc="http://owncloud.org/ns">
  <d:basicsearch>
    <d:select><d:prop>{PROPS}</d:prop></d:select>
    <d:from><d:scope><d:href>{xml_escape(scope)}</d:href><d:depth>infinity</d:depth></d:scope></d:from>
    <d:where><d:gt><d:prop><d:getlastmodified/></d:prop><d:literal>{stamp}</d:literal></d:gt></d:where>
    <d:orderby><d:order><d:prop><d:getlastmodified/></d:prop><d:ascending/></d:order></d:orderby>
    <d:limit><d:nresults>{max(1, min(2000, int(limit)))}</d:nresults></d:limit>
  </d:basicsearch>
</d:searchrequest>"""
    res = _request(cfg, "SEARCH", f"{cfg.base_url}/remote.php/dav/",
                   headers={"Content-Type": "text/xml; charset=utf-8"}, body=body)
    if res.status_code not in (200, 207):
        raise NextcloudError(f"Change search failed with status {res.status_code}")
    return parse_multistatus(res.text)


def download_file(rel_path: str, max_bytes: int = 0) -> tuple[bytes, str]:
    """Fetch a file's bytes. Returns (data, content_type).

    `max_bytes` aborts mid-stream rather than after the fact, so a mis-clicked
    2 GB video cannot fill the container's disk.
    """
    cfg = get_config()
    target = safe_rel(rel_path, cfg)
    res = _request(cfg, "GET", dav_url(cfg, target), stream=True)
    if res.status_code not in (200, 206):
        res.close()
        raise NextcloudError(f"Download failed with status {res.status_code}")
    content_type = res.headers.get("Content-Type", "application/octet-stream")
    chunks: list[bytes] = []
    total = 0
    try:
        for chunk in res.iter_content(chunk_size=262144):
            if not chunk:
                continue
            total += len(chunk)
            if max_bytes and total > max_bytes:
                raise NextcloudError(
                    f"File is larger than the {max_bytes / (1024 * 1024):.0f} MB import limit."
                    if max_bytes >= 1024 * 1024 else
                    f"File is larger than the {max_bytes} byte import limit.", 413)
            chunks.append(chunk)
    finally:
        res.close()
    return b"".join(chunks), content_type


def preview(rel_path: str, width: int = 320, height: int = 320) -> tuple[bytes, str]:
    """Nextcloud's generated thumbnail, falling back to the original file.

    Retries are off on purpose: the preview generator answers 500 when several
    tiles are requested at once, and waiting out the retry ladder per tile is
    slower than just serving the original.
    """
    cfg = get_config()
    target = safe_rel(rel_path, cfg)
    url = (f"{cfg.base_url}/index.php/core/preview.png"
           f"?file={quote('/' + target, safe='')}&x={int(width)}&y={int(height)}&a=1&mode=cover")
    try:
        res = _request(cfg, "GET", url, retries=0, timeout=min(cfg.timeout, 10.0))
        ctype = res.headers.get("Content-Type", "")
        if res.status_code == 200 and ctype.startswith("image/"):
            return res.content, ctype
    except NextcloudError:
        pass
    return download_file(target, max_bytes=25 * 1024 * 1024)
