# Per-workflow ChatGPT project. Each workflow starts its chat in its own project
# so the history stays separated by kind of work (and gives cleanly separated
# training data later). Fill in each project URL; an empty value falls back to
# PROJECT_URL_DEFAULT (a plain chat, no project).
PROJECT_URLS = {
    "text":    "https://chatgpt.com/g/g-p-6aa2c809a71c8191bed7d2f18478b929/project",   # DTF - Text Designs
    "mockup":  "https://chatgpt.com/g/g-p-6aa2c81e43bc8191818916d37702b902/project",   # DTF - Artwork Extraction
    "artwork": "https://chatgpt.com/g/g-p-6aa2c83472348191ab1ed508af3b605b/project",   # DTF - Artwork Generation
    "custom":  "https://chatgpt.com/g/g-p-6aa2c84537588191af25bf5f4d0673af/project",   # DTF - Custom Operations
    "identify": "https://chatgpt.com/g/g-p-6ab665a41ddc8191bbc7e4801d468079/project",
    "printready": "https://chatgpt.com/g/g-p-6abba3f58d948191ad2dfc2556602934/project",   # DTF - Print Ready QA
    "colorway": "https://chatgpt.com/g/g-p-6abbbc63c7d48191b3e184c30e7d45c9/project"  # DFT - Colorway
}
PROJECT_URL_DEFAULT = "https://chatgpt.com"

# The editable chat title control (pencil / "Rename" in the conversation menu).
# Used best-effort to name a chat "<client> - <task_id>"; skipped silently if
# ChatGPT's markup differs, since renaming is cosmetic.
CHAT_TITLE_INPUT = "input[aria-label='Chat title'], input[name='conversation-title']"
# ===========================================================================
# RESILIENCE NOTE — ChatGPT removed EVERY data-testid from the page (confirmed
# live: dumping all buttons showed not one has a data-testid, only aria-label;
# and [data-testid^='conversation-turn-'], article, [data-turn],
# [data-message-id], [data-message-author-role] all match 0). It also removed
# the #prompt-textarea id. So a data-testid is NOT a stable hook — treat every
# one as liable to vanish.
#
# Strategy used throughout:
#   * PRIMARY selector = a semantic attribute confirmed against the live page
#     (aria-label / role / contenteditable / a structural class fragment).
#   * FALLBACKS = the old data-testid values, kept so that IF ChatGPT ever
#     restores them nothing breaks again.
#   * For BUTTONS, the *_LABELS lists below drive a runtime resolver
#     (src.generator.find_button_by_labels) that tries each aria-label in turn
#     and LOGS which one matched — so the next markup change is diagnosable from
#     a log line rather than a hand-run console script.
# ===========================================================================

# File input: the <input type=file> is the real upload target and is confirmed
# to match (3 inputs on the live page). Keep the old testid as a fallback. The
# visible "Add files and more" control (aria-label) opens the menu, but
# set_input_files targets the hidden input directly, so input[type=file] leads.
FILE_INPUT    = "input[type='file'], input[data-testid='upload-photos-input']"

# The composer: match on SEMANTIC attributes (contenteditable + role=textbox),
# far less likely to churn than the .ProseMirror class, and keep the old id in
# case it ever returns. (Confirmed working live.)
PROMPT_BOX    = "#prompt-textarea, div[contenteditable='true'][role='textbox']"

# Stop button: confirmed live aria-label is just "Stop". Keep the older labels
# and the testid as fallbacks. STOP_BUTTON_LABELS drives the runtime resolver.
STOP_BUTTON_LABELS = ("Stop", "Stop streaming", "Stop generating")
STOP_BUTTON   = ("button[aria-label='Stop'], "
                 "button[aria-label='Stop streaming'], "
                 "button[aria-label='Stop generating'], "
                 "[data-testid='stop-button']")

IMAGE_LOADER  = "[data-testid='image-gen-loading-state']"   # not queried in code (docstring ref only)
IMAGE_OVERLAY = "[data-testid='image-gen-overlay-actions']"   # defined but currently unused
NEW_CHAT      = "[data-testid='create-new-chat-button']"      # defined but currently unused

# Generated images. ChatGPT now serves generated images as blob: URLs with
# alt="Generated image N" (confirmed live: all URL-pattern selectors matched 0,
# alt*=Generated matched 1, src was blob:https://chatgpt.com/...).
#
# PRIMARY match: the alt attribute — this is what reliably identifies a
# generated image now, regardless of how the src is served. IMPORTANT: do NOT
# match bare blob: srcs — COMPOSER_THUMBNAIL is also a blob: URL (the user's
# just-uploaded file preview), and collecting it as a result was a bug we fixed.
# The alt attribute is what distinguishes them: composer blobs have no alt text.
#
# FALLBACKS: the historical URL patterns, kept so that if ChatGPT ever switches
# back to estuary/oaiusercontent URLs they are collected without code changes.
GENERATED_IMG = ("img[alt^='Generated image'], "
                 "img[alt*=' Generated '], "
                 "[class*='DilRenderer'] img, "
                 "img[src*='backend-api/estuary/content'], "
                 "img[src*='oaiusercontent.com'], "
                 "img[src*='backend-api/content'], "
                 "img[src*='/files/']")
GENERATED_IMG_HOSTS = (
    "alt^=Generated image",           # live primary (blob: src, identified by alt)
    "backend-api/estuary/content",    # historical
    "oaiusercontent.com",             # historical
    "backend-api/content",            # historical
    "/files/",                        # historical
)

# Composer thumbnails are blob: URLs for files the user just attached. They must
# NOT be collected as generated-image results — the upload confirmation logic
# uses this to detect that the file reached the composer; the image collector
# uses alt-attribute matching (GENERATED_IMG above) to avoid picking these up.
COMPOSER_THUMBNAIL = "img[src^='blob:']"

# Buttons that remove a file already attached to the composer. Used to clear
# leftover attachments from a previous turn before attaching new files (a stale
# attachment made the upload count 2 when 1 was expected). aria-label based with
# common variants; harmless if none match.
COMPOSER_REMOVE_ATTACHMENT = ("button[aria-label='Remove file'], "
                              "button[aria-label='Remove attachment'], "
                              "button[aria-label*='Remove file'], "
                              "button[aria-label*='Remove attachment']")

# Send button: only exists once the composer has text, so its exact live
# aria-label was not captured. The runtime resolver tries SEND_BUTTON_LABELS in
# order and logs which matched; the CSS string is the fallback when none of the
# labels are present (or the testid returns). The "aria-label*='Send'" catch-all
# is last so any "Send…" variant still resolves.
SEND_BUTTON_LABELS = ("Send prompt", "Send message", "Send")
SEND_BUTTON   = ("button[aria-label='Send prompt'], "
                 "button[aria-label='Send message'], "
                 "button[aria-label='Send'], "
                 "button[aria-label*='Send'], "
                 "[data-testid='send-button']")

# The "Add files and more" composer control (aria-label confirmed live). Not
# used for the actual upload (that targets the hidden input above) but kept as a
# candidate list for the diagnostic / future menu-driven upload.
UPLOAD_LABELS = ("Add files and more", "Add photos & files", "Attach files", "Upload files")

# A single prompt/response TURN. No data-testid exists any more; the live page
# has 6 turn elements matching a class fragment inside <main>. Match on the
# class fragment SCOPED TO main (so composer/sidebar chrome that also uses
# "message"/"turn" classes is excluded), and keep the old testid as a fallback
# in case it returns. Kept at TURN granularity — code scopes "the last turn" to
# collect only that turn's images.
#
# NOTE: the bare `main [class*='turn']` was REMOVED — it matched the
# "turn-action-controls" bar (the Copy / like / dislike buttons under a reply),
# which has NO text, so "the last turn" resolved to that empty bar and the reply
# read back as 0 chars. The remaining selectors target the real turn container.
CONVERSATION_TURN = ("[data-testid^='conversation-turn-'], "
                     "main [class*='conversation-turn'], "
                     "main article[class*='turn']")

# The per-reply action bar (Copy / read-aloud / like / dislike). ChatGPT renders
# it once per assistant turn with a class fragment "turn-action-controls". It is
# the most reliable anchor we have for "where a reply ends": from the LAST one we
# find its Copy button (COPY_BUTTON, below) to lift the reply as markdown, and we
# walk UP to the enclosing turn container for the DOM fallback.
TURN_ACTION_CONTROLS = ("[class*='turn-action-controls'], "
                        "main [data-testid='turn-action-bar']")

# The Copy-to-clipboard button inside a turn's action bar. aria-label 'Copy' is
# what the live page uses; keep the older testid as a fallback in case it
# returns. Clicking it puts the reply on the clipboard as MARKDOWN (fenced
# ```json blocks intact), which is exactly what the identify/box JSON parsers
# want — far more robust than scraping rendered innerText.
COPY_BUTTON = ("button[aria-label='Copy'], "
               "button[data-testid='copy-turn-action-button']")

# The ASSISTANT message body specifically (not the whole user+assistant turn).
# ChatGPT removed [data-message-author-role] ENTIRELY (0 matches even for user
# turns), so that attribute is dropped from the primary match. The reply reader
# (src.generator.get_last_text_reply) now reads via the Copy button first and a
# turn-container innerText walk second; ASSISTANT_MESSAGE remains only as a hint
# for the diagnostics dump and the last-ditch innerText path. Match, in order:
#   1. the rendered-markdown container inside main (the assistant answer body,
#      including <pre>/<code> blocks), scoped to main so composer/sidebar excluded.
ASSISTANT_MESSAGE = ("main [class*='markdown'], "
                     "main [class*='prose']")

# The rendered reply container. Live inspection showed the assistant reply text
# sits in  P.TextBase-<hash> < DIV.DilRenderer-<hash>.  The class names are
# CSS-module hashes whose SUFFIX changes between ChatGPT builds, so match only
# the STABLE prefix "DilRenderer" (never the hash). This is the most direct
# anchor for a reply body found so far; the reply reader tries it FIRST and
# accepts the last one only when the count grew across the turn (so an older
# reply is never re-read). User turns are not DilRenderer, but the reader also
# guards by rejecting any candidate whose text equals the sent prompt.
DIL_RENDERER = "[class*='DilRenderer']"

# Positive markers that the LOGGED-OUT / login screen is showing. We use these to
# confirm a real logout rather than inferring it from a missing prompt box (which
# is also missing while the editor is merely slow to render). The data-testid
# entries are dead now (all testids removed) but harmless — they simply match
# nothing. The text/href markers are what actually carry this now, so they lead;
# kept broad so a wording/markup tweak on the auth screen still matches.
LOGIN_SCREEN = (
    "a[href*='/auth/login'], "
    "a[href*='/auth/signup'], "
    "button:has-text('Log in'), "
    "button:has-text('Sign up'), "
    "a:has-text('Log in'), "
    "a:has-text('Sign up'), "
    "[data-testid='login-button'], "
    "[data-testid='mobile-login-button'], "
    "[data-testid='signup-button'], "
    "[data-testid='welcome-login-button']"
)
