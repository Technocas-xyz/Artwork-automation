# Per-workflow ChatGPT project. Each workflow starts its chat in its own project
# so the history stays separated by kind of work (and gives cleanly separated
# training data later). Fill in each project URL; an empty value falls back to
# PROJECT_URL_DEFAULT (a plain chat, no project).
PROJECT_URLS = {
    "text":    "https://chatgpt.com/g/g-p-6aa2c809a71c8191bed7d2f18478b929/project",   # DTF - Text Designs
    "mockup":  "https://chatgpt.com/g/g-p-6aa2c81e43bc8191818916d37702b902/project",   # DTF - Artwork Extraction
    "artwork": "https://chatgpt.com/g/g-p-6aa2c83472348191ab1ed508af3b605b/project",   # DTF - Artwork Generation
    "custom":  "https://chatgpt.com/g/g-p-6aa2c84537588191af25bf5f4d0673af/project",   # DTF - Custom Operations
}
PROJECT_URL_DEFAULT = "https://chatgpt.com"

# The editable chat title control (pencil / "Rename" in the conversation menu).
# Used best-effort to name a chat "<client> - <task_id>"; skipped silently if
# ChatGPT's markup differs, since renaming is cosmetic.
CHAT_TITLE_INPUT = "input[aria-label='Chat title'], input[name='conversation-title']"

FILE_INPUT    = "input[data-testid='upload-photos-input']"
PROMPT_BOX    = "#prompt-textarea"
STOP_BUTTON   = "[data-testid='stop-button']"
IMAGE_LOADER  = "[data-testid='image-gen-loading-state']"
IMAGE_OVERLAY = "[data-testid='image-gen-overlay-actions']"
NEW_CHAT      = "[data-testid='create-new-chat-button']"

GENERATED_IMG = "img[src*='backend-api/estuary/content']"

COMPOSER_THUMBNAIL = "img[src^='blob:']"
SEND_BUTTON        = "[data-testid='send-button']"

CONVERSATION_TURN = "section[data-testid^='conversation-turn-']"
